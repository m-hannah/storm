#include "SparseDeterministicVisitingTimesHelper.h"

#include <algorithm>

#include "storm/environment/solver/SolverEnvironment.h"
#include "storm/environment/solver/TopologicalSolverEnvironment.h"
#include "storm/solver/LinearEquationSolver.h"

#include "storm/modelchecker/prctl/helper/BaierUpperRewardBoundsComputer.h"

#include "storm/utility/ProgressMeasurement.h"
#include "storm/utility/SignalHandler.h"
#include "storm/utility/constants.h"
#include "storm/utility/macros.h"
#include "storm/utility/vector.h"

#include "storm/exceptions/UnmetRequirementException.h"
#include "storm/exceptions/NotSupportedException.h"

namespace storm {
namespace modelchecker {
namespace helper {
template<typename ValueType>
SparseDeterministicVisitingTimesHelper<ValueType>::SparseDeterministicVisitingTimesHelper(storm::storage::SparseMatrix<ValueType> const& transitionMatrix)
    : _transitionMatrix(transitionMatrix), _exitRates(nullptr), _backwardTransitions(nullptr), _sccDecomposition(nullptr) {
    // Intentionally left empty
}

template<typename ValueType>
SparseDeterministicVisitingTimesHelper<ValueType>::SparseDeterministicVisitingTimesHelper(storm::storage::SparseMatrix<ValueType> const& transitionMatrix,
                                                                                          std::vector<ValueType> const& exitRates)
    : _transitionMatrix(transitionMatrix), _exitRates(&exitRates), _backwardTransitions(nullptr), _sccDecomposition(nullptr) {
    // For the CTMC case we assert that the caller actually provided the probabilistic transitions
    STORM_LOG_ASSERT(this->_transitionMatrix.isProbabilistic(), "Non-probabilistic transitions");
}

template<typename ValueType>
void SparseDeterministicVisitingTimesHelper<ValueType>::provideBackwardTransitions(storm::storage::SparseMatrix<ValueType> const& backwardTransitions) {
    STORM_LOG_WARN_COND(!_backwardTransitions, "Backward transition matrix was provided but it was already computed or provided before.");
    _backwardTransitions = &backwardTransitions;
}

template<typename ValueType>
void SparseDeterministicVisitingTimesHelper<ValueType>::provideSCCDecomposition(
    storm::storage::StronglyConnectedComponentDecomposition<ValueType> const& decomposition) {
    STORM_LOG_WARN_COND(!_sccDecomposition, "SCC Decomposition was provided but it was already computed or provided before.");
    _sccDecomposition = &decomposition;
}

template<typename ValueType>
std::vector<ValueType> SparseDeterministicVisitingTimesHelper<ValueType>::computeExpectedVisitingTimes(Environment const& env,
                                                                                                       storm::storage::BitVector const& initialStates) {
    STORM_LOG_ASSERT(!initialStates.empty(), "provided an empty set of initial states.");
    STORM_LOG_ASSERT(initialStates.size() == _transitionMatrix.getRowCount(), "Dimension mismatch.");
    ValueType const p = storm::utility::one<ValueType>() / storm::utility::convertNumber<ValueType, uint64_t>(initialStates.getNumberOfSetBits());
    std::vector<ValueType> result(_transitionMatrix.getRowCount(), storm::utility::zero<ValueType>());
    storm::utility::vector::setVectorValues(result, initialStates, p);
    computeExpectedVisitingTimes(env, result);
    return result;
}

template<typename ValueType>
std::vector<ValueType> SparseDeterministicVisitingTimesHelper<ValueType>::computeExpectedVisitingTimes(Environment const& env, uint64_t initialState) {
    STORM_LOG_ASSERT(initialState < _transitionMatrix.getRowCount(), "Invalid initial state index.");
    std::vector<ValueType> result(_transitionMatrix.getRowCount(), storm::utility::zero<ValueType>());
    result[initialState] = storm::utility::one<ValueType>();
    computeExpectedVisitingTimes(env, result);
    return result;
}

template<typename ValueType>
std::vector<ValueType> SparseDeterministicVisitingTimesHelper<ValueType>::computeExpectedVisitingTimes(Environment const& env,
                                                                                                       ValueGetter const& initialStateValueGetter) {
    std::vector<ValueType> result;
    result.reserve(_transitionMatrix.getRowCount());
    for (uint64_t s = 0; s != _transitionMatrix.getRowCount(); ++s) {
        result.push_back(initialStateValueGetter(s));
    }
    computeExpectedVisitingTimes(env, result);
    return result;
}

template<typename ValueType>
void SparseDeterministicVisitingTimesHelper<ValueType>::computeExpectedVisitingTimes(Environment const& env, std::vector<ValueType>& stateValues) {
    STORM_LOG_ASSERT(stateValues.size() == _transitionMatrix.getRowCount(), "Dimension missmatch.");
    createBackwardTransitions();
    createDecomposition(env);

    // Create auxiliary data and lambdas
    storm::storage::BitVector sccAsBitVector(stateValues.size(), false);
    auto isLeavingTransition = [&sccAsBitVector](auto const& e) { return !sccAsBitVector.get(e.getColumn()); };
    auto isExitState = [this, &isLeavingTransition](uint64_t state) {
        auto row = this->_transitionMatrix.getRow(state);
        return std::any_of(row.begin(), row.end(), isLeavingTransition);
    };
    auto isLeavingTransitionWithNonZeroValue = [&isLeavingTransition, &stateValues](auto const& e) {
        return isLeavingTransition(e) && !storm::utility::isZero(stateValues[e.getColumn()]);
    };
    auto isReachableInState = [this, &isLeavingTransitionWithNonZeroValue, &stateValues](uint64_t state) {
        if (!storm::utility::isZero(stateValues[state])) {
            return true;
        }
        auto row = this->_backwardTransitions->getRow(state);
        return std::any_of(row.begin(), row.end(), isLeavingTransitionWithNonZeroValue);
    };

    if (env.solver().getLinearEquationSolverType() == storm::solver::EquationSolverType::Topological) {

        // We only need to adapt precision if we solve each SCC separately (in topological order)
        auto sccEnv = getEnvironmentForTopologicalSolver(env);

        STORM_LOG_WARN("Solving topo");

        // TODO h new fct (stateValues=solveTopological ...)
        // We solve each SCC individually in *forward* topological order
        storm::utility::ProgressMeasurement progress("sccs");
        progress.setMaxCount(_sccDecomposition->size());
        progress.startNewMeasurement(0);
        uint64_t sccIndex = 0;
        auto sccItEnd = std::make_reverse_iterator(_sccDecomposition->begin());
        for (auto sccIt = std::make_reverse_iterator(_sccDecomposition->end()); sccIt != sccItEnd; ++sccIt) {
            auto const& scc = *sccIt;
            if (scc.size() == 1) {
                processSingletonScc(*scc.begin(), stateValues);
            } else {
                sccAsBitVector.set(scc.begin(), scc.end(), true);
                if (std::any_of(sccAsBitVector.begin(), sccAsBitVector.end(), isExitState)) {
                    // This is not a BSCC
                    auto sccResult = computeValueForNonTrivialScc(sccEnv, sccAsBitVector, stateValues);
                    storm::utility::vector::setVectorValues(stateValues, sccAsBitVector, sccResult);
                } else {
                    // This is a BSCC
                    if (std::any_of(sccAsBitVector.begin(), sccAsBitVector.end(), isReachableInState)) {
                        storm::utility::vector::setVectorValues(stateValues, sccAsBitVector, storm::utility::infinity<ValueType>());
                    } else {
                        storm::utility::vector::setVectorValues(stateValues, sccAsBitVector, storm::utility::zero<ValueType>());
                    }
                }
                sccAsBitVector.clear();
            }
            ++sccIndex;
            progress.updateProgress(sccIndex);
            if (storm::utility::resources::isTerminate()) {
                STORM_LOG_WARN("Visiting times computation aborted after analyzing " << sccIndex << "/" << this->_computedSccDecomposition->size() << " SCCs.");
                break;
            }
        }
    }
    else {
        // We solve the complete chain (not each SCC individually - adaption of precision is not necessary)
        storm::storage::BitVector bsccStates = storm::storage::BitVector(_transitionMatrix.getRowGroupCount(), false);

        STORM_LOG_WARN("Solving NOT topo");

        uint64_t sccIndex = 0;
        auto sccItEnd = std::make_reverse_iterator(_sccDecomposition->begin());
        for (auto sccIt = std::make_reverse_iterator(_sccDecomposition->end()); sccIt != sccItEnd; ++sccIt) {
            auto const& scc = *sccIt;
            sccAsBitVector.set(scc.begin(), scc.end(), true);

            auto forwardRow = _transitionMatrix.getRow(*scc.begin());
            if ((scc.size()==1 &&  (forwardRow.getNumberOfEntries() == 1 && forwardRow.begin()->getColumn() == *scc.begin())) //TODO h singleton BSCC
                    || !(std::any_of(sccAsBitVector.begin(), sccAsBitVector.end(), isExitState))) { //TODO h why does this exclude singleton Bsccs?
                // This is a BSCC, mark the states.
                bsccStates = bsccStates | sccAsBitVector;
                // We set the values of the states to infinity or 0.
                if (std::any_of(sccAsBitVector.begin(), sccAsBitVector.end(), isReachableInState)) {
                    // There is some non-zero "input".
                    storm::utility::vector::setVectorValues(stateValues, sccAsBitVector, storm::utility::infinity<ValueType>());
                } else {
                    // There is no non-zero "input".
                    storm::utility::vector::setVectorValues(stateValues, sccAsBitVector, storm::utility::zero<ValueType>());
                }

            }
            sccAsBitVector.clear();
        }
        // Compute the values for the non-BSCC states if there are some.
        if(bsccStates.size()< _transitionMatrix.getRowGroupCount()){
            auto result = computeValueForNonTrivialScc(env, ~bsccStates, stateValues);
            // TODO h warning: jacobi and gs might not converge(?)
            storm::utility::vector::setVectorValues(stateValues, ~bsccStates, result);
        }


    }

    if (isContinuousTime()) {
        // Divide with the exit rates
        // Since storm::utility::infinity<storm::RationalNumber>() is just set to some big number, we have to treat the infinity-case explicitly.
        storm::utility::vector::applyPointwise(stateValues, *_exitRates, stateValues, [](ValueType const& xi, ValueType const& yi) -> ValueType {
            return storm::utility::isInfinity(xi) ? xi : xi / yi;
        });
    }
}

template<typename ValueType>
bool SparseDeterministicVisitingTimesHelper<ValueType>::isContinuousTime() const {
    return _exitRates;
}

template<typename ValueType>
void SparseDeterministicVisitingTimesHelper<ValueType>::createBackwardTransitions() {
    if (!this->_backwardTransitions) {
        this->_computedBackwardTransitions =
            std::make_unique<storm::storage::SparseMatrix<ValueType>>(_transitionMatrix.transpose(true, false));  // will drop zeroes
        this->_backwardTransitions = this->_computedBackwardTransitions.get();
    }
}

template<typename ValueType>
void SparseDeterministicVisitingTimesHelper<ValueType>::createDecomposition(Environment const& env) {
    if (this->_sccDecomposition && !this->_sccDecomposition->hasSccDepth() && env.solver().isForceSoundness()) {
        // We are missing SCCDepths in the given decomposition.
        STORM_LOG_WARN("Recomputing SCC Decomposition because the currently available decomposition is computed without SCCDepths.");
        this->_computedSccDecomposition.reset();
        this->_sccDecomposition = nullptr;
    }

    if (!this->_sccDecomposition) {
        // The decomposition has not been provided or computed, yet.
        auto options =
            storm::storage::StronglyConnectedComponentDecompositionOptions().forceTopologicalSort().computeSccDepths(env.solver().isForceSoundness());
        this->_computedSccDecomposition =
            std::make_unique<storm::storage::StronglyConnectedComponentDecomposition<ValueType>>(this->_transitionMatrix, options);
        this->_sccDecomposition = this->_computedSccDecomposition.get();
    }
}

template<>
void SparseDeterministicVisitingTimesHelper<storm::RationalFunction>::createUpperBounds() const {
    STORM_LOG_THROW(false, storm::exceptions::NotSupportedException,
                    "Computing upper bounds for expected visiting times over rational functions is not supported.");
}

template<typename ValueType>
void SparseDeterministicVisitingTimesHelper<ValueType>::createUpperBounds() const {
    // compute vector that contains upperBound for each non-BSCC state

    storm::storage::BitVector sccAsBitVector(_transitionMatrix.getRowGroupCount(), false);
    auto isLeavingTransition = [&sccAsBitVector](auto const& e) { return !sccAsBitVector.get(e.getColumn()); };
    auto isExitState = [this, &isLeavingTransition](uint64_t state) {
        auto row = this->_transitionMatrix.getRow(state);
        return std::any_of(row.begin(), row.end(), isLeavingTransition);
    };

    storm::storage::BitVector bsccStates = storm::storage::BitVector(_transitionMatrix.getRowGroupCount(), false);

    std::vector<uint64_t> stateToScc(_transitionMatrix.getRowGroupCount());

    uint64_t sccIndex = 0;
    auto sccItEnd = std::make_reverse_iterator(_sccDecomposition->begin());
    for (auto sccIt = std::make_reverse_iterator(_sccDecomposition->end()); sccIt != sccItEnd; ++sccIt) {
        auto const& scc = *sccIt;

        // We mark the states which do belong to a BSCC.
        auto forwardRow = _transitionMatrix.getRow(*scc.begin());
        sccAsBitVector.set(scc.begin(), scc.end(), true);
        if ((scc.size()==1 &&  (forwardRow.getNumberOfEntries() == 1 && forwardRow.begin()->getColumn() == *scc.begin())) //TODO h singleton BSCC
            || !(std::any_of(sccAsBitVector.begin(), sccAsBitVector.end(), isExitState))) { //TODO h why does this exclude singleton Bsccs?

            // This is a BSCC
            bsccStates = bsccStates | sccAsBitVector;
        }
        sccAsBitVector.clear();

        // Save the SCC to which the state belongs.
        for (auto const& state : scc) {
            stateToScc[state] = sccIndex;
        }
        ++sccIndex;
    }

    // Compute the one-step probabilities that lead to BSCC states.
    std::vector<ValueType> probabilitiesToBottomStates = _transitionMatrix.getConstrainedRowGroupSumVector(~bsccStates, bsccStates);

    // Build the submatrix that only has the transitions between non-BSCC states.
    storm::storage::SparseMatrix<ValueType> nonBsccTransitions = _transitionMatrix.getSubmatrix(false, ~bsccStates, ~bsccStates);

    // Compute the upper bounds on EVTs for non-BSCC states (using the same state-to-scc mapping).
    std::vector<ValueType> upperBoundsNonBsccStates = storm::modelchecker::helper::BaierUpperRewardBoundsComputer<ValueType>::computeUpperBoundOnExpectedVisitingTimes(
        nonBsccTransitions, probabilitiesToBottomStates, [&stateToScc](uint64_t s) { return stateToScc[s]; });

    // Set the upper bounds for non-BSCC states.
    _upperBounds = std::vector<ValueType>(_transitionMatrix.getRowCount());
    storm::utility::vector::setVectorValues<ValueType>(_upperBounds.get(), ~bsccStates, upperBoundsNonBsccStates);
    // TODO h Ignore bounds for BSCC states as they are not used.
    storm::utility::vector::setVectorValues<ValueType>(_upperBounds.get(), bsccStates, storm::utility::infinity<ValueType>());
}


template<typename ValueType>
storm::Environment SparseDeterministicVisitingTimesHelper<ValueType>::getEnvironmentForTopologicalSolver(storm::Environment const& env) const {
    storm::Environment subEnv(env);
    subEnv.solver().setLinearEquationSolverType(env.solver().topological().getUnderlyingEquationSolverType(),
                                                env.solver().topological().isUnderlyingEquationSolverTypeSetFromDefault());


    if (env.solver().isForceSoundness()) {
        STORM_LOG_ASSERT(_sccDecomposition->hasSccDepth(), "Did not compute the longest SCC chain size although it is needed.");
        // For sound computations and if the solver has a precision, we need to increase the solver's precision that is used in an SCC.
        auto subEnvPrec = subEnv.solver().getPrecisionOfLinearEquationSolver(subEnv.solver().getLinearEquationSolverType());
        if (subEnvPrec.first.is_initialized()) {
            // The solver has a precision which needs to be increased by
            // TODO h adjust for epsilon-soundness
            subEnv.solver().setLinearEquationSolverPrecision(static_cast<storm::RationalNumber>(
                subEnvPrec.first.get() / (storm::utility::convertNumber<storm::RationalNumber>(_sccDecomposition->getMaxSccDepth()))));
        }

    }
    return subEnv;
}

template<typename ValueType>
void SparseDeterministicVisitingTimesHelper<ValueType>::processSingletonScc(uint64_t sccState, std::vector<ValueType>& stateValues) const {
    auto& stateVal = stateValues[sccState];
    auto forwardRow = _transitionMatrix.getRow(sccState);
    auto backwardRow = _backwardTransitions->getRow(sccState);
    if (forwardRow.getNumberOfEntries() == 1 && forwardRow.begin()->getColumn() == sccState) {
        // This is a BSCC. We only have to check if there is some non-zero "input"
        if (!storm::utility::isZero(stateVal) || std::any_of(backwardRow.begin(), backwardRow.end(),
                                                             [&stateValues](auto const& e) { return !storm::utility::isZero(stateValues[e.getColumn()]); })) {
            stateVal = storm::utility::infinity<ValueType>();
            STORM_LOG_WARN("o");
        }  // else stateVal = 0 (already implied by !(if-condition))
    } else {
        // This is not a BSCC. Compute the state value
        ValueType divisor = storm::utility::one<ValueType>();
        for (auto const& entry : backwardRow) {
            if (entry.getColumn() == sccState) {
                STORM_LOG_ASSERT(!storm::utility::isOne(entry.getValue()), "found a self-loop state. This is not expected");
                divisor -= entry.getValue();
            } else {
                stateVal += entry.getValue() * stateValues[entry.getColumn()];
            }
        }
        stateVal /= divisor;
    }
}

template<typename ValueType>
std::vector<ValueType> SparseDeterministicVisitingTimesHelper<ValueType>::computeValueForNonTrivialScc(storm::Environment const& env,
                                                                                                       storm::storage::BitVector const& sccAsBitVector,
                                                                                                       std::vector<ValueType> const& stateValues) const {
    // Here we assume that the SCC is not a BSCC
    // Let P be the SCC matrix. We solve the equation system
    //       x * P + b = x
    // <=> P^T * x + b = x   <- fixpoint system
    // <=> (1-P^T) * x = b   <- equation system

    // TODO We need to check if SVI works on this kind of equation system (OVI and II work)
    storm::solver::GeneralLinearEquationSolverFactory<ValueType> linearEquationSolverFactory;
    bool isFixpointFormat = linearEquationSolverFactory.getEquationProblemFormat(env) == storm::solver::LinearEquationSolverProblemFormat::FixedPointSystem;

    // Get the matrix for the equation system
    auto sccMatrix = _backwardTransitions->getSubmatrix(false, sccAsBitVector, sccAsBitVector, !isFixpointFormat);
    if (!isFixpointFormat) {
        sccMatrix.convertToEquationSystem();
    }

    // Get the vector for the equation system
    auto sccVector = storm::utility::vector::filterVector(stateValues, sccAsBitVector);
    auto valIt = sccVector.begin();
    for (auto sccState : sccAsBitVector) {
        for (auto const& entry : _backwardTransitions->getRow(sccState)) {
            if (!sccAsBitVector.get(entry.getColumn())) {
                (*valIt) += entry.getValue() * stateValues[entry.getColumn()];
            }
        }
        ++valIt;
    }

    // Get the solver object and satisfy requirements
    auto solver = linearEquationSolverFactory.create(env, std::move(sccMatrix));
    solver->setLowerBound(storm::utility::zero<ValueType>());
    auto req = solver->getRequirements(env);
    req.clearLowerBounds();
    if (req.upperBounds().isCritical()) { //TODO H is critical is correct? otherwise every method with --sound needs upper bound?
        // Compute upper bounds on EVTs using techniques from by Baier et al. [CAV'17] (https://doi.org/10.1007/978-3-319-63387-9_8)
        if (!_upperBounds.is_initialized()) {
            createUpperBounds();
        }
        auto sccUpperBounds = storm::utility::vector::filterVector(_upperBounds.get(), sccAsBitVector);
        solver->setUpperBounds(sccUpperBounds);
        req.clearUpperBounds();
    }

    STORM_LOG_THROW(!req.hasEnabledCriticalRequirement(), storm::exceptions::UnmetRequirementException,
                    "Solver requirements " + req.getEnabledRequirementsAsString() + " not checked.");
    std::vector<ValueType> eqSysValues(sccVector.size());
    solver->solveEquations(env, eqSysValues, sccVector);
    return eqSysValues;
}

template class SparseDeterministicVisitingTimesHelper<double>;
template class SparseDeterministicVisitingTimesHelper<storm::RationalNumber>;
template class SparseDeterministicVisitingTimesHelper<storm::RationalFunction>;

}  // namespace helper
}  // namespace modelchecker
}  // namespace storm