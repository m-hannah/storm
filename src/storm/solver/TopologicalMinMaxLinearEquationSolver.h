#pragma once

#include <optional>

#include "storm/solver/StandardMinMaxLinearEquationSolver.h"

#include "storm/solver/SolverSelectionOptions.h"
#include "storm/storage/StronglyConnectedComponentDecomposition.h"

namespace storm {

class Environment;

namespace solver {

template<typename ValueType>
class TopologicalMinMaxLinearEquationSolver : public StandardMinMaxLinearEquationSolver<ValueType> {
   public:
    TopologicalMinMaxLinearEquationSolver();
    TopologicalMinMaxLinearEquationSolver(storm::storage::SparseMatrix<ValueType> const& A);
    TopologicalMinMaxLinearEquationSolver(storm::storage::SparseMatrix<ValueType>&& A);

    virtual ~TopologicalMinMaxLinearEquationSolver() {}

    virtual void clearCache() const override;

    virtual MinMaxLinearEquationSolverRequirements getRequirements(Environment const& env,
                                                                   boost::optional<storm::solver::OptimizationDirection> const& direction = boost::none,
                                                                   bool const& hasInitialScheduler = false) const override;

   protected:
    virtual bool internalSolveEquations(storm::Environment const& env, OptimizationDirection d, std::vector<ValueType>& x,
                                        std::vector<ValueType> const& b) const override;

    virtual void setOneMinusRowSumVector(std::vector<ValueType> const& oneMinusRowSum) override {
        oneMinusRowSumVector = &oneMinusRowSum;
    }

   private:
    storm::Environment getEnvironmentForUnderlyingSolver(storm::Environment const& env, bool adaptPrecision = false) const;

    // Creates an SCC decomposition and sorts the SCCs according to a topological sort.
    void createSortedSccDecomposition(bool needLongestChainSize) const;

    // Solves the SCC with the given index
    // ... for the case that the SCC is trivial
    bool solveTrivialScc(uint64_t const& sccState, OptimizationDirection d, std::vector<ValueType>& globalX, std::vector<ValueType> const& globalB) const;
    // ... for the case that there is just one large SCC
    bool solveFullyConnectedEquationSystem(storm::Environment const& sccSolverEnvironment, OptimizationDirection d, std::vector<ValueType>& x,
                                           std::vector<ValueType> const& b) const;
    // ... for the remaining cases (1 < scc.size() < x.size())
    bool solveScc(storm::Environment const& sccSolverEnvironment, OptimizationDirection d, storm::storage::BitVector const& sccRowGroups,
                  storm::storage::BitVector const& sccRows, std::vector<ValueType>& globalX, std::vector<ValueType> const& globalB,
                  std::optional<storm::storage::BitVector> const& globalRelevantValues) const;

    std::vector<ValueType> computeSccExitProbabilities(storm::storage::BitVector const& sccRowGroups, storm::storage::BitVector const& sccRows) const {
        assert(this->oneMinusRowSumVector);
        std::vector<ValueType> result;
        result.reserve(sccRows.size());
        for (auto rowIndex : sccRows) {
            ValueType exitProb = (*oneMinusRowSumVector)[rowIndex];
            for (auto const& entry : this->A->getRow(rowIndex)) {
                if (!sccRowGroups.get(entry.getColumn())) {
                    exitProb += entry.getValue();
                }
            }
            result.push_back(exitProb);
        }
        return result;
    }

    // cached auxiliary data
    mutable std::unique_ptr<storm::storage::StronglyConnectedComponentDecomposition<ValueType>> sortedSccDecomposition;
    mutable boost::optional<uint64_t> longestSccChainSize;
    std::vector<ValueType> const* oneMinusRowSumVector{nullptr};
    mutable std::unique_ptr<storm::solver::MinMaxLinearEquationSolver<ValueType>> sccSolver;
    mutable std::unique_ptr<std::vector<ValueType>> auxiliaryRowGroupVector;  // A.rowGroupCount() entries
};
}  // namespace solver
}  // namespace storm
