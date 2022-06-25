#include "storm/models/sparse/DeterministicModel.h"

#include "storage/StronglyConnectedComponentDecomposition.h"
#include "storm/adapters/RationalFunctionAdapter.h"
#include "storm/io/export.h"
#include "storm/models/sparse/StandardRewardModel.h"
#include "storm/utility/constants.h"
#include "utility/graph.h"

namespace storm {
namespace models {
namespace sparse {

template<typename ValueType, typename RewardModelType>
DeterministicModel<ValueType, RewardModelType>::DeterministicModel(ModelType modelType,
                                                                   storm::storage::sparse::ModelComponents<ValueType, RewardModelType> const& components)
    : Model<ValueType, RewardModelType>(modelType, components) {
    // Intentionally left empty
}

template<typename ValueType, typename RewardModelType>
DeterministicModel<ValueType, RewardModelType>::DeterministicModel(ModelType modelType,
                                                                   storm::storage::sparse::ModelComponents<ValueType, RewardModelType>&& components)
    : Model<ValueType, RewardModelType>(modelType, std::move(components)) {
    // Intentionally left empty
}

template<typename ValueType, typename RewardModelType>
void DeterministicModel<ValueType, RewardModelType>::writeDotToStream(std::ostream& outStream, size_t maxWidthLabel, bool includeLabeling,
                                                                      storm::storage::BitVector const* subsystem, std::vector<ValueType> const* firstValue,
                                                                      std::vector<ValueType> const* secondValue,
                                                                      std::vector<uint_fast64_t> const* stateColoring, std::vector<std::string> const* colors,
                                                                      std::vector<uint_fast64_t>* scheduler, bool finalizeOutput) const {
    Model<ValueType, RewardModelType>::writeDotToStream(outStream, maxWidthLabel, includeLabeling, subsystem, firstValue, secondValue, stateColoring, colors,
                                                        scheduler, false);

    // iterate over all transitions and draw the arrows with probability information attached.
    auto rowIt = this->getTransitionMatrix().begin();
    for (uint_fast64_t i = 0; i < this->getTransitionMatrix().getRowCount(); ++i, ++rowIt) {
        // Put in an intermediate node if there is a choice labeling
        std::string arrowOrigin = std::to_string(i);
        if (this->hasChoiceLabeling()) {
            arrowOrigin = "\"" + arrowOrigin + "c\"";
            outStream << "\t" << arrowOrigin << " [shape = \"point\"]\n";
            outStream << "\t" << i << " -> " << arrowOrigin << " [label= \"{";
            storm::utility::outputFixedWidth(outStream, this->getChoiceLabeling().getLabelsOfChoice(i), maxWidthLabel);
            outStream << "}\"];\n";
        }

        typename storm::storage::SparseMatrix<ValueType>::const_rows row = this->getTransitionMatrix().getRow(i);
        for (auto const& transition : row) {
            if (transition.getValue() != storm::utility::zero<ValueType>()) {
                if (subsystem == nullptr || subsystem->get(transition.getColumn())) {
                    outStream << "\t" << arrowOrigin << " -> " << transition.getColumn() << " [ label= \"" << transition.getValue() << "\" ];\n";
                }
            }
        }
    }

    if (finalizeOutput) {
        outStream << "}\n";
    }
}

template<typename ValueType, typename RewardModelType>
void DeterministicModel<ValueType, RewardModelType>::printModelInformationToStream(std::ostream& out) const {
    this->printModelInformationHeaderToStream(out);
    this->printModelInformationFooterToStream(out);

    // TODO h: remove this (?)
    // Compute information about the model's topology:
    // Create auxiliary data and lambdas
    storm::storage::BitVector sccAsBitVector(this->getNumberOfStates(), false);
    auto isLeavingTransition = [&sccAsBitVector](auto const& e) {return !sccAsBitVector.get(e.getColumn());};
    auto isExitState = [this, &isLeavingTransition](uint64_t state) {
        auto row = this->getTransitionMatrix().getRow(state);
        return std::any_of(row.begin(), row.end(), isLeavingTransition);
    };

    auto options = storm::storage::StronglyConnectedComponentDecompositionOptions().forceTopologicalSort().computeSccDepths(true);
    auto sccDecomposition = storm::storage::StronglyConnectedComponentDecomposition<ValueType>(this->getTransitionMatrix(), options);

    storm::storage::BitVector nonBsccStates = storm::storage::BitVector(this->getNumberOfStates(), false);
    uint64_t numSccs = 0;
    uint64_t maxSccSize = 0;
    for (auto const& scc : sccDecomposition) {
        sccAsBitVector.set(scc.begin(), scc.end(), true);
        if (std::any_of(sccAsBitVector.begin(), sccAsBitVector.end(), isExitState)) {
            // This is not a BSCC, mark the states correspondingly.
            nonBsccStates = nonBsccStates | sccAsBitVector;
            // Increase counter for number of non-bottom SCCs
            numSccs ++;
            // Update the maximal number of states in one SCC
            maxSccSize = std::max(maxSccSize, sccAsBitVector.getNumberOfSetBits());
        }
        sccAsBitVector.clear();
    }

    out << "# Topology of the input model without BSCCs: ";
    if (!nonBsccStates.empty() && storm::utility::graph::hasCycle(this->getTransitionMatrix().getSubmatrix(false, nonBsccStates, nonBsccStates))) {
        out << "cyclic\n";
    }
    else {
        out << "acyclic\n";
    }

    out << "# Number of non-BSCC states: " << (nonBsccStates.getNumberOfSetBits()) << '\n';

    out << "# Number of non-bottom SCCs: " << numSccs << '\n';
    out << "# Size of largest non-bottom SCC: " << maxSccSize << " states\n";


    out << "# Length of max SCC chain: " << sccDecomposition.getMaxSccDepth() << '\n';

    out << "-------------------------------------------------------------- \n";
}



template class DeterministicModel<double>;
#ifdef STORM_HAVE_CARL
template class DeterministicModel<storm::RationalNumber>;

template class DeterministicModel<double, storm::models::sparse::StandardRewardModel<storm::Interval>>;
template class DeterministicModel<storm::RationalFunction>;
#endif
}  // namespace sparse
}  // namespace models
}  // namespace storm
