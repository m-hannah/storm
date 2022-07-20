#pragma once

#include <cstdint>
#include <functional>
#include <vector>

namespace storm {
namespace storage {
template<typename ValueType>
class SparseMatrix;
}

namespace modelchecker {
namespace helper {

template<typename ValueType>
class BaierUpperRewardBoundsComputer {
   public:
    /*!
     * Creates an object that can compute
     *    * upper bounds on the *maximal* expected rewards and
     * for the provided MDP. This also works with mixtures of positive and negative rewards.
     * @see http://doi.org/10.1007/978-3-319-63387-9_8
     * @param transitionMatrix The matrix defining the transitions of the system without the transitions
     * that lead directly to the goal state.
     * @param rewards The rewards of each choice.
     * @param oneStepTargetProbabilities For each choice the probability to go to a goal state in one step.
     */
    BaierUpperRewardBoundsComputer(storm::storage::SparseMatrix<ValueType> const& transitionMatrix, std::vector<ValueType> const& rewards,
                                   std::vector<ValueType> const& oneStepTargetProbabilities, std::function<uint64_t(uint64_t)> const& stateToScc = {});

    /*!
     * Computes an upper bound on the expected rewards.
     * This also works when there are mixtures of positive and negative rewards present.
     */
    ValueType computeUpperBound();

    /*!
     * Computes for each state an upper bound for the maximal recurrence probability for each state.
     * @param transitionMatrix The matrix defining the transitions of the system without the transitions
     * that lead directly to the goal state.
     * @param oneStepTargetProbabilities For each choice the probability to go to a goal state in one step.
     */
    static std::vector<ValueType> computeUpperBoundOnRecurrenceProbabilities(storm::storage::SparseMatrix<ValueType> const& transitionMatrix,
                                                                           std::vector<ValueType> const& oneStepTargetProbabilities);

    /*!
     * Computes for each state an upper bound for the maximal recurrence probability for each state.
     * @param transitionMatrix The matrix defining the transitions of the system without the transitions
     * that lead directly to the goal state.
     * @param oneStepTargetProbabilities For each choice the probability to go to a goal state in one step.
     * @param stateToScc Returns the SCC index for each state
     */
    static std::vector<ValueType> computeUpperBoundOnRecurrenceProbabilities(storm::storage::SparseMatrix<ValueType> const& transitionMatrix,
                                                                           std::vector<ValueType> const& oneStepTargetProbabilities,
                                                                           std::function<uint64_t(uint64_t)> const& stateToScc);

    /*!
     * Computes for each state an upper bound for the maximal expected times each state is visited.
     * @param transitionMatrix The matrix defining the transitions of the system without the transitions
     * that lead directly to the goal state.
     * @param oneStepTargetProbabilities For each choice the probability to go to a goal state in one step.
     */
    static std::vector<ValueType> computeUpperBoundOnExpectedVisitingTimes(storm::storage::SparseMatrix<ValueType> const& transitionMatrix,
                                                                           std::vector<ValueType> const& oneStepTargetProbabilities);

    /*!
     * Computes for each state an upper bound for the maximal expected times each state is visited.
     * @param transitionMatrix The matrix defining the transitions of the system without the transitions
     * that lead directly to the goal state.
     * @param oneStepTargetProbabilities For each choice the probability to go to a goal state in one step.
     * @param stateToScc Returns the SCC index for each state
     */
    static std::vector<ValueType> computeUpperBoundOnExpectedVisitingTimes(storm::storage::SparseMatrix<ValueType> const& transitionMatrix,
                                                                           std::vector<ValueType> const& oneStepTargetProbabilities,
                                                                           std::function<uint64_t(uint64_t)> const& stateToScc);

   private:
    storm::storage::SparseMatrix<ValueType> const& _transitionMatrix;
    std::function<uint64_t(uint64_t)> _stateToScc;
    std::vector<ValueType> const& _rewards;
    std::vector<ValueType> const& _oneStepTargetProbabilities;
};
}  // namespace helper
}  // namespace modelchecker
}  // namespace storm
