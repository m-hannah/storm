#pragma once
#include <type_traits>
#include <vector>
#include "storm/solver/SolverStatus.h"
#include "storm/solver/helper/ValueIterationHelper.h"
#include "storm/solver/helper/ValueIterationOperator.h"
#include "storm/storage/SparseMatrix.h"
#include "storm/utility/Extremum.h"
#include "storm/utility/constants.h"
#include "storm/utility/vector.h"

namespace storm {
class Environment;

namespace solver::helper {

template<typename ValueType, bool TrivialRowGrouping = false>
class OptimisticValueIterationHelper {
   public:
    OptimisticValueIterationHelper(std::shared_ptr<ValueIterationOperator<ValueType, TrivialRowGrouping>> viOperator) : _operator(viOperator) {
        // Intentionally left empty.
    }

    template<OptimizationDirection Dir, bool Relative>
    auto OVI(std::pair<std::vector<ValueType>, std::vector<ValueType>>& vu, std::vector<ValueType> const& offsets, uint64_t& numIterations,
             ValueType const& precision, ValueType const& guessValue, std::optional<ValueType> const& lowerBound = {},
             std::optional<ValueType> const& upperBound = {},
             std::function<SolverStatus(SolverStatus const&, std::vector<ValueType> const&)> const& iterationCallback = {}) {
        if (SolverStatus status = GSVI<Dir, Relative>(vu.first, offsets, numIterations, precision, iterationCallback); status != SolverStatus::Converged) {
            return status;
        }
        guessCandidate<Relative>(vu, guessValue, lowerBound, upperBound);
        OVIBackend<Dir, Relative> backend;
        uint64_t const maxIters =
            numIterations + storm::utility::convertNumber<uint64_t, ValueType>(storm::utility::ceil<ValueType>(storm::utility::one<ValueType>() / precision));
        while (numIterations < maxIters) {
            ++numIterations;
            if (_operator->template applyInPlace(vu, offsets, backend)) {
                if (backend.allDown()) {
                    return SolverStatus::Converged;
                } else {
                    assert(backend.allUp());
                    break;
                }
            }
            if (backend.abort()) {
                break;
            }
            if (iterationCallback) {
                if (auto status = iterationCallback(SolverStatus::InProgress, vu.first); status != SolverStatus::InProgress) {
                    return status;
                }
            }
        }
        return OVI<Dir, Relative>(vu, offsets, numIterations, backend.error() / storm::utility::convertNumber<ValueType, uint64_t>(2u), guessValue, lowerBound,
                                  upperBound, iterationCallback);
    }

    template<storm::OptimizationDirection Dir>
    auto OVI(std::pair<std::vector<ValueType>, std::vector<ValueType>>& vu, std::vector<ValueType> const& offsets, uint64_t& numIterations, bool relative,
             ValueType const& precision, ValueType const& guessValue, std::optional<ValueType> const& lowerBound, std::optional<ValueType> const& upperBound,
             std::function<SolverStatus(SolverStatus const&, std::vector<ValueType> const&)> const& iterationCallback) {
        if (relative) {
            return OVI<Dir, true>(vu, offsets, numIterations, precision, guessValue, lowerBound, upperBound, iterationCallback);
        } else {
            return OVI<Dir, false>(vu, offsets, numIterations, precision, guessValue, lowerBound, upperBound, iterationCallback);
        }
    }

    auto OVI(std::pair<std::vector<ValueType>, std::vector<ValueType>>& vu, std::vector<ValueType> const& offsets, uint64_t& numIterations, bool relative,
             ValueType const& precision, std::optional<storm::OptimizationDirection> const& dir, ValueType const& guessValue,
             std::optional<ValueType> const& lowerBound, std::optional<ValueType> const& upperBound,
             std::function<SolverStatus(SolverStatus const&, std::vector<ValueType> const&)> const& iterationCallback) {
        // Catch the case where lower- and upper bound are already close enough. (when guessing candidates, OVI handles this case not very well, in particular
        // when lowerBound==upperBound)
        if (lowerBound && upperBound) {
            ValueType diff = *upperBound - *lowerBound;
            if ((relative && diff <= precision * std::min(storm::utility::abs(*lowerBound), storm::utility::abs(*upperBound))) ||
                (!relative && diff <= precision)) {
                vu.first.assign(vu.first.size(), *lowerBound);
                vu.second.assign(vu.second.size(), *upperBound);
                return SolverStatus::Converged;
            }
        }

        if (!dir.has_value() || maximize(*dir)) {
            return OVI<OptimizationDirection::Maximize>(vu, offsets, numIterations, relative, precision, guessValue, lowerBound, upperBound, iterationCallback);
        } else {
            return OVI<OptimizationDirection::Minimize>(vu, offsets, numIterations, relative, precision, guessValue, lowerBound, upperBound, iterationCallback);
        }
    }

    auto OVI(std::vector<ValueType>& operand, std::vector<ValueType> const& offsets, uint64_t& numIterations, bool relative, ValueType const& precision,
             std::optional<storm::OptimizationDirection> const& dir = {}, std::optional<ValueType> const& guessValue = {},
             std::optional<ValueType> const& lowerBound = {}, std::optional<ValueType> const& upperBound = {},
             std::function<SolverStatus(SolverStatus const&, std::vector<ValueType> const&)> const& iterationCallback = {}) {
        // Create two vectors v and u using the given operand plus an auxiliary vector.
        std::pair<std::vector<ValueType>, std::vector<ValueType>> vu;
        auto& auxVector = _operator->allocateAuxiliaryVector(operand.size());
        vu.first.swap(operand);
        vu.second.swap(auxVector);
        auto doublePrec = precision + precision;
        if constexpr (std::is_same_v<ValueType, double>) {
            doublePrec -= precision * 1e-6;  // be slightly more precise to avoid a good chunk of floating point issues
        }
        auto status =
            OVI(vu, offsets, numIterations, relative, doublePrec, dir, guessValue ? *guessValue : doublePrec, lowerBound, upperBound, iterationCallback);
        auto two = storm::utility::convertNumber<ValueType>(2.0);
        // get the average of lower- and upper result
        storm::utility::vector::applyPointwise<ValueType, ValueType, ValueType>(
            vu.first, vu.second, vu.first, [&two](ValueType const& a, ValueType const& b) -> ValueType { return (a + b) / two; });
        // Swap operand and aux vector back to original positions.
        vu.first.swap(operand);
        vu.second.swap(auxVector);
        _operator->freeAuxiliaryVector();
        return status;
    }

    auto OVI(std::vector<ValueType>& operand, std::vector<ValueType> const& offsets, bool relative, ValueType const& precision,
             std::optional<storm::OptimizationDirection> const& dir = {}, std::optional<ValueType> const& guessValue = {},
             std::optional<ValueType> const& lowerBound = {}, std::optional<ValueType> const& upperBound = {},
             std::function<SolverStatus(SolverStatus const&, std::vector<ValueType> const&)> const& iterationCallback = {}) {
        uint64_t numIterations = 0;
        return OVI(operand, offsets, numIterations, relative, precision, dir, guessValue, lowerBound, upperBound, iterationCallback);
    }

   private:
    template<bool Relative>
    static ValueType diff(ValueType const& oldValue, ValueType const& newValue) {
        if constexpr (Relative) {
            return storm::utility::abs<ValueType>((newValue - oldValue) / newValue);
        } else {
            return storm::utility::abs<ValueType>(newValue - oldValue);
        }
    }

    template<bool Relative>
    struct GSVITermCrit {
        ValueType const precision;
        bool operator()(ValueType const& oldValue, ValueType const& newValue) const {
            return storm::utility::isZero(newValue) || diff<Relative>(oldValue, newValue) <= precision;
        }
    };

    template<storm::OptimizationDirection Dir, bool Relative>
    auto GSVI(std::vector<ValueType>& operand, std::vector<ValueType> const& offsets, uint64_t& numIterations, ValueType const& precision,
              std::function<SolverStatus(SolverStatus const&, std::vector<ValueType> const&)> const& iterationCallback = {}) {
        VIOperatorBackend<ValueType, Dir, GSVITermCrit<Relative>> backend{precision};
        ValueIterationHelper<ValueType, TrivialRowGrouping> viHelper(_operator);
        auto viCallback = [&](SolverStatus const& status) { return iterationCallback(status, operand); };
        return viHelper.VI(operand, offsets, numIterations, std::move(backend), viCallback);
    }

    template<bool Relative>
    void guessCandidate(std::pair<std::vector<ValueType>, std::vector<ValueType>>& vu, ValueType const& guessValue, std::optional<ValueType> const& lowerBound,
                        std::optional<ValueType> const& upperBound) {
        std::function<ValueType(ValueType const&)> guess;
        [[maybe_unused]] ValueType factor = storm::utility::one<ValueType>() + guessValue;
        if constexpr (Relative) {
            if (lowerBound && *lowerBound < storm::utility::zero<ValueType>()) {
                guess = [&guessValue](ValueType const& val) { return val + storm::utility::abs<ValueType>(val * guessValue); };
            } else {
                guess = [&factor](ValueType const& val) { return val * factor; };
            }
        } else {
            guess = [&guessValue](ValueType const& val) { return storm::utility::isZero(val) ? storm::utility::zero<ValueType>() : val + guessValue; };
        }
        if (lowerBound || upperBound) {
            std::function<ValueType(ValueType const&)> guessAndClamp;
            if (!lowerBound) {
                guessAndClamp = [&guess, &upperBound](ValueType const& val) { return std::min(guess(val), *upperBound); };
            } else if (!upperBound) {
                guessAndClamp = [&guess, &lowerBound](ValueType const& val) { return std::max(guess(val), *lowerBound); };
            } else {
                guessAndClamp = [&guess, &lowerBound, &upperBound](ValueType const& val) { return std::clamp(guess(val), *lowerBound, *upperBound); };
            }
            storm::utility::vector::applyPointwise(vu.first, vu.second, guessAndClamp);
        } else {
            storm::utility::vector::applyPointwise(vu.first, vu.second, guess);
        }
    }

    template<OptimizationDirection Dir, bool Relative>
    class OVIBackend {
       public:
        void startNewIteration() {
            _allUp = true;
            _allDown = true;
            _crossed = false;
            _error = storm::utility::zero<ValueType>();
        }

        void firstRow(std::pair<ValueType, ValueType>&& value, [[maybe_unused]] uint64_t rowGroup, [[maybe_unused]] uint64_t row) {
            _vBest = std::move(value.first);
            _uBest = std::move(value.second);
        }

        void nextRow(std::pair<ValueType, ValueType>&& value, [[maybe_unused]] uint64_t rowGroup, [[maybe_unused]] uint64_t row) {
            assert(!TrivialRowGrouping);
            _vBest &= std::move(value.first);
            _uBest &= std::move(value.second);
        }

        void applyUpdate(ValueType& vCurr, ValueType& uCurr, [[maybe_unused]] uint64_t rowGroup) {
            if (*_vBest != storm::utility::zero<ValueType>()) {
                _error &= diff<Relative>(vCurr, *_vBest);
            }
            if (*_uBest < uCurr) {
                uCurr = *_uBest;
                _allUp = false;
            } else if (*_uBest > uCurr) {
                _allDown = false;
            }
            vCurr = *_vBest;
            if (vCurr > uCurr) {
                _crossed = true;
            }
        }

        void endOfIteration() const {
            // intentionally left empty.
        }

        bool converged() const {
            return _allDown || _allUp;
        }

        bool allUp() const {
            return _allUp;
        }

        bool allDown() const {
            return _allDown;
        }

        bool abort() const {
            return _crossed;
        }

        ValueType error() {
            return *_error;
        }

       private:
        bool _allUp{true};
        bool _allDown{true};
        bool _crossed{false};
        storm::utility::Extremum<Dir, ValueType> _vBest, _uBest;
        storm::utility::Extremum<OptimizationDirection::Maximize, ValueType> _error;
    };

    std::shared_ptr<ValueIterationOperator<ValueType, TrivialRowGrouping>> _operator;
};

}  // namespace solver::helper
}  // namespace storm