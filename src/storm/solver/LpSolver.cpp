
#include "LpSolver.h"

#include "storm/storage/expressions/Expression.h"
#include "storm/storage/expressions/ExpressionManager.h"

namespace storm {
namespace solver {

template<typename ValueType>
RawLpConstraint<ValueType>::RawLpConstraint(storm::expressions::RelationType relationType, ValueType const& rhs, uint64_t reservedSize)
    : _relationType(relationType), _rhs(rhs) {
    _lhsCoefficients.reserve(reservedSize);
    _lhsVariableIndices.reserve(reservedSize);
}

template<typename ValueType>
void RawLpConstraint<ValueType>::addToLhs(VariableIndexType const& variable, ValueType const& coefficient) {
    _lhsCoefficients.push_back(coefficient);
    _lhsVariableIndices.push_back(variable);
}

template<typename ValueType, bool RawMode>
LpSolver<ValueType, RawMode>::LpSolver()
    : manager(new storm::expressions::ExpressionManager()), currentModelHasBeenOptimized(false), optimizationDirection(OptimizationDirection::Minimize) {
    // Intentionally left empty.
}

template<typename ValueType, bool RawMode>
LpSolver<ValueType, RawMode>::LpSolver(OptimizationDirection const& optimizationDir)
    : manager(new storm::expressions::ExpressionManager()), currentModelHasBeenOptimized(false), optimizationDirection(optimizationDir) {
    // Intentionally left empty.
}

template<typename ValueType, bool RawMode>
typename LpSolver<ValueType, RawMode>::Variable LpSolver<ValueType, RawMode>::addBoundedContinuousVariable(std::string const& name, ValueType lowerBound,
                                                                                                           ValueType upperBound,
                                                                                                           ValueType objectiveFunctionCoefficient) {
    return addVariable(name, VariableType::Continuous, lowerBound, upperBound, objectiveFunctionCoefficient);
}

template<typename ValueType, bool RawMode>
typename LpSolver<ValueType, RawMode>::Variable LpSolver<ValueType, RawMode>::addLowerBoundedContinuousVariable(std::string const& name, ValueType lowerBound,
                                                                                                                ValueType objectiveFunctionCoefficient) {
    return addVariable(name, VariableType::Continuous, lowerBound, std::nullopt, objectiveFunctionCoefficient);
}

template<typename ValueType, bool RawMode>
typename LpSolver<ValueType, RawMode>::Variable LpSolver<ValueType, RawMode>::addUpperBoundedContinuousVariable(std::string const& name, ValueType upperBound,
                                                                                                                ValueType objectiveFunctionCoefficient) {
    return addVariable(name, VariableType::Continuous, std::nullopt, upperBound, objectiveFunctionCoefficient);
}

template<typename ValueType, bool RawMode>
typename LpSolver<ValueType, RawMode>::Variable LpSolver<ValueType, RawMode>::addUnboundedContinuousVariable(std::string const& name,
                                                                                                             ValueType objectiveFunctionCoefficient) {
    return addVariable(name, VariableType::Continuous, std::nullopt, std::nullopt, objectiveFunctionCoefficient);
}

template<typename ValueType, bool RawMode>
typename LpSolver<ValueType, RawMode>::Variable LpSolver<ValueType, RawMode>::addContinuousVariable(std::string const& name,
                                                                                                    std::optional<ValueType> const& lowerBound,
                                                                                                    std::optional<ValueType> const& upperBound,
                                                                                                    ValueType objectiveFunctionCoefficient) {
    return addVariable(name, VariableType::Continuous, lowerBound, upperBound, objectiveFunctionCoefficient);
}

template<typename ValueType, bool RawMode>
typename LpSolver<ValueType, RawMode>::Variable LpSolver<ValueType, RawMode>::addBoundedIntegerVariable(std::string const& name, ValueType lowerBound,
                                                                                                        ValueType upperBound,
                                                                                                        ValueType objectiveFunctionCoefficient) {
    return addVariable(name, VariableType::Integer, lowerBound, upperBound, objectiveFunctionCoefficient);
}

template<typename ValueType, bool RawMode>
typename LpSolver<ValueType, RawMode>::Variable LpSolver<ValueType, RawMode>::addLowerBoundedIntegerVariable(std::string const& name, ValueType lowerBound,
                                                                                                             ValueType objectiveFunctionCoefficient) {
    return addVariable(name, VariableType::Integer, lowerBound, std::nullopt, objectiveFunctionCoefficient);
}

template<typename ValueType, bool RawMode>
typename LpSolver<ValueType, RawMode>::Variable LpSolver<ValueType, RawMode>::addUpperBoundedIntegerVariable(std::string const& name, ValueType upperBound,
                                                                                                             ValueType objectiveFunctionCoefficient) {
    return addVariable(name, VariableType::Integer, std::nullopt, upperBound, objectiveFunctionCoefficient);
}

template<typename ValueType, bool RawMode>
typename LpSolver<ValueType, RawMode>::Variable LpSolver<ValueType, RawMode>::addUnboundedIntegerVariable(std::string const& name,
                                                                                                          ValueType objectiveFunctionCoefficient) {
    return addVariable(name, VariableType::Integer, std::nullopt, std::nullopt, objectiveFunctionCoefficient);
}

template<typename ValueType, bool RawMode>
typename LpSolver<ValueType, RawMode>::Variable LpSolver<ValueType, RawMode>::addIntegerVariable(std::string const& name,
                                                                                                 std::optional<ValueType> const& lowerBound,
                                                                                                 std::optional<ValueType> const& upperBound,
                                                                                                 ValueType objectiveFunctionCoefficient) {
    return addVariable(name, VariableType::Integer, lowerBound, upperBound, objectiveFunctionCoefficient);
}
template<typename ValueType, bool RawMode>
typename LpSolver<ValueType, RawMode>::Variable LpSolver<ValueType, RawMode>::addBinaryVariable(std::string const& name,
                                                                                                ValueType objectiveFunctionCoefficient) {
    return addVariable(name, VariableType::Binary, std::nullopt, std::nullopt, objectiveFunctionCoefficient);
}

template<typename ValueType, bool RawMode>
typename LpSolver<ValueType, RawMode>::Constant LpSolver<ValueType, RawMode>::getConstant(ValueType value) const {
    if constexpr (RawMode) {
        return value;
    } else {
        return manager->rational(value);
    }
}

template<typename ValueType, bool RawMode>
storm::expressions::Variable LpSolver<ValueType, RawMode>::declareOrGetExpressionVariable(std::string const& name, VariableType const& type) {
    switch (type) {
        case VariableType::Continuous:
            return this->manager->declareOrGetVariable(name, this->manager->getRationalType());
        case VariableType::Integer:
        case VariableType::Binary:
            return this->manager->declareOrGetVariable(name, this->manager->getIntegerType());
    }
    STORM_LOG_ASSERT(false, "Unable to declare or get expression variable: Unknown type");
    return {};
}

template class LpSolver<double, true>;
template class LpSolver<double, false>;
template class LpSolver<storm::RationalNumber, true>;
template class LpSolver<storm::RationalNumber, false>;

}  // namespace solver
}  // namespace storm
