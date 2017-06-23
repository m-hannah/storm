#pragma once

#include "storm-pars/modelchecker/region/RegionModelChecker.h"
#include "storm-pars/modelchecker/instantiation/SparseInstantiationModelChecker.h"
#include "storm-pars/storage/ParameterRegion.h"
#include "storm-pars/utility/parametric.h"

#include "storm/logic/Formulas.h"
#include "storm/modelchecker/CheckTask.h"
#include "storm/modelchecker/results/CheckResult.h"
#include "storm/solver/OptimizationDirection.h"

namespace storm {
    namespace modelchecker {
            
        /*!
         * Class to approximatively check a formula on a parametric model for all parameter valuations within a region
         * It is assumed that all considered valuations are graph-preserving and well defined, i.e.,
         * * all non-const transition probabilities evaluate to some non-zero value
         * * the sum of all outgoing transitions is one
         */
        template <typename SparseModelType, typename ConstantType>
        class SparseParameterLiftingModelChecker : public RegionModelChecker<typename SparseModelType::ValueType> {
        public:
            SparseParameterLiftingModelChecker(SparseModelType const& parametricModel);
            virtual ~SparseParameterLiftingModelChecker() = default;
            
            virtual void specifyFormula(CheckTask<storm::logic::Formula, typename SparseModelType::ValueType> const& checkTask) override;

            /*!
             * Analyzes the given region by means of parameter lifting.
             * We first check whether there is one point in the region for which the property is satisfied/violated.
             * If the given initialResults already indicates that there is such a point, this step is skipped.
             * Then, we check whether ALL points in the region violate/satisfy the property
             *
             */
            virtual RegionResult analyzeRegion(storm::storage::ParameterRegion<typename SparseModelType::ValueType> const& region, RegionResult const& initialResult = RegionResult::Unknown, bool sampleVerticesOfRegion = false) override;

            /*!
             * Checks the specified formula on the given region by applying parameter lifting (Parameter choices are lifted to nondeterministic choices)
             * This yields a (sound) approximative model checking result.

             * @param region the region on which parameter lifting is applied
             * @param dirForParameters  The optimization direction for the parameter choices. If this is, e.g., minimize, then the returned result will be a lower bound for all results induced by the parameter evaluations inside the region.
             */
            std::unique_ptr<CheckResult> check(storm::storage::ParameterRegion<typename SparseModelType::ValueType> const& region, storm::solver::OptimizationDirection const& dirForParameters);
            
        protected:
            
            // Resets all data that correspond to the currently defined property.
            virtual void reset() = 0;

            virtual void specifyBoundedUntilFormula(CheckTask<storm::logic::BoundedUntilFormula, ConstantType> const& checkTask);
            virtual void specifyUntilFormula(CheckTask<storm::logic::UntilFormula, ConstantType> const& checkTask);
            virtual void specifyReachabilityProbabilityFormula(CheckTask<storm::logic::EventuallyFormula, ConstantType> const& checkTask);
            virtual void specifyReachabilityRewardFormula(CheckTask<storm::logic::EventuallyFormula, ConstantType> const& checkTask);
            virtual void specifyCumulativeRewardFormula(CheckTask<storm::logic::CumulativeRewardFormula, ConstantType> const& checkTask);

            
            virtual storm::modelchecker::SparseInstantiationModelChecker<SparseModelType, ConstantType>& getInstantiationChecker() = 0;
            
            virtual std::unique_ptr<CheckResult> computeQuantitativeValues(storm::storage::ParameterRegion<typename SparseModelType::ValueType> const& region, storm::solver::OptimizationDirection const& dirForParameters) = 0;
            
            
            SparseModelType const& parametricModel;
            std::unique_ptr<CheckTask<storm::logic::Formula, ConstantType>> currentCheckTask;

        private:
            // store the current formula. Note that currentCheckTask only stores a reference to the formula.
            std::shared_ptr<storm::logic::Formula const> currentFormula;
            
        };
    }
}
