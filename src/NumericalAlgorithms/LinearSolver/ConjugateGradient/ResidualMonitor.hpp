// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "AlgorithmSingleton.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "IO/Observer/Actions.hpp"
#include "Informer/Tags.hpp"
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/LinearSolver/Observe.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
namespace LinearSolver {
namespace cg_detail {
template <typename FieldsTag>
struct InitializeResidualMonitor;
}  // namespace cg_detail
}  // namespace LinearSolver
namespace Convergence {
struct Criteria;
}  // namespace Convergence
/// \endcond

namespace LinearSolver {
namespace cg_detail {

template <typename Metavariables, typename FieldsTag>
struct ResidualMonitor {
  using chare_type = Parallel::Algorithms::Singleton;
  using const_global_cache_tags =
      tmpl::list<LinearSolver::Tags::Verbosity,
                 LinearSolver::Tags::ConvergenceCriteria>;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             tmpl::list<InitializeResidualMonitor<FieldsTag>>>,

      Parallel::PhaseActions<
          typename Metavariables::Phase,
          Metavariables::Phase::RegisterWithObserver,
          tmpl::list<observers::Actions::RegisterSingletonWithObserverWriter<
              LinearSolver::observe_detail::Registration>>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<ResidualMonitor>(local_cache)
        .start_phase(next_phase);
  }
};

template <typename FieldsTag>
struct InitializeResidualMonitor {
 private:
  using fields_tag = FieldsTag;
  using residual_square_tag = db::add_tag_prefix<
      LinearSolver::Tags::MagnitudeSquare,
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>>;
  using initial_residual_magnitude_tag = db::add_tag_prefix<
      LinearSolver::Tags::Initial,
      db::add_tag_prefix<
          LinearSolver::Tags::Magnitude,
          db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>>>;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using compute_tags = db::AddComputeTags<
        LinearSolver::Tags::MagnitudeCompute<residual_square_tag>,
        LinearSolver::Tags::HasConvergedCompute<fields_tag>>;
    return std::make_tuple(
        ::Initialization::merge_into_databox<
            InitializeResidualMonitor,
            db::AddSimpleTags<LinearSolver::Tags::IterationId,
                              residual_square_tag,
                              initial_residual_magnitude_tag>,
            compute_tags>(std::move(box),
                          db::item_type<LinearSolver::Tags::IterationId>{0},
                          std::numeric_limits<double>::signaling_NaN(),
                          std::numeric_limits<double>::signaling_NaN()),
        true);
  }
};

}  // namespace cg_detail
}  // namespace LinearSolver
