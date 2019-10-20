// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <numeric>
#include <pup.h>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/Interpolate.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatedVars.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InterpolatorRegisterElement.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_include "DataStructures/DataBox/Prefixes.hpp"
// Reason for above pragma: If I include
// "DataStructures/DataBox/Prefixes.hpp", IWYU tells me to remove it.
// If I remove it, IWYU tells me to include it.

/// \cond
class DataVector;
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
namespace intrp {
namespace Actions {
struct InterpolatorReceiveVolumeData;
}  // namespace Actions
}  // namespace intrp
/// \endcond

namespace {

namespace Tags {
struct Lapse : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "Lapse"; }
};
}  // namespace Tags

struct MockInterpolatorReceiveVolumeData {
  struct Results {
    // Hardcode expected types here.
    db::item_type<::Tags::TimeStepId> temporal_id{};
    ElementId<3> element_id{};
    Mesh<3> mesh{};
    Variables<tmpl::list<Tags::Lapse>> vars{};
  };
  static Results results;

  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex, size_t VolumeDim>
  static void apply(
      db::DataBox<DbTags>& /*box*/,
      Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      const db::const_item_type<typename Metavariables::temporal_id>&
          temporal_id,
      const ElementId<VolumeDim>& element_id, const ::Mesh<VolumeDim>& mesh,
      Variables<typename Metavariables::interpolator_source_vars>&&
          vars) noexcept {
    results.temporal_id = temporal_id;
    results.element_id = element_id;
    results.mesh = mesh;
    results.vars = vars;
  }
};

MockInterpolatorReceiveVolumeData::Results
    MockInterpolatorReceiveVolumeData::results{};

template <typename Metavariables>
struct mock_interpolator {
  using component_being_mocked = intrp::Interpolator<Metavariables>;
  using replace_these_simple_actions =
      tmpl::list<intrp::Actions::InterpolatorReceiveVolumeData>;
  using with_these_simple_actions =
      tmpl::list<MockInterpolatorReceiveVolumeData>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<intrp::Actions::InitializeInterpolator>>>;
  using initial_databox = db::compute_databox_type<
      typename ::intrp::Actions::InitializeInterpolator::
          template return_tag_list<Metavariables>>;
};

template <typename Metavariables>
struct mock_element {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIndex<Metavariables::volume_dim>;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        tmpl::list<>>>;
  using initial_databox = db::compute_databox_type<db::AddSimpleTags<>>;
};

struct MockMetavariables {
  struct InterpolatorTargetA {
    using vars_to_interpolate_to_target = tmpl::list<Tags::Lapse>;
  };
  using temporal_id = ::Tags::TimeStepId;
  static constexpr size_t volume_dim = 3;
  using interpolator_source_vars = tmpl::list<Tags::Lapse>;
  using interpolation_target_tags = tmpl::list<InterpolatorTargetA>;

  using component_list = tmpl::list<mock_interpolator<MockMetavariables>,
                                    mock_element<MockMetavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Interpolator.InterpolateEvent",
                  "[Unit]") {
  using metavars = MockMetavariables;
  const ElementId<metavars::volume_dim> element_id(2);
  const ElementIndex<metavars::volume_dim> array_index(element_id);

  using interp_component = mock_interpolator<metavars>;
  using elem_component = mock_element<metavars>;
  ActionTesting::MockRuntimeSystem<metavars> runner{{}};
  runner.set_phase(metavars::Phase::Initialization);
  ActionTesting::emplace_component<interp_component>(&runner, 0);
  ActionTesting::next_action<interp_component>(make_not_null(&runner), 0);
  ActionTesting::emplace_component<elem_component>(&runner, array_index);
  ActionTesting::next_action<elem_component>(make_not_null(&runner),
                                             array_index);
  runner.set_phase(metavars::Phase::Testing);

  const Mesh<metavars::volume_dim> mesh(5, Spectral::Basis::Legendre,
                                        Spectral::Quadrature::GaussLobatto);
  const double observation_time = 2.0;
  Variables<metavars::interpolator_source_vars> vars(
      mesh.number_of_grid_points());
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  std::iota(vars.data(), vars.data() + vars.size(), 1.0);

  const auto box = db::create<db::AddSimpleTags<
      metavars::temporal_id, ::Tags::Mesh<metavars::volume_dim>,
      ::Tags::Variables<typename decltype(vars)::tags_list>>>(
      TimeStepId(true, 0, Slab(0., observation_time).end()), mesh, vars);

  intrp::Events::Interpolate<metavars::volume_dim,
                             metavars::interpolator_source_vars> event{};

  event.run(box, runner.cache(), array_index,
            std::add_pointer_t<elem_component>{});

  // Invoke all actions
  runner.invoke_queued_simple_action<interp_component>(0);

  // No more queued simple actions.
  CHECK(runner.is_simple_action_queue_empty<interp_component>(0));
  CHECK(runner.is_simple_action_queue_empty<elem_component>(array_index));

  const auto& results = MockInterpolatorReceiveVolumeData::results;
  CHECK(results.temporal_id.substep_time().value() == observation_time);
  CHECK(results.element_id == element_id);
  CHECK(results.mesh == mesh);
  CHECK(results.vars == vars);
}

}  // namespace
