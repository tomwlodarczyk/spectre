// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "AlgorithmArray.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "IO/Observer/Actions.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "Options/Options.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Printf.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Actions {
template <size_t Dim>
struct ExportCoordinates {
  // Compile-time interface for observers
  struct ObservationType {};
  template <typename ParallelComponent, typename DbTagsList,
            typename ArrayIndex>
  static std::pair<observers::TypeOfObservation, observers::ObservationId>
  register_info(const db::DataBox<DbTagsList>& /*box*/,
                const ArrayIndex& /*array_index*/) noexcept {
    return {observers::TypeOfObservation::ReductionAndVolume,
            observers::ObservationId(0., ObservationType{})};
  }

  template <
      typename DbTagsList, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<tmpl::list_contains_v<DbTagsList, Tags::Mesh<Dim>>> = nullptr>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& mesh = get<Tags::Mesh<Dim>>(box);
    const auto& inertial_coordinates =
        db::get<::Tags::Coordinates<Dim, Frame::Inertial>>(box);
    const std::string element_name = MakeString{} << ElementId<Dim>(array_index)
                                                  << '/';
    // Collect volume data
    // Remove tensor types, only storing individual components
    std::vector<TensorComponent> components;
    components.reserve(Dim);
    for (size_t d = 0; d < Dim; d++) {
      components.emplace_back(element_name + "InertialCoordinates_" +
                                  inertial_coordinates.component_name(
                                      inertial_coordinates.get_tensor_index(d)),
                              inertial_coordinates.get(d));
    }
    // Send data to volume observer
    auto& local_observer =
        *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
             cache)
             .ckLocalBranch();
    Parallel::simple_action<observers::Actions::ContributeVolumeData>(
        local_observer, observers::ObservationId(0., ObservationType{}),
        std::string{"/element_data"},
        observers::ArrayComponentId(
            std::add_pointer_t<ParallelComponent>{nullptr},
            Parallel::ArrayIndex<ElementIndex<Dim>>(array_index)),
        std::move(components), mesh.extents());
    return {std::move(box), true};
  }
};
}  // namespace Actions

template <size_t Dim>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;

  static constexpr OptionString help{
      "Export the inertial coordinates of the Domain specified in the input "
      "file. The output can be used to compute initial data externally, for "
      "instance."};

  enum class Phase { Initialization, RegisterWithObserver, Export, Exit };

  using component_list = tmpl::list<
      DgElementArray<
          Metavariables,
          tmpl::list<
              Parallel::PhaseActions<
                  typename Metavariables::Phase,
                  Metavariables::Phase::Initialization,
                  tmpl::list<dg::Actions::InitializeDomain<Dim>,
                             ::Initialization::Actions::
                                 RemoveOptionsAndTerminatePhase>>,
              Parallel::PhaseActions<
                  typename Metavariables::Phase,
                  Metavariables::Phase::RegisterWithObserver,
                  tmpl::list<observers::Actions::RegisterWithObservers<
                                 Actions::ExportCoordinates<Dim>>,
                             Parallel::Actions::TerminatePhase>>,
              Parallel::PhaseActions<
                  typename Metavariables::Phase, Metavariables::Phase::Export,
                  tmpl::list<Actions::ExportCoordinates<Dim>>>>>,
      observers::Observer<Metavariables>,
      observers::ObserverWriter<Metavariables>>;

  using observed_reduction_data_tags = tmpl::list<>;

  static Phase determine_next_phase(
      const Phase& current_phase,
      const Parallel::CProxy_ConstGlobalCache<
          Metavariables>& /*cache_proxy*/) noexcept {
    switch (current_phase) {
      case Phase::Initialization:
        return Phase::RegisterWithObserver;
      case Phase::RegisterWithObserver:
        return Phase::Export;
      case Phase::Export:
        return Phase::Exit;
      case Phase::Exit:
        ERROR(
            "Should never call determine_next_phase with the current phase "
            "being 'Exit'");
      default:
        ERROR("Unknown type of phase.");
    }
  }
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &domain::creators::register_derived_with_charm};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
/// \endcond
