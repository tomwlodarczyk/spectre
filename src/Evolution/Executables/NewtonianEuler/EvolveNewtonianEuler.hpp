// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Evolution/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/Actions/ComputeVolumeFluxes.hpp"
#include "Evolution/Actions/ComputeVolumeSources.hpp"
#include "Evolution/ComputeTags.hpp"
#include "Evolution/Conservative/UpdateConservatives.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/LimiterActions.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Minmod.tpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Tags.hpp"
#include "Evolution/Initialization/ConservativeSystem.hpp"
#include "Evolution/Initialization/DiscontinuousGalerkin.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/Limiter.hpp"
#include "Evolution/Systems/NewtonianEuler/SoundSpeedSquared.hpp"
#include "Evolution/Systems/NewtonianEuler/System.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "IO/Observer/Actions.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/RegisterObservers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ApplyBoundaryFluxesLocalTimeStepping.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ApplyFluxes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/FluxCommunication.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ImposeBoundaryConditions.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/Hll.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeInterfaces.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeMortars.hpp"
#include "ParallelAlgorithms/Events/ObserveErrorNorms.hpp"
#include "ParallelAlgorithms/Events/ObserveFields.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/RiemannProblem.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/Actions/ChangeStepSize.hpp"
#include "Time/Actions/RecordTimeStepperData.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/Actions/UpdateU.hpp"
#include "Time/StepChoosers/Cfl.hpp"
#include "Time/StepChoosers/Constant.hpp"
#include "Time/StepChoosers/Increase.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepControllers/StepController.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/Triggers/TimeTriggers.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
namespace Parallel {
template <typename Metavariables>
class CProxy_ConstGlobalCache;
}  // namespace Parallel
/// \endcond

template <size_t Dim>
struct EvolutionMetavars {
  static constexpr size_t volume_dim = Dim;
  using initial_data = NewtonianEuler::Solutions::RiemannProblem<Dim>;

  using equation_of_state_type = typename initial_data::equation_of_state_type;

  using system = NewtonianEuler::System<Dim, equation_of_state_type>;

  using temporal_id = Tags::TimeStepId;
  static constexpr bool local_time_stepping = false;

  using initial_data_tag = Tags::AnalyticSolution<initial_data>;
  using boundary_condition_tag = initial_data_tag;
  using analytic_variables_tags =
      typename system::primitive_variables_tag::tags_list;

  using equation_of_state_tag =
      hydro::Tags::EquationOfState<equation_of_state_type>;

  using normal_dot_numerical_flux =
      Tags::NumericalFlux<dg::NumericalFluxes::Hll<system>>;

  using limiter = Tags::Limiter<Limiters::Minmod<
      Dim, tmpl::list<NewtonianEuler::Tags::MassDensityCons<DataVector>,
                      NewtonianEuler::Tags::MomentumDensity<DataVector, Dim,
                                                            Frame::Inertial>,
                      NewtonianEuler::Tags::EnergyDensity<DataVector>>>>;

  using events = tmpl::list<
      dg::Events::Registrars::ObserveErrorNorms<Tags::Time,
                                                analytic_variables_tags>,
      dg::Events::Registrars::ObserveFields<
          Dim, Tags::Time,
          tmpl::append<
              db::get_variables_tags_list<typename system::variables_tag>,
              db::get_variables_tags_list<
                  typename system::primitive_variables_tag>>,
          analytic_variables_tags>>;
  using triggers = Triggers::time_triggers;

  using step_choosers =
      tmpl::list<StepChoosers::Registrars::Cfl<Dim, Frame::Inertial>,
                 StepChoosers::Registrars::Constant,
                 StepChoosers::Registrars::Increase>;

  struct ObservationType {};
  using element_observation_type = ObservationType;

  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      typename Event<events>::creatable_classes>;

  using step_actions = tmpl::flatten<tmpl::list<
      Actions::ComputeVolumeFluxes,
      dg::Actions::SendDataForFluxes<EvolutionMetavars>,
      Actions::ComputeTimeDerivative,
      dg::Actions::ImposeDirichletBoundaryConditions<EvolutionMetavars>,
      dg::Actions::ReceiveDataForFluxes<EvolutionMetavars>,
      tmpl::conditional_t<local_time_stepping, tmpl::list<>,
                          dg::Actions::ApplyFluxes>,
      Actions::RecordTimeStepperData,
      tmpl::conditional_t<local_time_stepping,
                          dg::Actions::ApplyBoundaryFluxesLocalTimeStepping,
                          tmpl::list<>>,
      Actions::UpdateU, Limiters::Actions::SendData<EvolutionMetavars>,
      Limiters::Actions::Limit<EvolutionMetavars>,
      // Conservative `UpdatePrimitives` expects system to possess
      // list of recovery schemes so we use `MutateApply` instead.
      Actions::MutateApply<typename system::primitive_from_conservative>>>;

  enum class Phase {
    Initialization,
    InitializeTimeStepperHistory,
    RegisterWithObserver,
    Evolve,
    Exit
  };

  using initialization_actions = tmpl::list<
      dg::Actions::InitializeDomain<Dim>,
      Initialization::Actions::ConservativeSystem,
      Initialization::Actions::AddComputeTags<
          tmpl::list<NewtonianEuler::Tags::SoundSpeedSquaredCompute<DataVector>,
                     NewtonianEuler::Tags::SoundSpeedCompute<DataVector>>>,
      Actions::UpdateConservatives,
      dg::Actions::InitializeInterfaces<
          system,
          dg::Initialization::slice_tags_to_face<
              typename system::variables_tag,
              typename system::primitive_variables_tag,
              NewtonianEuler::Tags::SoundSpeed<DataVector>>,
          dg::Initialization::slice_tags_to_exterior<
              typename system::primitive_variables_tag,
              NewtonianEuler::Tags::SoundSpeed<DataVector>>>,
      Initialization::Actions::Evolution<EvolutionMetavars>,
      tmpl::conditional_t<
          evolution::is_analytic_solution_v<initial_data>,
          Initialization::Actions::AddComputeTags<
              tmpl::list<evolution::Tags::AnalyticCompute<
                  Dim, initial_data_tag, analytic_variables_tags>>>,
          tmpl::list<>>,
      dg::Actions::InitializeMortars<EvolutionMetavars>,
      Initialization::Actions::DiscontinuousGalerkin<EvolutionMetavars>,
      Initialization::Actions::Minmod<Dim>,
      Initialization::Actions::RemoveOptionsAndTerminatePhase>;

  using component_list = tmpl::list<
      observers::Observer<EvolutionMetavars>,
      observers::ObserverWriter<EvolutionMetavars>,
      DgElementArray<
          EvolutionMetavars,
          tmpl::list<
              Parallel::PhaseActions<Phase, Phase::Initialization,
                                     initialization_actions>,

              Parallel::PhaseActions<
                  Phase, Phase::InitializeTimeStepperHistory,
                  tmpl::flatten<tmpl::list<
                      SelfStart::self_start_procedure<step_actions>>>>,

              Parallel::PhaseActions<
                  Phase, Phase::RegisterWithObserver,
                  tmpl::list<observers::Actions::RegisterWithObservers<
                                 observers::RegisterObservers<
                                     Tags::Time, element_observation_type>>,
                             Parallel::Actions::TerminatePhase>>,

              Parallel::PhaseActions<
                  Phase, Phase::Evolve,
                  tmpl::flatten<tmpl::list<
                      Actions::UpdateConservatives,
                      Actions::RunEventsAndTriggers,
                      tmpl::conditional_t<
                          local_time_stepping,
                          Actions::ChangeStepSize<step_choosers>, tmpl::list<>>,
                      step_actions, Actions::AdvanceTime>>>>>>;

  using const_global_cache_tags =
      tmpl::list<initial_data_tag,
                 Tags::TimeStepper<tmpl::conditional_t<
                     local_time_stepping, LtsTimeStepper, TimeStepper>>,
                 Tags::EventsAndTriggers<events, triggers>>;

  static constexpr OptionString help{
      "Evolve the Newtonian Euler system in conservative form.\n\n"};

  static Phase determine_next_phase(
      const Phase& current_phase,
      const Parallel::CProxy_ConstGlobalCache<
          EvolutionMetavars>& /*cache_proxy*/) noexcept {
    switch (current_phase) {
      case Phase::Initialization:
        return Phase::InitializeTimeStepperHistory;
      case Phase::InitializeTimeStepperHistory:
        return Phase::RegisterWithObserver;
      case Phase::RegisterWithObserver:
        return Phase::Evolve;
      case Phase::Evolve:
        return Phase::Exit;
      case Phase::Exit:
        ERROR(
            "Should never call determine_next_phase with the current phase "
            "being 'Exit'");
      default:
        ERROR(
            "Unknown type of phase. Did you static_cast<Phase> to an integral "
            "value?");
    }
  }
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling,
    &domain::creators::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<
        Event<metavariables::events>>,
    &Parallel::register_derived_classes_with_charm<
        StepChooser<metavariables::step_choosers>>,
    &Parallel::register_derived_classes_with_charm<StepController>,
    &Parallel::register_derived_classes_with_charm<TimeStepper>,
    &Parallel::register_derived_classes_with_charm<
        Trigger<metavariables::triggers>>};

static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
