// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Actions/InitializeSystem.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Elliptic/DiscontinuousGalerkin/ImposeBoundaryConditions.hpp"
#include "Elliptic/DiscontinuousGalerkin/ImposeInhomogeneousBoundaryConditionsOnSource.hpp"
#include "Elliptic/DiscontinuousGalerkin/InitializeFluxes.hpp"
#include "Elliptic/FirstOrderOperator.hpp"
#include "Elliptic/Systems/Poisson/Actions/Observe.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Elliptic/Tags.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "IO/Observer/Actions.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ApplyFluxes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ComputeNonconservativeBoundaryFluxes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/FluxCommunication.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearSolver/Actions/TerminateIfConverged.hpp"
#include "NumericalAlgorithms/LinearSolver/Gmres/Gmres.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeInterfaces.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeMortars.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/ProductOfSinusoids.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;

  static constexpr OptionString help{
      "Find the solution to a Poisson problem in Dim spatial dimensions.\n"
      "Analytic solution: ProductOfSinusoids\n"
      "Linear solver: GMRES\n"
      "Numerical flux: FirstOrderInternalPenaltyFlux"};

  // The system provides all equations specific to the problem.
  using system = Poisson::FirstOrderSystem<Dim>;

  // The analytic solution and corresponding source to solve the Poisson
  // equation for
  using analytic_solution_tag =
      Tags::AnalyticSolution<Poisson::Solutions::ProductOfSinusoids<Dim>>;

  // The linear solver algorithm. We must use GMRES since the operator is
  // not positive-definite for the first-order system.
  using linear_solver =
      LinearSolver::Gmres<Metavariables, typename system::fields_tag>;
  using temporal_id = LinearSolver::Tags::IterationId;

  // This is needed for InitializeMortars and will be removed ASAP.
  static constexpr bool local_time_stepping = false;

  // Parse numerical flux parameters from the input file to store in the cache.
  using normal_dot_numerical_flux =
      Tags::NumericalFlux<Poisson::FirstOrderInternalPenaltyFlux<Dim>>;

  // Collect all items to store in the cache.
  using const_global_cache_tags =
      tmpl::list<analytic_solution_tag,
                 elliptic::Tags::FluxesComputer<typename system::fluxes>>;

  // Collect all reduction tags for observers
  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      tmpl::list<Poisson::Actions::Observe, linear_solver>>;

  // Specify all global synchronization points.
  enum class Phase { Initialization, RegisterWithObserver, Solve, Exit };

  using initialization_actions = tmpl::list<
      dg::Actions::InitializeDomain<Dim>, elliptic::Actions::InitializeSystem,
      dg::Actions::InitializeInterfaces<
          system,
          dg::Initialization::slice_tags_to_face<
              typename system::variables_tag>,
          dg::Initialization::slice_tags_to_exterior<>>,
      elliptic::dg::Actions::ImposeInhomogeneousBoundaryConditionsOnSource<
          Metavariables>,
      typename linear_solver::initialize_element,
      dg::Actions::InitializeMortars<Metavariables>,
      elliptic::dg::Actions::InitializeFluxes<Metavariables>,
      // Initialization is done. Avoid introducing an extra phase by
      // advancing the linear solver to the first step here.
      typename linear_solver::prepare_step,
      Initialization::Actions::RemoveOptionsAndTerminatePhase>;

  // Specify all parallel components that will execute actions at some point.
  using component_list = tmpl::append<
      tmpl::list<elliptic::DgElementArray<
          Metavariables,
          tmpl::list<
              Parallel::PhaseActions<Phase, Phase::Initialization,
                                     initialization_actions>,

              Parallel::PhaseActions<
                  Phase, Phase::RegisterWithObserver,
                  tmpl::list<observers::Actions::RegisterWithObservers<
                                 Poisson::Actions::Observe>,
                             Parallel::Actions::TerminatePhase>>,

              Parallel::PhaseActions<
                  Phase, Phase::Solve,
                  tmpl::list<Poisson::Actions::Observe,
                             LinearSolver::Actions::TerminateIfConverged,
                             dg::Actions::SendDataForFluxes<Metavariables>,
                             Actions::MutateApply<elliptic::FirstOrderOperator<
                                 Dim, LinearSolver::Tags::OperatorAppliedTo,
                                 typename system::variables_tag>>,
                             elliptic::dg::Actions::
                                 ImposeHomogeneousDirichletBoundaryConditions<
                                     Metavariables>,
                             dg::Actions::ReceiveDataForFluxes<Metavariables>,
                             dg::Actions::ApplyFluxes,
                             typename linear_solver::perform_step,
                             typename linear_solver::prepare_step>>>>>,
      typename linear_solver::component_list,
      tmpl::list<observers::Observer<Metavariables>,
                 observers::ObserverWriter<Metavariables>>>;

  // Specify the transitions between phases.
  static Phase determine_next_phase(
      const Phase& current_phase,
      const Parallel::CProxy_ConstGlobalCache<
          Metavariables>& /*cache_proxy*/) noexcept {
    switch (current_phase) {
      case Phase::Initialization:
        return Phase::RegisterWithObserver;
      case Phase::RegisterWithObserver:
        return Phase::Solve;
      case Phase::Solve:
        return Phase::Exit;
      case Phase::Exit:
        ERROR(
            "Should never call determine_next_phase with the current phase "
            "being 'Exit'");
      default:
        ERROR(
            "Unknown type of phase. Did you static_cast<Phase> an integral "
            "value?");
    }
  }
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &domain::creators::register_derived_with_charm};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
/// \endcond
