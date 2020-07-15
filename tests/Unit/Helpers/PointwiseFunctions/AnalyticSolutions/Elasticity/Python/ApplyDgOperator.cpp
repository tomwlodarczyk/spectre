// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cmath>
#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "DataStructures/Tensor/EagerMath/Norms.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Elliptic/Systems/Elasticity/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "Helpers/Elliptic/DiscontinuousGalerkin/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "PointwiseFunctions/AnalyticSolutions/Elasticity/BentBeam.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Elasticity/HalfSpaceMirror.hpp"
#include "PointwiseFunctions/Elasticity/PotentialEnergy.hpp"
#include "PythonBindings/BoostOptional.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace py = pybind11;

namespace TestHelpers {
namespace Elasticity {
namespace Solutions {
namespace py_bindings {

namespace helpers = TestHelpers::elliptic::dg;

namespace {
template <size_t Dim, typename SolutionType>
auto apply_dg_operator_impl(
    const SolutionType& solution, const DomainCreator<Dim>& domain_creator,
    const boost::optional<std::string>& file_to_dump_to = boost::none) {
  using system = ::Elasticity::FirstOrderSystem<Dim>;
  const typename system::fluxes fluxes_computer{};
  const auto& constitutive_relation = solution.constitutive_relation();
  const auto package_fluxes_args = make_overloader(
      [&constitutive_relation](const ElementId<Dim>& /*element_id*/,
                               const helpers::DgElement<Dim>& dg_element) {
        const auto logical_coords = logical_coordinates(dg_element.mesh);
        const auto inertial_coords = dg_element.element_map(logical_coords);
        return std::make_tuple(constitutive_relation, inertial_coords);
      },
      [&constitutive_relation](const ElementId<Dim>& /*element_id*/,
                               const helpers::DgElement<Dim>& dg_element,
                               const Direction<Dim>& direction) {
        const auto face_mesh =
            dg_element.mesh.slice_away(direction.dimension());
        const auto logical_coords =
            interface_logical_coordinates(face_mesh, direction);
        const auto inertial_coords = dg_element.element_map(logical_coords);
        return std::make_tuple(constitutive_relation, inertial_coords);
      });

  const auto operator_applied_to_solution =
      helpers::apply_dg_operator_to_solution<system>(
          solution, domain_creator,
          [&fluxes_computer, &package_fluxes_args](const auto&... args) {
            return helpers::apply_first_order_dg_operator<system>(
                args..., fluxes_computer, package_fluxes_args,
                // The fluxes and sources need no arguments, so we return
                // empty tuples
                [](const auto&... /* unused */) { return std::tuple<>{}; },
                // Disable boundary terms since the analytic solutions are
                // continuous across element boundaries.
                [](const auto&... /* unused */) {
                  return TestHelpers::elliptic::dg::EmptyBoundaryData{};
                },
                [](const auto&... /* unused */) {});
          },
          file_to_dump_to);

  //    apply_dg_operator(solution, domain_creator);
  // Convert the data to a type that Python understands. We just take a
  // norm of the Displacement field over each element for now.
  std::unordered_map<ElementId<Dim>, std::array<double , 2>> result{};
  for (const auto& id_and_var : operator_applied_to_solution) {
    const auto& element_id = id_and_var.first;
    result[element_id][0] =
        l2_norm(get<::Elasticity::Tags::Displacement<Dim>>(id_and_var.second));
    result[element_id][1] =
        l2_norm(get<::Elasticity::Tags::Strain<Dim>>(id_and_var.second));
  }
  return result;
}

template <size_t Dim, typename SolutionType>
auto evaluate_potential_energy_impl(const SolutionType& solution,
                                    const DomainCreator<Dim>& domain_creator) {
  const auto& constitutive_relation = solution.constitutive_relation();
  std::unordered_map<ElementId<Dim>, double> result{};
  const auto dg_elements = helpers::create_elements(domain_creator);
  for (const auto& id_and_elem : dg_elements) {
    const auto& element_id = id_and_elem.first;
    const auto& dg_element = id_and_elem.second;
    const auto logical_coords = logical_coordinates(dg_element.mesh);
    const auto inertial_coords = dg_element.element_map(logical_coords);
    // typename ::Elasticity::Tags::Strain<Dim>::type strain =
    //     get<::Elasticity::Tags::Strain<Dim>>(solution.variables(
    //         inertial_coords, tmpl::list<::Elasticity::Tags::Strain<Dim>>{}));
   const auto displacement = variables_from_tagged_tuple(solution.variables(
        inertial_coords, tmpl::list<::Elasticity::Tags::Displacement<Dim>>{}));
    const auto grad_displacement =
        get<::Tags::deriv<::Elasticity::Tags::Displacement<Dim>,
                          tmpl::size_t<Dim>, Frame::Inertial>>(
            partial_derivatives<
                // The compiler error was here: This must be a tmpl::list<>
                tmpl::list<::Elasticity::Tags::Displacement<Dim>>>(
                displacement, dg_element.mesh, dg_element.inv_jacobian));
    auto strain =
        make_with_value<tnsr::ii<DataVector, Dim>>(grad_displacement, 0.);
    for (size_t i = 0; i < Dim; i++) {
      // Diagonal elements
      strain.get(i, i) = grad_displacement.get(i, i);
      // Symmetric off-diagonal elements
      for (size_t j = 0; j < i; j++) {
        strain.get(i, j) =
            0.5 * (grad_displacement.get(i, j) + grad_displacement.get(j, i));
      }
    }
    const auto potential_energy = ::Elasticity::potential_energy_density<Dim>(
        strain, inertial_coords, constitutive_relation);
    const DataVector det_jacobian = get(determinant(dg_element.jacobian));
    // result[element_id] = definite_integral(det_jacobian, dg_element.mesh);
    result[element_id] = definite_integral(det_jacobian * get(potential_energy),
                                           dg_element.mesh);
  }
  return result;
}

auto test_cos2_impl(const DomainCreator<3>& domain_creator) {
  std::unordered_map<ElementId<3>, double> result{};
  const auto dg_elements = helpers::create_elements(domain_creator);
  for (const auto& id_and_elem : dg_elements) {
    const auto& element_id = id_and_elem.first;
    const auto& dg_element = id_and_elem.second;
    const auto logical_coords = logical_coordinates(dg_element.mesh);
    const auto x = dg_element.element_map(logical_coords);
    const auto radius2 = square(get<0>(x)) + square(get<1>(x));
    Variables<tmpl::list<::Tags::TempScalar<0>>> cos2_phi =
        square(get<0>(x)) / radius2;
    auto analytical_deriv = make_with_value<tnsr::i<DataVector, 3>>(x, 0.);
    get<0>(analytical_deriv) = 2. * get<0>(x) * square(get<1>(x)) / radius2;
    get<1>(analytical_deriv) = -2. * square(get<0>(x)) * get<1>(x) / radius2;
    const auto grad_cos2_phi = get<
        ::Tags::deriv<::Tags::TempScalar<0>, tmpl::size_t<3>, Frame::Inertial>>(
        partial_derivatives<tmpl::list<::Tags::TempScalar<0>>>(
            cos2_phi, dg_element.mesh, dg_element.inv_jacobian));
    auto residual = make_with_value<tnsr::i<DataVector, 3>>(x, 0.);
    for (size_t i = 0; i < 3; i++) {
      residual.get(i) = analytical_deriv.get(i) - grad_cos2_phi.get(i);
    }
    result[element_id] = l2_norm(residual);
  }
  return result;
}

void bind_apply_dg_operator_to_elasticity_halfspacemirror(
    py::module& m) {  // NOLINT
  m.def(("apply_dg_operator_to_elasticity_halfspacemirror"),
        &apply_dg_operator_impl<3, ::Elasticity::Solutions::HalfSpaceMirror>,
        py::arg("solution"), py::arg("domain_creator"),
        py::arg("dump_to_file") = boost::optional<std::string>{boost::none});
}

void bind_evaluate_potential_energy_of_elasticity_halfspacemirror(
    py::module& m) {  // NOLINT
  m.def(
      ("evaluate_potential_energy_of_elasticity_halfspacemirror"),
      &evaluate_potential_energy_impl<3,
                                      ::Elasticity::Solutions::HalfSpaceMirror>,
      py::arg("solution"), py::arg("domain_creator"));
}
}  // namespace

void bind_apply_dg_operator(py::module& m) {  // NOLINT
  bind_apply_dg_operator_to_elasticity_halfspacemirror(m);
}

void bind_evaluate_potential_energy_of_halfspacemirror(
    py::module& m) {  // NOLINT
  bind_evaluate_potential_energy_of_elasticity_halfspacemirror(m);
}

void bind_cos2(py::module& m) {  // NOLINT
  bind_test_cos2(m);
}

}  // namespace py_bindings
}  // namespace Solutions
}  // namespace Elasticity
}  // namespace TestHelpers
