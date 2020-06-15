// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "DataStructures/Tensor/EagerMath/Norms.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/ElementId.hpp"
//#include
//"Helpers/PointwiseFunctions/AnalyticSolutions/Elasticity/TestHelpers.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Elliptic/Systems/Elasticity/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Elasticity/PotentialEnergy.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "Helpers/Elliptic/DiscontinuousGalerkin/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Elasticity/HalfSpaceMirror.hpp"
#include "PythonBindings/BoostOptional.hpp"
#include "Utilities/GetOutput.hpp"

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
  std::unordered_map<ElementId<Dim>, double> result{};
  for (const auto& id_and_var : operator_applied_to_solution) {
    const auto& element_id = id_and_var.first;
    result[element_id] =
        l2_norm(get<::Elasticity::Tags::Displacement<Dim>>(id_and_var.second));
  return result;
}

template <size_t Dim>
void bind_apply_dg_operator_to_elasticity_halfspacemirror(
    py::module& m) {  // NOLINT
  m.def(("apply_dg_operator_to_elasticity_halfspacemirror"),
        &apply_dg_operator_impl<Dim, ::Elasticity::Solutions::HalfSpaceMirror>,
        py::arg("solution"), py::arg("domain_creator"),
        py::arg("dump_to_file") = boost::optional<std::string>{boost::none});
}
}  // namespace

void bind_apply_dg_operator(py::module& m) {  // NOLINT
  bind_apply_dg_operator_to_elasticity_halfspacemirror<3>(m);
}

}  // namespace py_bindings
}  // namespace Solutions
}  // namespace Elasticity
}  // namespace TestHelpers
