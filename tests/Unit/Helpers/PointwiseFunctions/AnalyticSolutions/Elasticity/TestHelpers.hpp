// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/ElementMap.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Elliptic/Systems/Elasticity/FirstOrderSystem.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
#include "Helpers/Elliptic/DiscontinuousGalerkin/TestHelpers.hpp"

/// \cond
template <size_t Dim>
struct DomainCreator;
/// \endcond

namespace helpers = TestHelpers::elliptic::dg;

namespace TestHelpers {
namespace Elasticity {
namespace Solutions {

template <size_t Dim, typename SolutionType>
auto apply_dg_operator(const SolutionType& solution,
                       const DomainCreator<Dim>& domain_creator) {
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
  const auto package_sources_args =
      [](const ElementId<Dim>& /*element_id*/,
         const helpers::DgElement<Dim>& /*dg_element*/) {
        return std::tuple<>{};
      };
  // Disable boundary terms since analytic solutions are continuous across
  // boundaries.
  const auto package_boundary_data = [](const Mesh<Dim - 1>& face_mesh,
                                        const auto&... /* unused */) {
    // Unused, but needed to compile projection logic
    return ::dg::SimpleBoundaryData<
        tmpl::list<::Elasticity::Tags::Displacement<Dim>>, tmpl::list<>>{
        face_mesh.number_of_grid_points()};
  };

  return helpers::apply_dg_operator_to_solution<system>(
      solution, domain_creator,
      [&fluxes_computer, &package_fluxes_args, &package_sources_args,
       &package_boundary_data, &apply_boundary_contribution](
          const auto&. args) {
        return helpers::apply_first_order_dg_operator<system>(
            local_element_id, local_elements, local_all_variables,,
            fluxes_computer, package_fluxes_args, package_sources_args,
            package_boundary_data, [](const auto&... /* unused */) {});
      });
}

}  // namespace Solutions
}  // namespace Elasticity
}  // namespace TestHelpers
