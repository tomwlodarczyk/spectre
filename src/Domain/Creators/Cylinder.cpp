// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Cylinder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <variant>
#include <vector>

#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/ExpandOverBlocks.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeArray.hpp"

namespace Frame {
struct Logical;
struct Inertial;
}  // namespace Frame

namespace domain::creators {
Cylinder::Cylinder(
    const double inner_radius, const double outer_radius,
    const double lower_z_bound, const double upper_z_bound,
    const bool is_periodic_in_z,
    const typename InitialRefinement::type& initial_refinement,
    const typename InitialGridPoints::type& initial_number_of_grid_points,
    const bool use_equiangular_map, std::vector<double> radial_partitioning,
    std::vector<double> partitioning_in_z,
    std::vector<domain::CoordinateMaps::Distribution> radial_distribution,
    std::vector<domain::CoordinateMaps::Distribution> distribution_in_z,
    const Options::Context& context)
    : inner_radius_(inner_radius),
      outer_radius_(outer_radius),
      lower_z_bound_(lower_z_bound),
      upper_z_bound_(upper_z_bound),
      is_periodic_in_z_(is_periodic_in_z),
      use_equiangular_map_(use_equiangular_map),
      radial_partitioning_(std::move(radial_partitioning)),
      partitioning_in_z_(std::move(partitioning_in_z)),
      radial_distribution_(std::move(radial_distribution)),
      distribution_in_z_(std::move(distribution_in_z)) {
  if (inner_radius_ > outer_radius_) {
    PARSE_ERROR(context,
                "Inner radius must be smaller than outer radius, but inner "
                "radius is " +
                    std::to_string(inner_radius_) + " and outer radius is " +
                    std::to_string(outer_radius_) + ".");
  }
  if (lower_z_bound_ > upper_z_bound_) {
    PARSE_ERROR(context,
                "Lower z-bound must be smaller than upper z-bound, but lower "
                "bound is " +
                    std::to_string(lower_z_bound_) + " and upper bound is " +
                    std::to_string(upper_z_bound_) + ".");
  }
  if (not std::is_sorted(radial_partitioning_.begin(),
                         radial_partitioning_.end())) {
    PARSE_ERROR(context,
                "Specify radial partitioning in ascending order. Specified "
                "radial partitioning is: " +
                    get_output(radial_partitioning_));
  }
  if (not radial_partitioning_.empty()) {
    if (radial_partitioning_.front() <= inner_radius_) {
      PARSE_ERROR(
          context,
          "First radial partition must be larger than inner radius, but is: " +
              std::to_string(inner_radius_));
    }
    if (radial_partitioning_.back() >= outer_radius_) {
      PARSE_ERROR(
          context,
          "Last radial partition must be smaller than outer radius, but is: " +
              std::to_string(outer_radius_));
    }
  }
  if (not std::is_sorted(partitioning_in_z_.begin(),
                         partitioning_in_z_.end())) {
    PARSE_ERROR(context,
                "Specify partitioning in z in ascending order. Specified "
                "partitioning is: " +
                    get_output(partitioning_in_z_));
  }
  if (not partitioning_in_z_.empty()) {
    if (partitioning_in_z_.front() <= lower_z_bound_) {
      PARSE_ERROR(
          context,
          "First partition in z must be larger than lower z-bound, but is: " +
              std::to_string(lower_z_bound_));
    }
    if (partitioning_in_z_.back() >= upper_z_bound_) {
      PARSE_ERROR(
          context,
          "Last partition in z must be smaller than upper z-bound, but is: " +
              std::to_string(upper_z_bound_));
    }
  }
  const size_t num_shells = 1 + radial_partitioning_.size();
  const size_t num_layers = 1 + partitioning_in_z_.size();
  if (radial_distribution_.size() != num_shells) {
    PARSE_ERROR(
        context,
        "Specify a 'RadialDistribution' for every cylindrical shell. You "
        "specified "
            << radial_distribution_.size() << " items, but the domain has "
            << num_shells << " shells.");
  }
  if (radial_distribution_.front() !=
      domain::CoordinateMaps::Distribution::Linear) {
    PARSE_ERROR(context,
                "The 'RadialDistribution' must be 'Linear' for the innermost "
                "shell because it changes in circularity. Add entries to "
                "'RadialPartitioning' to add outer shells for which you can "
                "select different radial distributions.");
  }
  if (distribution_in_z_.size() != num_layers) {
    PARSE_ERROR(context,
                "Specify a 'DistributionInZ' for every layer. You specified "
                    << distribution_in_z_.size()
                    << " items, but the domain has " << num_layers
                    << " layers.");
  }
  if (distribution_in_z_.front() !=
      domain::CoordinateMaps::Distribution::Linear) {
    PARSE_ERROR(context,
                "The 'DistributionInZ' must be 'Linear' for the lowermost "
                "layer because a 'Logarithmic' distribution places its "
                "singularity at 'LowerZBound'. Add entries to "
                "'PartitioningInZ' to add layers for which you can "
                "select different distributions along z.");
  }

  // Create block names and groups
  static std::array<std::string, 4> direction_descriptions{"East", "North",
                                                           "West", "South"};
  std::vector<std::string> block_names{};
  std::unordered_map<std::string, std::unordered_set<std::string>>
      block_groups{};
  for (size_t layer = 0; layer < num_layers; ++layer) {
    std::string layer_prefix =
        num_layers > 1 ? "Layer" + std::to_string(layer) : "";
    block_names.push_back(layer_prefix + "InnerCube");
    for (size_t shell = 0; shell < num_shells; ++shell) {
      std::string shell_prefix =
          num_shells > 1 ? "Shell" + std::to_string(shell) : "";
      std::string group_name = layer_prefix + shell_prefix + "Wedges";
      for (size_t direction = 0; direction < 4; ++direction) {
        std::string block_name = layer_prefix + shell_prefix +
                                 gsl::at(direction_descriptions, direction);
        block_names.push_back(block_name);
        block_groups[group_name].insert(block_name);
      }
    }
  }

  // Expand initial refinement and number of grid points over all blocks
  const ExpandOverBlocks<size_t, 3> expand_over_blocks{block_names,
                                                       std::move(block_groups)};
  try {
    initial_refinement_ = std::visit(expand_over_blocks, initial_refinement);
  } catch (const std::exception& error) {
    PARSE_ERROR(context, "Invalid 'InitialRefinement': " << error.what());
  }
  try {
    initial_number_of_grid_points_ =
        std::visit(expand_over_blocks, initial_number_of_grid_points);
  } catch (const std::exception& error) {
    PARSE_ERROR(context, "Invalid 'InitialGridPoints': " << error.what());
  }

  // Set refinement and number of grid points of the central cubes in x and y to
  // the value in angular direction of the wedges and shells
  for (size_t layer = 0; layer < num_layers; layer++) {
    auto& central_cube_refinement =
        initial_refinement_.at(layer * (1 + 4 * num_shells));
    auto& central_cube_grid_points =
        initial_number_of_grid_points_.at(layer * (1 + 4 * num_shells));
    central_cube_refinement[0] = central_cube_refinement[1];
    central_cube_grid_points[0] = central_cube_grid_points[1];
  }
}

Cylinder::Cylinder(
    const double inner_radius, const double outer_radius,
    const double lower_z_bound, const double upper_z_bound,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        lower_z_boundary_condition,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        upper_z_boundary_condition,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        mantle_boundary_condition,
    const typename InitialRefinement::type& initial_refinement,
    const typename InitialGridPoints::type& initial_number_of_grid_points,
    const bool use_equiangular_map, std::vector<double> radial_partitioning,
    std::vector<double> partitioning_in_z,
    std::vector<domain::CoordinateMaps::Distribution> radial_distribution,
    std::vector<domain::CoordinateMaps::Distribution> distribution_in_z,
    const Options::Context& context)
    : Cylinder(inner_radius, outer_radius, lower_z_bound, upper_z_bound, false,
               initial_refinement, initial_number_of_grid_points,
               use_equiangular_map, std::move(radial_partitioning),
               std::move(partitioning_in_z), std::move(radial_distribution),
               std::move(distribution_in_z), context) {
  lower_z_boundary_condition_ = std::move(lower_z_boundary_condition);
  upper_z_boundary_condition_ = std::move(upper_z_boundary_condition);
  mantle_boundary_condition_ = std::move(mantle_boundary_condition);

  using domain::BoundaryConditions::is_periodic;
  if (is_periodic(lower_z_boundary_condition_) xor
      is_periodic(upper_z_boundary_condition_)) {
    PARSE_ERROR(context,
                "Either both lower and upper z-boundary condition must be "
                "periodic, or neither.");
  }
  if (is_periodic(lower_z_boundary_condition_) and
      is_periodic(upper_z_boundary_condition_)) {
    is_periodic_in_z_ = true;
    lower_z_boundary_condition_ = nullptr;
    upper_z_boundary_condition_ = nullptr;
  }
  if (is_periodic(mantle_boundary_condition_)) {
    PARSE_ERROR(context,
                "A Cylinder can't have periodic boundary conditions in the "
                "radial direction.");
  }
  using domain::BoundaryConditions::is_none;
  if (is_none(lower_z_boundary_condition_) or
      is_none(upper_z_boundary_condition_) or
      is_none(mantle_boundary_condition_)) {
    PARSE_ERROR(
        context,
        "None boundary condition is not supported. If you would like an "
        "outflow boundary condition, you must use that.");
  }
  if (mantle_boundary_condition_ == nullptr or
      (not is_periodic_in_z_ and (lower_z_boundary_condition_ == nullptr or
                                  upper_z_boundary_condition_ == nullptr))) {
    PARSE_ERROR(context,
                "z-boundary conditions must not be 'nullptr'. Use the other "
                "constructor to specify 'is_periodic_in_z' instead of boundary "
                "conditions.");
  }
}

Domain<3> Cylinder::create_domain() const noexcept {
  const size_t number_of_shells = 1 + radial_partitioning_.size();
  const size_t number_of_layers = 1 + partitioning_in_z_.size();
  std::vector<PairOfFaces> pairs_of_faces{};
  if (is_periodic_in_z_) {
    // connect faces of end caps in the periodic z-direction
    const size_t corners_per_layer = 4 * (number_of_shells + 1);
    const size_t num_corners = number_of_layers * corners_per_layer;
    PairOfFaces center{
        {0, 1, 2, 3},
        {num_corners + 0, num_corners + 1, num_corners + 2, num_corners + 3}};
    pairs_of_faces.push_back(std::move(center));
    for (size_t j = 0; j < number_of_shells; j++) {
      PairOfFaces east{{1 + 4 * j, 5 + 4 * j, 3 + 4 * j, 7 + 4 * j},
                       {num_corners + 4 * j + 1, num_corners + 4 * j + 5,
                        num_corners + 4 * j + 3, num_corners + 4 * j + 7}};
      PairOfFaces north{{3 + 4 * j, 7 + 4 * j, 2 + 4 * j, 6 + 4 * j},
                        {num_corners + 4 * j + 3, num_corners + 4 * j + 7,
                         num_corners + 4 * j + 2, num_corners + 4 * j + 6}};
      PairOfFaces west{{2 + 4 * j, 6 + 4 * j, 0 + 4 * j, 4 + 4 * j},
                       {num_corners + 4 * j + 2, num_corners + 4 * j + 6,
                        num_corners + 4 * j + 0, num_corners + 4 * j + 4}};
      PairOfFaces south{{0 + 4 * j, 4 + 4 * j, 1 + 4 * j, 5 + 4 * j},
                        {num_corners + 4 * j + 0, num_corners + 4 * j + 4,
                         num_corners + 4 * j + 1, num_corners + 4 * j + 5}};
      pairs_of_faces.push_back(std::move(east));
      pairs_of_faces.push_back(std::move(north));
      pairs_of_faces.push_back(std::move(west));
      pairs_of_faces.push_back(std::move(south));
    }
  }

  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions_all_blocks{};
  if (mantle_boundary_condition_ != nullptr) {
    // Note: The first block in each disk is the central cube.
    boundary_conditions_all_blocks.resize((1 + 4 * number_of_shells) *
                                          number_of_layers);

    // Boundary conditions in z
    for (size_t block_id = 0; not is_periodic_in_z_ and
                              block_id < boundary_conditions_all_blocks.size();
         ++block_id) {
      if (block_id < (1 + number_of_shells * 4)) {
        boundary_conditions_all_blocks[block_id][Direction<3>::lower_zeta()] =
            lower_z_boundary_condition_->get_clone();
      }
      if (block_id >=
          boundary_conditions_all_blocks.size() - (1 + number_of_shells * 4)) {
        boundary_conditions_all_blocks[block_id][Direction<3>::upper_zeta()] =
            upper_z_boundary_condition_->get_clone();
      }
    }
    // Radial boundary conditions
    for (size_t block_id = 1 + 4 * (number_of_shells - 1);
         block_id < boundary_conditions_all_blocks.size(); ++block_id) {
      // clang-tidy thinks we can get division by zero on the modulus operator.
      // NOLINTNEXTLINE
      if (block_id % (1 + 4 * number_of_shells) == 0) {
        // skip the central cubes and the inner radial wedges
        block_id += 4 * (number_of_shells - 1);
        continue;
      }
      boundary_conditions_all_blocks[block_id][Direction<3>::upper_xi()] =
          mantle_boundary_condition_->get_clone();
    }
  }

  return Domain<3>{
      cyl_wedge_coordinate_maps<Frame::Inertial>(
          inner_radius_, outer_radius_, lower_z_bound_, upper_z_bound_,
          use_equiangular_map_, radial_partitioning_, partitioning_in_z_,
          radial_distribution_, distribution_in_z_),
      corners_for_cylindrical_layered_domains(number_of_shells,
                                              number_of_layers),
      pairs_of_faces, std::move(boundary_conditions_all_blocks)};
}

std::vector<std::array<size_t, 3>> Cylinder::initial_extents() const noexcept {
  return initial_number_of_grid_points_;
}

std::vector<std::array<size_t, 3>> Cylinder::initial_refinement_levels()
    const noexcept {
  return initial_refinement_;
}
}  // namespace domain::creators
