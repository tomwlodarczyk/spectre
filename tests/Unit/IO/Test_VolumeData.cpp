// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <boost/iterator/transform_iterator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TensorData.hpp"
#include "ErrorHandling/Error.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/VolumeData.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Numeric.hpp"

SPECTRE_TEST_CASE("Unit.IO.H5.VolumeData", "[Unit][IO][H5]") {
  const std::string h5_file_name("Unit.IO.H5.VolumeData.h5");
  const uint32_t version_number = 4;
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }

  h5::H5File<h5::AccessType::ReadWrite> my_file(h5_file_name);
  const std::vector<DataVector> tensor_components_and_coords{
      {8.9, 7.6, 3.9, 2.1, 18.9, 17.6, 13.9, 12.1},
      {0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0},
      {0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0},
      {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0},
      {-78.9, -7.6, -1.9, 8.1, 6.3, 8.7, 9.8, 0.2},
      {-7.9, 7.6, 1.9, -8.1, -6.3, 2.7, 6.8, -0.2},
      {17.9, 27.6, 21.9, -28.1, -26.3, 32.7, 26.8, -30.2}};
  const std::vector<size_t> observation_ids{8435087234, size_t(-1)};
  const std::vector<double> observation_values{8.0, 2.3};
  const std::vector<std::string> grid_names{"[[2,3,4]]", "[[5,6,7]]"};
  {
    auto& volume_file =
        my_file.insert<h5::VolumeData>("/element_data", version_number);
    const auto write_to_file = [
      &volume_file, &tensor_components_and_coords, &grid_names
    ](const size_t observation_id, const double observation_value) noexcept {
      std::string first_grid = grid_names.front();
      std::string last_grid = grid_names.back();
      volume_file.write_volume_data(
          observation_id, observation_value,
          std::vector<ExtentsAndTensorVolumeData>{
              {{2, 2, 2},
               {TensorComponent{
                    first_grid + "/S",
                    observation_value * tensor_components_and_coords[0]},
                TensorComponent{
                    first_grid + "/x-coord",
                    observation_value * tensor_components_and_coords[1]},
                TensorComponent{
                    first_grid + "/y-coord",
                    observation_value * tensor_components_and_coords[2]},
                TensorComponent{
                    first_grid + "/z-coord",
                    observation_value * tensor_components_and_coords[3]},
                TensorComponent{
                    first_grid + "/T_x",
                    observation_value * tensor_components_and_coords[4]},
                TensorComponent{
                    first_grid + "/T_y",
                    observation_value * tensor_components_and_coords[5]},
                TensorComponent{
                    first_grid + "/T_z",
                    observation_value * tensor_components_and_coords[6]}}},
              // Second Element Data
              {{2, 2, 2},
               {TensorComponent{
                    last_grid + "/S",
                    observation_value * tensor_components_and_coords[1]},
                TensorComponent{
                    last_grid + "/x-coord",
                    observation_value * tensor_components_and_coords[0]},
                TensorComponent{
                    last_grid + "/y-coord",
                    observation_value * tensor_components_and_coords[5]},
                TensorComponent{
                    last_grid + "/z-coord",
                    observation_value * tensor_components_and_coords[3]},
                TensorComponent{
                    last_grid + "/T_x",
                    observation_value * tensor_components_and_coords[6]},
                TensorComponent{
                    last_grid + "/T_y",
                    observation_value * tensor_components_and_coords[4]},
                TensorComponent{
                    last_grid + "/T_z",
                    observation_value * tensor_components_and_coords[2]}}}});
    };
    for (size_t i = 0; i < observation_ids.size(); ++i) {
      write_to_file(observation_ids[i], observation_values[i]);
    }
  }
  // Open the read volume file and check that the observation id and values are
  // correct.
  const auto& volume_file =
      my_file.get<h5::VolumeData>("/element_data", version_number);
  const auto read_observation_ids = volume_file.list_observation_ids();
  CHECK(alg::all_of(read_observation_ids, [&observation_ids](const size_t id) {
    return alg::found(observation_ids, id);
  }));
  // Check that the volume data is correct
  const auto check_time = [
    &volume_file, &tensor_components_and_coords, &grid_names
  ](const size_t observation_id, const double observation_value) noexcept {
    CHECK(std::vector<std::vector<size_t>>{{2, 2, 2}, {2, 2, 2}} ==
          volume_file.get_extents(observation_id));
    CHECK(volume_file.get_observation_value(observation_id) ==
          observation_value);

    // Check that all of the grid names were written correctly by checking their
    // equality of elements
    const std::vector<std::string> read_grid_names =
        volume_file.get_grid_names(observation_id);
    [&read_grid_names, &grid_names]() {
      auto sortable_grid_names = grid_names;
      auto sortable_read_grid_names = read_grid_names;
      std::sort(sortable_grid_names.begin(), sortable_grid_names.end(),
                std::less<>{});
      std::sort(sortable_read_grid_names.begin(),
                sortable_read_grid_names.end(), std::less<>{});
      REQUIRE(sortable_read_grid_names == sortable_grid_names);
    }();
    // Find the order the grids were written in
    std::vector<size_t> grid_positions(read_grid_names.size());
    for (size_t i = 0; i < grid_positions.size(); i++) {
      auto grid_name = grid_names[i];
      auto position =
          std::find(read_grid_names.begin(), read_grid_names.end(), grid_name);
      // We know the grid name is in the read_grid_names because of the previous
      // so we know `position` is an actual pointer to an element
      grid_positions[i] =
          static_cast<size_t>(std::distance(read_grid_names.begin(), position));
    }

    const std::vector<std::string> expected_components{
        "S", "x-coord", "y-coord", "z-coord", "T_x", "T_y", "T_z",
    };
    const auto read_components =
        volume_file.list_tensor_components(observation_id);
    CHECK(alg::all_of(read_components,
                      [&expected_components](const std::string& id) {
                        return alg::found(expected_components, id);
                      }));
    // Helper Function to get number of points on a particular grid
    const auto accumulate_extents =
        [](const std::vector<size_t>& grid_extents) {
          return alg::accumulate(grid_extents, 1, std::multiplies<>{});
        };

    const auto read_extents = volume_file.get_extents(observation_id);
    std::vector<size_t> element_num_points(
        boost::make_transform_iterator(read_extents.begin(),
                                       accumulate_extents),
        boost::make_transform_iterator(read_extents.end(), accumulate_extents));
    const auto read_points_by_element = [&element_num_points]() {
      std::vector<size_t> read_points(element_num_points.size());
      read_points[0] = 0;
      for (size_t index = 1; index < element_num_points.size(); index++) {
        read_points[index] =
            read_points[index - 1] + element_num_points[index - 1];
      }
      return read_points;
    }();
    // Given a DataVector, corresponding to contiguous data read out of a
    // file, find the data which was written by the grid whose extents are
    // found at position `grid_index` in the vector of extents.
    const auto get_grid_data = [&element_num_points, &read_points_by_element](
                                   DataVector all_data,
                                   const size_t grid_index) {
      DataVector result(element_num_points[grid_index]);
      // clang-tidy: do not use pointer arithmetic
      std::copy(&all_data[read_points_by_element[grid_index]],
                &all_data[read_points_by_element[grid_index]] +  // NOLINT
                    element_num_points[grid_index],
                result.begin());
      return result;
    };
    // The order the fake tensor data was assigned to the tensor components
    std::vector<std::vector<size_t>> grid_data_orders{{0, 1, 2, 3, 4, 5, 6},
                                                      {1, 0, 5, 3, 6, 4, 2}};
    // The tensor components can be written in any order to the file, we loop
    // over the expected components rather than the read components because they
    // are in a particular order.
    for (size_t i = 0; i < expected_components.size(); i++) {
      const auto& component = expected_components[i];
      // for each grid
      for (size_t j = 0; j < grid_names.size(); j++) {
        CHECK(get_grid_data(
                  volume_file.get_tensor_component(observation_id, component),
                  grid_positions[j]) ==
              observation_value *
                  tensor_components_and_coords[grid_data_orders[j][i]]);
      }
    }
  };
  for (size_t i = 0; i < observation_ids.size(); ++i) {
    check_time(observation_ids[i], observation_values[i]);
  }
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}

// [[OutputRegex, The expected format of the tensor component names is
// 'GROUP_NAME/COMPONENT_NAME' but could not find a '/' in]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.IO.H5.VolumeData.ComponentFormat0",
                               "[Unit][IO][H5]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  const std::string h5_file_name("Unit.IO.H5.VolumeData.ComponentFormat.h5");
  const uint32_t version_number = 4;
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
  h5::H5File<h5::AccessType::ReadWrite> my_file(h5_file_name);
  auto& volume_file =
      my_file.insert<h5::VolumeData>("/element_data", version_number);
  volume_file.write_volume_data(100, 10.0,
                                std::vector<ExtentsAndTensorVolumeData>{
                                    {{2}, {TensorComponent{"S", {1.0, 2.0}}}}});
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
  // clang-format off
}

// [[OutputRegex, The expected format of the tensor component names is
// 'GROUP_NAME/COMPONENT_NAME' but could not find a '/' in]]
[[noreturn]]
SPECTRE_TEST_CASE("Unit.IO.H5.VolumeData.ComponentFormat1",
                               "[Unit][IO][H5]") {
  // clang-format on
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  const std::string h5_file_name("Unit.IO.H5.VolumeData.ComponentFormat1.h5");
  const uint32_t version_number = 4;
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
  h5::H5File<h5::AccessType::ReadWrite> my_file(h5_file_name);
  auto& volume_file =
      my_file.insert<h5::VolumeData>("/element_data", version_number);
  volume_file.write_volume_data(
      100, 10.0,
      std::vector<ExtentsAndTensorVolumeData>{
          ExtentsAndTensorVolumeData({2}, {TensorComponent{"A/S", {1.0, 2.0}},
                                           TensorComponent{"S", {1.0, 2.0}}})});
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
  // clang-format off
}


// [[OutputRegex, Trying to write tensor component 'S' which already exists
// in HDF5 file in group 'element_data.vol/ObservationId100']]
[[noreturn]] SPECTRE_TEST_CASE("Unit.IO.H5.VolumeData.WriteTwice",
                               "[Unit][IO][H5]") {
  // clang-format on
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  const std::string h5_file_name("Unit.IO.H5.VolumeData.WriteTwice.h5");
  const uint32_t version_number = 4;
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
  h5::H5File<h5::AccessType::ReadWrite> my_file(h5_file_name);
  auto& volume_file =
      my_file.insert<h5::VolumeData>("/element_data", version_number);
  volume_file.write_volume_data(100, 10.0,
                                std::vector<ExtentsAndTensorVolumeData>{
                                    {{2},
                                     {TensorComponent{"A/S", {1.0, 2.0}},
                                      TensorComponent{"A/S", {1.0, 2.0}}}}});
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
  // clang-format off
}
