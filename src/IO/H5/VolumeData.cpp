// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/H5/VolumeData.hpp"

#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <hdf5.h>
#include <memory>
#include <ostream>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TensorData.hpp"
#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "IO/Connectivity.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Header.hpp"
#include "IO/H5/Helpers.hpp"
#include "IO/H5/Version.hpp"
#include "Utilities/Numeric.hpp"
/// \cond HIDDEN_SYMBOLS
namespace h5 {
namespace {
// Append the element extents and connectevity to the total extents and
// connectivity
void append_element_extents_and_connectivity(
    const gsl::not_null<std::vector<size_t>*> total_extents,
    const gsl::not_null<std::vector<int>*> total_connectivity,
    const gsl::not_null<int*> total_points_so_far, const size_t dim,
    const ExtentsAndTensorVolumeData& element) noexcept {
  // Process the element extents
  const auto& extents = element.extents;
  if (extents.size() != dim) {
    ERROR("Trying to write data of dimensionality"
          << extents.size() << "but the VolumeData file has dimensionality"
          << dim << ".");
  }
  total_extents->insert(total_extents->end(), extents.begin(), extents.end());
  // Find the number of points in the local connectivity
  const int element_num_points =
      alg::accumulate(extents, 1, std::multiplies<>{});
  // Generate the connectivity data for the element
  // Possible optimization: local_connectivity.reserve(BLAH) if we can figure
  // out size without computing all the connectivities.
  const std::vector<int> connectivity =
      [&extents, &total_points_so_far ]() noexcept {
    std::vector<int> local_connectivity;
    for (const auto& cell : vis::detail::compute_cells(extents)) {
      for (const auto& bounding_indices : cell.bounding_indices) {
        local_connectivity.emplace_back(*total_points_so_far +
                                        static_cast<int>(bounding_indices));
      }
    }
    return local_connectivity;
  }
  ();
  *total_points_so_far += element_num_points;
  total_connectivity->insert(total_connectivity->end(), connectivity.begin(),
                             connectivity.end());
}

// Append the name of an element to the string of grid names
void append_element_name(const gsl::not_null<std::string*> grid_names,
                         const ExtentsAndTensorVolumeData& element) noexcept {
  // Get the name of the element
  const auto& first_tensor_name = element.tensor_components.front().name;
  ASSERT(first_tensor_name.find_last_of('/') != std::string::npos,
         "The expected format of the tensor component names is "
         "'GROUP_NAME/COMPONENT_NAME' but could not find a '/' in '"
             << first_tensor_name << "'.");
  const auto spatial_name =
      first_tensor_name.substr(0, first_tensor_name.find_last_of('/'));
  *grid_names += spatial_name + VolumeData::separator();
}
}  // namespace

VolumeData::VolumeData(const bool subfile_exists, detail::OpenGroup&& group,
                       const hid_t /*location*/, const std::string& name,
                       const uint32_t version) noexcept
    : group_(std::move(group)),
      name_(name.size() > extension().size()
                ? (extension() == name.substr(name.size() - extension().size())
                       ? name
                       : name + extension())
                : name + extension()),
      version_(version),
      volume_data_group_(group_.id(), name_, h5::AccessType::ReadWrite) {
  if (subfile_exists) {
    // We treat this as an internal version for now. We'll need to deal with
    // proper versioning later.
    const Version open_version(true, detail::OpenGroup{},
                               volume_data_group_.id(), "version");
    version_ = open_version.get_version();
    const Header header(true, detail::OpenGroup{}, volume_data_group_.id(),
                        "header");
    header_ = header.get_header();
  } else {  // file does not exist
    // Subfiles are closed as they go out of scope, so we have the extra
    // braces here to add the necessary scope
    {
      Version open_version(false, detail::OpenGroup{}, volume_data_group_.id(),
                           "version", version_);
    }
    {
      Header header(false, detail::OpenGroup{}, volume_data_group_.id(),
                    "header");
      header_ = header.get_header();
    }
  }
}

// Write Volume Data stored in a vector of `ExtentsAndTensorVolumeData` to
// an `observation_group` in a `VolumeData` file.
void VolumeData::write_volume_data(
    const size_t observation_id, const double observation_value,
    const std::vector<ExtentsAndTensorVolumeData>& elements) noexcept {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadWrite);
  if (contains_attribute(observation_group.id(), "", "observation_value")) {
    ERROR("Trying to write ObservationId "
          << std::to_string(observation_id) << " with observation_value "
          << observation_group.id() << " which already exists in file at "
          << path << ".");
  }
  h5::write_to_attribute(observation_group.id(), "observation_value",
                         observation_value);
  // Get first element to extract the component names and dimension
  const auto get_component_name = [](const auto& component) noexcept {
    ASSERT(component.name.find_last_of('/') != std::string::npos,
           "The expected format of the tensor component names is "
           "'GROUP_NAME/COMPONENT_NAME' but could not find a '/' in '"
               << component.name << "'.");
    return component.name.substr(component.name.find_last_of('/') + 1);
  };
  const std::vector<std::string> component_names(
      boost::make_transform_iterator(elements.front().tensor_components.begin(),
                                     get_component_name),
      boost::make_transform_iterator(elements.front().tensor_components.end(),
                                     get_component_name));

  // The dimension of the grid is the number of extents per element.
  // Only written once per VolumeData file, as if two observation id's
  if (not contains_attribute(volume_data_group_.id(), "", "dimension")) {
    h5::write_to_attribute(volume_data_group_.id(), "dimension",
                           elements.front().extents.size());
  }
  const auto dim =
      h5::read_value_attribute<size_t>(volume_data_group_.id(), "dimension");
  // Extract Tensor Data one component at a time
  std::vector<size_t> total_extents;
  std::string grid_names;
  std::vector<int> total_connectivity;
  // We need to keep track the total number of points inserted into the
  // connectivity after each iteration to be sure each point gets a
  // unique representation in the topology data
  int total_points_so_far = 0;
  // Loop over tensor componenents
  for (size_t i = 0; i < component_names.size(); i++) {
    std::string component_name = component_names[i];

    std::vector<double> contiguous_tensor_data{};
    for (auto& element : elements) {
      if (i == 0) {  // True if first tensor component being accessed
        append_element_name(&grid_names, element);
        append_element_extents_and_connectivity(
            &total_extents, &total_connectivity, &total_points_so_far, dim,
            element);
      }
      const DataVector& tensor_data_on_grid = element.tensor_components[i].data;
      contiguous_tensor_data.insert(contiguous_tensor_data.end(),
                                    tensor_data_on_grid.begin(),
                                    tensor_data_on_grid.end());

    }  // for each element

    // Write the data for the tensor component
    if (h5::contains_dataset_or_group(observation_group.id(), "",
                                      component_name)) {
      ERROR("Trying to write tensor component '"
            << component_name
            << "' which already exists in HDF5 file in group '" << name_ << '/'
            << "ObservationId" << std::to_string(observation_id) << "'");
    }

    h5::write_data(observation_group.id(), contiguous_tensor_data,
                   {contiguous_tensor_data.size()}, component_name);
  }  // for each component

  // Write the grid extents contiguously, the first `dim` belong to the
  // First grid, the second `dim` belong to the second grid, and so on,
  // Ordering is `x, y, z, ... `
  h5::write_data(observation_group.id(), total_extents, {total_extents.size()},
                 "total_extents");
  // Write the names of the grids as vector of chars with individual names
  // separated by `separator()`
  std::vector<char> grid_names_as_chars(grid_names.begin(), grid_names.end());
  h5::write_data(observation_group.id(), grid_names_as_chars,
                 {grid_names_as_chars.size()}, "grid_names");
  // Write the Connectivity
  h5::write_data(observation_group.id(), total_connectivity,
                 {total_connectivity.size()}, "connectivity");
}

std::vector<size_t> VolumeData::list_observation_ids() const noexcept {
  const auto names = get_group_names(volume_data_group_.id(), "");
  const auto helper = [](const std::string& s) noexcept {
    return std::stoul(s.substr(std::string("ObservationId").size()));
  };
  return {boost::make_transform_iterator(names.begin(), helper),
          boost::make_transform_iterator(names.end(), helper)};
}

double VolumeData::get_observation_value(const size_t observation_id) const
    noexcept {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadOnly);
  return h5::read_value_attribute<double>(observation_group.id(),
                                          "observation_value");
}

std::vector<std::string> VolumeData::list_tensor_components(
    const size_t observation_id) const noexcept {
  auto tensor_components =
      get_group_names(volume_data_group_.id(),
                      "ObservationId" + std::to_string(observation_id));
  std::remove(tensor_components.begin(), tensor_components.end(),
              "connectivity");
  std::remove(tensor_components.begin(), tensor_components.end(),
              "total_extents");
  std::remove(tensor_components.begin(), tensor_components.end(), "grid_names");
  // std::remove moves the element to the end of the vector, so we still need to
  // actually erase it from the vector
  tensor_components.erase(tensor_components.end() - 3, tensor_components.end());

  return tensor_components;
}

std::vector<std::string> VolumeData::get_grid_names(
    const size_t observation_id) const noexcept {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadOnly);
  const std::vector<char> names =
      h5::read_data<1, std::vector<char>>(observation_group.id(), "grid_names");
  const std::string all_names(names.begin(), names.end());
  std::vector<std::string> grid_names{};
  boost::split(grid_names, all_names, [this](const char c) noexcept {
    return c == this->separator();
  });
  // boost::split counts the last separator as a split even though there are no
  // characters after it, so the last entry of the vector is empty
  grid_names.pop_back();
  return grid_names;
}

DataVector VolumeData::get_tensor_component(
    const size_t observation_id, const std::string& tensor_component) const
    noexcept {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadOnly);

  const hid_t dataset_id =
      h5::open_dataset(observation_group.id(), tensor_component);
  const hid_t dataspace_id = h5::open_dataspace(dataset_id);
  const auto rank =
      static_cast<size_t>(H5Sget_simple_extent_ndims(dataspace_id));
  h5::close_dataspace(dataspace_id);
  h5::close_dataset(dataset_id);
  switch (rank) {
    case 1:
      return h5::read_data<1, DataVector>(observation_group.id(),
                                          tensor_component);
    case 2:
      return h5::read_data<2, DataVector>(observation_group.id(),
                                          tensor_component);
    case 3:
      return h5::read_data<3, DataVector>(observation_group.id(),
                                          tensor_component);
    default:
      ERROR("Rank must be 1, 2, or 3. Received data with Rank = " << rank);
  }
}

std::vector<std::vector<size_t>> VolumeData::get_extents(
    const size_t observation_id) const noexcept {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadOnly);
  const auto dim =
      h5::read_value_attribute<size_t>(volume_data_group_.id(), "dimension");
  const auto extents_per_element = static_cast<long>(dim);
  const auto total_extents = h5::read_data<1, std::vector<size_t>>(
      observation_group.id(), "total_extents");
  std::vector<std::vector<size_t>> individual_extents;
  individual_extents.reserve(total_extents.size() / dim);
  for (auto iter = total_extents.begin(); iter != total_extents.end();
       iter += extents_per_element) {
    individual_extents.emplace_back(iter, iter + extents_per_element);
  }
  return individual_extents;
}

size_t VolumeData::get_dimension() const noexcept {
  return h5::read_value_attribute<double>(volume_data_group_.id(), "dimension");
}

}  // namespace h5
/// \endcond HIDDEN_SYMBOLS
