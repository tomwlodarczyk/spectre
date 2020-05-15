// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "Domain/Creators/AlignedLattice.hpp"
#include "Domain/Creators/DomainCreator.hpp"

namespace py = pybind11;

namespace domain {
namespace creators {
namespace py_bindings {

namespace {
template <size_t VolumeDim>
void bind_aligned_lattice_dim(py::module& m) {  // NOLINT
  py::class_<RotatedBricks, DomainCreator<3>>(m, "AlignedLattice")
      .def(py::init<std::array<std::vector<double>, VolumeDim>,
                    std::array<bool, VolumeDim>, std::array<size_t, VolumeDim>,
                    std::array<size_t, VolumeDim>,
                    std::vector<RefinementRegion<VolumeDim>>,
                    std::vector<RefinementRegion<VolumeDim>>,
                    std::vector<std::array<size_t, VolumeDim>>>(),
           py::arg("BlockBounds"), py::arg("IsPeriodicIn"),
           py::arg("InitialLevels"), py::arg("InitialGridPoints"),
           py::arg("RefinedLevels"), py::arg("RefinedGridPoints"),
           py::arg("BlocksToExclude"));
}
}  // namespace

void bind_aligned_lattice(py::module& m) {  // NOLINT
  bind_aligned_lattice_dim<1>(m);
  bind_aligned_lattice_dim<2>(m);
  bind_aligned_lattice_dim<3>(m);
}

}  // namespace py_bindings
}  // namespace creators
}  // namespace domain
