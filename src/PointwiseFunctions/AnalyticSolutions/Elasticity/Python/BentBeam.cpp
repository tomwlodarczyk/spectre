// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <gsl/gsl_sf_bessel.h>
#include <ostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "PointwiseFunctions/AnalyticSolutions/Elasticity/BentBeam.hpp"
#include "Utilities/GetOutput.hpp"

namespace py = pybind11;

namespace Elasticity {
namespace Solutions {
namespace py_bindings {

namespace {

void bind_bent_beam_impl(py::module& m) {  // NOLINT
  py::class_<BentBeam>(m, ("BentBeam"))
      .def(py::init<double, double, double, double, double>(),
           py::arg("length"), py::arg("height"),
           py::arg("bending_moment"), py::arg("bulk_modulus"),
           py::arg("shear_modulus"));
}
}  // namespace

void bind_bent_beam(py::module& m) {  // NOLINT
  bind_bent_beam_impl(m);
}
}  // namespace py_bindings
}  // namespace Solutions
}  // namespace Elasticity
