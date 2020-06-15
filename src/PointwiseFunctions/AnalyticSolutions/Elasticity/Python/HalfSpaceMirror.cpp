// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <gsl/gsl_sf_bessel.h>
#include <ostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "PointwiseFunctions/AnalyticSolutions/Elasticity/HalfSpaceMirror.hpp"
#include "Utilities/GetOutput.hpp"

namespace py = pybind11;

namespace Elasticity {
namespace Solutions {
namespace py_bindings {

namespace {

void bind_halfspacemirror_impl(py::module& m) {  // NOLINT
  py::class_<HalfSpaceMirror>(m, ("HalfSpaceMirror"))
      .def(py::init<double, double, double, size_t, double, double>(),
           py::arg("beam_width"), py::arg("bulk_modulus"),
           py::arg("shear_modulus"), py::arg("no_intervals") = 350,
           py::arg("absolute_tolerance") = 1e-12,
           py::arg("relative_tolerance") = 1e-10);
}
}  // namespace

void bind_halfspacemirror(py::module& m) {  // NOLINT
  bind_halfspacemirror_impl(m);
}
}  // namespace py_bindings
}  // namespace Solutions
}  // namespace Elasticity
