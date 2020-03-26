// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace Elasticity {
namespace Solutions {
namespace py_bindings {
void bind_halfspacemirror(py::module& m);  // NOLINT
}  // namespace py_bindings

PYBIND11_MODULE(_PyElasticitySolutions, m) {  // NOLINT
  py_bindings::bind_halfspacemirror(m);
}

}  // namespace Solutions
}  // namespace Elasticity
