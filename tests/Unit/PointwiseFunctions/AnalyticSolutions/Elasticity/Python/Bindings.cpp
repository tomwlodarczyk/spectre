// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace TestHelpers {
namespace Elasticity {
namespace Solutions {

namespace py_bindings {
void bind_apply_dg_operator(py::module& m);  // NOLINT
}  // namespace py_bindings

PYBIND11_MODULE(_PyElasticitySolutionsTestHelpers, m) {  // NOLINT
  py_bindings::bind_apply_dg_operator(m);
}

}  // namespace Solutions
}  // namespace Elasticity
}  // namespace TestHelpers
