// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "DataStructures/Tensor/EagerMath/Norms.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/ElementId.hpp"
#include "Elliptic/Systems/Elasticity//Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Elasticity/HalfSpaceMirror.hpp"
#include "Utilities/GetOutput.hpp"
#include "tests/Unit/PointwiseFunctions/AnalyticSolutions/Elasticity/TestHelpers.hpp"

namespace py = pybind11;

namespace TestHelpers {
namespace Elasticity {
namespace Solutions {
namespace py_bindings {

namespace {

template <size_t Dim, typename SolutionType>
auto apply_dg_operator_impl(const SolutionType& solution,
                            const DomainCreator<Dim>& domain_creator) {
  const auto operator_applied_to_solution =
      apply_dg_operator(solution, domain_creator);
  // Convert the data to a type that Python understands. We just take a
  // norm of the Displacement field over each element for now.
  std::unordered_map<ElementId<Dim>, double> result{};
  for (const auto& id_and_var : operator_applied_to_solution) {
    const auto& element_id = id_and_var.first;
    result[element_id] =
        l2_norm(get<::Elasticity::Tags::Displacement<Dim>>(id_and_var.second));
  }
  return result;
}

template <size_t Dim>
void bind_apply_dg_operator_to_elasticity_halfspacemirror(
    py::module& m) {  // NOLINT
  m.def(("apply_dg_operator_to_elasticity_halfspacemirror"),
        &apply_dg_operator_impl<Dim, ::Elasticity::Solutions::HalfSpaceMirror>,
        py::arg("solution"), py::arg("domain_creator"));
}
}  // namespace

void bind_apply_dg_operator(py::module& m) {  // NOLINT
  bind_apply_dg_operator_to_elasticity_halfspacemirror<3>(m);
}

}  // namespace py_bindings
}  // namespace Solutions
}  // namespace Elasticity
}  // namespace TestHelpers
