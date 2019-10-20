// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DenseVector.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/LinearSolver/ConjugateGradient/ElementActions.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearSolver/ConjugateGradient/InitializeElement.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_include <boost/variant/get.hpp>
// IWYU pragma: no_forward_declare db::DataBox

namespace {

struct VectorTag : db::SimpleTag {
  using type = DenseVector<double>;
  static std::string name() noexcept { return "VectorTag"; }
};

using fields_tag = VectorTag;
using operand_tag = LinearSolver::Tags::Operand<fields_tag>;
using residual_tag = LinearSolver::Tags::Residual<fields_tag>;

template <typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<
              tmpl::list<VectorTag, operand_tag,
                         LinearSolver::Tags::IterationId, residual_tag,
                         LinearSolver::Tags::HasConverged>,
              tmpl::list<
                  ::Tags::NextCompute<LinearSolver::Tags::IterationId>>>>>,

      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing,
                             tmpl::list<LinearSolver::cg_detail::PrepareStep>>>;
};

struct Metavariables {
  using component_list = tmpl::list<ElementArray<Metavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Numerical.LinearSolver.ConjugateGradient.ElementActions",
    "[Unit][NumericalAlgorithms][LinearSolver][Actions]") {
  using element_array = ElementArray<Metavariables>;

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};

  // Setup mock element array
  ActionTesting::emplace_component_and_initialize<element_array>(
      make_not_null(&runner), 0,
      {DenseVector<double>(3, 0.), DenseVector<double>(3, 2.),
       std::numeric_limits<size_t>::max(), DenseVector<double>(3, 1.),
       db::item_type<LinearSolver::Tags::HasConverged>{}});

  // DataBox shortcuts
  const auto get_tag = [&runner](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<element_array, tag>(runner, 0);
  };

  runner.set_phase(Metavariables::Phase::Testing);

  // Can't test the other element actions because reductions are not yet
  // supported. The full algorithm is tested in
  // `Test_ConjugateGradientAlgorithm.cpp` and
  // `Test_DistributedConjugateGradientAlgorithm.cpp`.

  SECTION("InitializeHasConverged") {
    ActionTesting::simple_action<
        element_array,
        LinearSolver::cg_detail::InitializeHasConverged<fields_tag>>(
        make_not_null(&runner), 0,
        db::item_type<LinearSolver::Tags::HasConverged>{
            {1, 0., 0.}, 1, 0., 0.});
    CHECK(get_tag(LinearSolver::Tags::HasConverged{}));
  }
  SECTION("PrepareStep") {
    ActionTesting::next_action<element_array>(make_not_null(&runner), 0);
    CHECK(get_tag(LinearSolver::Tags::IterationId{}) == 0);
    CHECK(get_tag(Tags::Next<LinearSolver::Tags::IterationId>{}) == 1);
  }
  SECTION("UpdateOperand") {
    ActionTesting::next_action<element_array>(make_not_null(&runner), 0);
    ActionTesting::simple_action<
        element_array, LinearSolver::cg_detail::UpdateOperand<fields_tag>>(
        make_not_null(&runner), 0, 2.,
        db::item_type<LinearSolver::Tags::HasConverged>{
            {1, 0., 0.}, 1, 0., 0.});
    CHECK(get_tag(LinearSolver::Tags::Operand<VectorTag>{}) ==
          DenseVector<double>(3, 5.));
    CHECK(get_tag(LinearSolver::Tags::HasConverged{}));
  }
}
