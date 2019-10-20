// Distributed under the MIT License.
// See LICENSE.txt for details.

// This file is separated from `Test_ComplexDiagonalModalOperator.cpp` in an
// effort to parallelize the test builds.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <tuple>

#include "DataStructures/ComplexDiagonalModalOperator.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DiagonalModalOperator.hpp"
#include "DataStructures/ModalVector.hpp"  // IWYU pragma: keep
#include "ErrorHandling/Error.hpp"         // IWYU pragma: keep
#include "Utilities/Functional.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep
#include "Utilities/TypeTraits.hpp"  // IWYU pragma: keep
#include "tests/Unit/DataStructures/VectorImplTestHelper.hpp"

// IWYU pragma: no_include <algorithm>

void test_additional_complex_diagonal_modal_operator_math() noexcept {
  const TestHelpers::VectorImpl::Bound generic{{-100.0, 100.0}};

  const auto acting_on_modal_vector = std::make_tuple(std::make_tuple(
      funcl::Multiplies<>{}, std::make_tuple(generic, generic)));

  // the operation isn't really "inplace", but we carefully forbid the operation
  // between two ModalVectors, which will be avoided in the inplace test case,
  // which checks only combinations with the ComplexDiagonalModalOperator as the
  // first argument.
  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Inplace, ComplexDiagonalModalOperator,
      ModalVector, ComplexModalVector>(acting_on_modal_vector);
  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::GivenOrderOfArgumentsOnly,
      DiagonalModalOperator, ComplexModalVector>(acting_on_modal_vector);
  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::GivenOrderOfArgumentsOnly,
      ComplexModalVector, DiagonalModalOperator>(acting_on_modal_vector);
  // testing the other ordering
  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::GivenOrderOfArgumentsOnly, ModalVector,
      ComplexDiagonalModalOperator>(acting_on_modal_vector);
  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::GivenOrderOfArgumentsOnly,
      ComplexModalVector, ComplexDiagonalModalOperator>(acting_on_modal_vector);
  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::GivenOrderOfArgumentsOnly,
      ComplexModalVector, DiagonalModalOperator>(acting_on_modal_vector);

  const auto cascaded_ops = std::make_tuple(
      std::make_tuple(funcl::Multiplies<funcl::Plus<>, funcl::Identity>{},
                      std::make_tuple(generic, generic, generic)),
      std::make_tuple(funcl::Minus<funcl::Plus<>, funcl::Identity>{},
                      std::make_tuple(generic, generic, generic)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Strict, ComplexDiagonalModalOperator,
      DiagonalModalOperator>(cascaded_ops);

  const auto array_binary_ops = std::make_tuple(
      std::make_tuple(funcl::Minus<>{}, std::make_tuple(generic, generic)),
      std::make_tuple(funcl::Plus<>{}, std::make_tuple(generic, generic)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Strict,
      std::array<ComplexDiagonalModalOperator, 2>>(array_binary_ops);
}

SPECTRE_TEST_CASE(
    "Unit.DataStructures.ComplexDiagonalModalOperator.AdditionalMath",
    "[DataStructures][Unit]") {
  test_additional_complex_diagonal_modal_operator_math();
}
