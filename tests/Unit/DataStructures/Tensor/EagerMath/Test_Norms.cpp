// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/Norms.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {
struct MyScalar : db::SimpleTag {
  static std::string name() noexcept { return "MyScalar"; }
  using type = Scalar<DataVector>;
};
template <size_t Dim, typename Frame>
struct Vector : db::SimpleTag {
  static std::string name() noexcept { return "Vector"; }
  using type = tnsr::I<DataVector, Dim, Frame>;
};
template <size_t Dim, typename Frame>
struct Covector : db::SimpleTag {
  static std::string name() noexcept { return "Covector"; }
  using type = tnsr::i<DataVector, Dim, Frame>;
};
template <size_t Dim, typename Frame>
struct Metric : db::SimpleTag {
  static std::string name() noexcept { return "Metric"; }
  using type = tnsr::ii<DataVector, Dim, Frame>;
};
template <size_t Dim, typename Frame>
struct InverseMetric : db::SimpleTag {
  static std::string name() noexcept { return "InverseMetric"; }
  using type = tnsr::II<DataVector, Dim, Frame>;
};

template <typename Frame>
void test_l2_norm_tag() {
  constexpr size_t npts = 5;

  const DataVector one(npts, 1.);
  const DataVector two(npts, 2.);
  const DataVector minus_three(npts, -3.);
  const DataVector four(npts, 4.);
  const DataVector minus_five(npts, -5.);
  const DataVector twelve(npts, 12.);
  const auto mixed = []() {
    DataVector tmp(npts, 0.);
    for (size_t i = 0; i < npts; ++i) {
      if (i % 2) {
        tmp[i] = 3.5;
      } else {
        tmp[i] = 4.5;
      }
    }
    return tmp;
  }();

  // create test tensors
  const auto psi_1d = [&npts, &minus_five]() {
    db::item_type<Metric<1, Frame>> tmp{npts};
    get<0, 0>(tmp) = minus_five;
    return tmp;
  }();
  const auto psi_2d = [&npts, &one, &two]() {
    db::item_type<Metric<2, Frame>> tmp{npts};
    get<0, 0>(tmp) = two;
    get<0, 1>(tmp) = two;
    get<1, 0>(tmp) = two;
    get<1, 1>(tmp) = one;
    return tmp;
  }();
  const auto psi_3d = [&npts, &one, &two, &minus_three]() {
    db::item_type<Metric<3, Frame>> tmp{npts};
    get<0, 0>(tmp) = one;
    get<0, 1>(tmp) = two;
    get<0, 2>(tmp) = minus_three;
    get<1, 1>(tmp) = two;
    get<1, 2>(tmp) = two;
    get<2, 2>(tmp) = one;
    return tmp;
  }();

  const auto box = db::create<
      db::AddSimpleTags<MyScalar, Vector<1, Frame>, Vector<2, Frame>,
                        Vector<3, Frame>, Covector<1, Frame>,
                        Covector<2, Frame>, Covector<3, Frame>,
                        Metric<1, Frame>, Metric<2, Frame>, Metric<3, Frame>,
                        InverseMetric<1, Frame>, InverseMetric<2, Frame>,
                        InverseMetric<3, Frame>>,
      db::AddComputeTags<Tags::PointwiseL2NormCompute<MyScalar>,
                         Tags::PointwiseL2NormCompute<Vector<1, Frame>>,
                         Tags::PointwiseL2NormCompute<Vector<2, Frame>>,
                         Tags::PointwiseL2NormCompute<Vector<3, Frame>>,
                         Tags::PointwiseL2NormCompute<Covector<1, Frame>>,
                         Tags::PointwiseL2NormCompute<Covector<2, Frame>>,
                         Tags::PointwiseL2NormCompute<Covector<3, Frame>>,
                         Tags::PointwiseL2NormCompute<Metric<1, Frame>>,
                         Tags::PointwiseL2NormCompute<Metric<2, Frame>>,
                         Tags::PointwiseL2NormCompute<Metric<3, Frame>>,
                         Tags::PointwiseL2NormCompute<InverseMetric<1, Frame>>,
                         Tags::PointwiseL2NormCompute<InverseMetric<2, Frame>>,
                         Tags::PointwiseL2NormCompute<InverseMetric<3, Frame>>,
                         Tags::L2NormCompute<MyScalar>,
                         Tags::L2NormCompute<Vector<1, Frame>>,
                         Tags::L2NormCompute<Vector<2, Frame>>,
                         Tags::L2NormCompute<Vector<3, Frame>>,
                         Tags::L2NormCompute<Covector<1, Frame>>,
                         Tags::L2NormCompute<Covector<2, Frame>>,
                         Tags::L2NormCompute<Covector<3, Frame>>,
                         Tags::L2NormCompute<Metric<1, Frame>>,
                         Tags::L2NormCompute<Metric<2, Frame>>,
                         Tags::L2NormCompute<Metric<3, Frame>>,
                         Tags::L2NormCompute<InverseMetric<1, Frame>>,
                         Tags::L2NormCompute<InverseMetric<2, Frame>>,
                         Tags::L2NormCompute<InverseMetric<3, Frame>>>>(
      db::item_type<MyScalar>{{{minus_three}}},
      db::item_type<Vector<1, Frame>>{{{mixed}}},
      db::item_type<Vector<2, Frame>>{{{minus_three, mixed}}},
      db::item_type<Vector<3, Frame>>{{{minus_three, mixed, four}}},
      db::item_type<Covector<1, Frame>>{{{four}}},
      db::item_type<Covector<2, Frame>>{{{four, two}}},
      db::item_type<Covector<3, Frame>>{{{four, two, twelve}}}, psi_1d, psi_2d,
      psi_3d, determinant_and_inverse(psi_1d).second,
      determinant_and_inverse(psi_2d).second,
      determinant_and_inverse(psi_3d).second);

  // Test point-wise L2-norm against precomputed values
  // rank 0
  CHECK_ITERABLE_APPROX(get(db::get<Tags::PointwiseL2Norm<MyScalar>>(box)),
                        DataVector(npts, 3.));
  // rank (1, 0)
  const auto verification_vec_1d = []() {
    DataVector tmp(npts, 0.);
    for (size_t i = 0; i < npts; ++i) {
      if (i % 2) {
        tmp[i] = 3.5;
      } else {
        tmp[i] = 4.5;
      }
    }
    return tmp;
  }();
  const auto verification_vec_2d = []() {
    DataVector tmp(npts, 0.);
    for (size_t i = 0; i < npts; ++i) {
      if (i % 2) {
        tmp[i] = 4.6097722286464435;
      } else {
        tmp[i] = 5.408326913195984;
      }
    }
    return tmp;
  }();
  const auto verification_vec_3d = []() {
    DataVector tmp(npts, 0.);
    for (size_t i = 0; i < npts; ++i) {
      if (i % 2) {
        tmp[i] = 6.103277807866851;
      } else {
        tmp[i] = 6.726812023536855;
      }
    }
    return tmp;
  }();
  CHECK_ITERABLE_APPROX(
      get(db::get<Tags::PointwiseL2Norm<Vector<1, Frame>>>(box)),
      verification_vec_1d);
  CHECK_ITERABLE_APPROX(
      get(db::get<Tags::PointwiseL2Norm<Vector<2, Frame>>>(box)),
      verification_vec_2d);
  CHECK_ITERABLE_APPROX(
      get(db::get<Tags::PointwiseL2Norm<Vector<3, Frame>>>(box)),
      verification_vec_3d);
  // rank (0, 1)
  CHECK_ITERABLE_APPROX(
      get(db::get<Tags::PointwiseL2Norm<Covector<1, Frame>>>(box)),
      DataVector(npts, 4.));
  CHECK_ITERABLE_APPROX(
      get(db::get<Tags::PointwiseL2Norm<Covector<2, Frame>>>(box)),
      DataVector(npts, 4.47213595499958));
  CHECK_ITERABLE_APPROX(
      get(db::get<Tags::PointwiseL2Norm<Covector<3, Frame>>>(box)),
      DataVector(npts, 12.806248474865697));
  // rank (0, 2)
  CHECK_ITERABLE_APPROX(
      get(db::get<Tags::PointwiseL2Norm<Metric<1, Frame>>>(box)),
      DataVector(npts, 5.));
  CHECK_ITERABLE_APPROX(
      get(db::get<Tags::PointwiseL2Norm<Metric<2, Frame>>>(box)),
      DataVector(npts, 3.605551275463989));
  CHECK_ITERABLE_APPROX(
      get(db::get<Tags::PointwiseL2Norm<Metric<3, Frame>>>(box)),
      DataVector(npts, 6.324555320336759));
  // rank (2, 0)
  CHECK_ITERABLE_APPROX(
      get(db::get<Tags::PointwiseL2Norm<InverseMetric<1, Frame>>>(box)),
      DataVector(npts, 0.2));
  CHECK_ITERABLE_APPROX(
      get(db::get<Tags::PointwiseL2Norm<InverseMetric<2, Frame>>>(box)),
      DataVector(npts, 1.8027756377319946));
  CHECK_ITERABLE_APPROX(
      get(db::get<Tags::PointwiseL2Norm<InverseMetric<3, Frame>>>(box)),
      DataVector(npts, 0.4787135538781691));

  // Test L2-norm reduced over domain, against precomputed values
  // rank 0
  CHECK(db::get<Tags::L2Norm<MyScalar>>(box) == approx(3.));
  // rank (1, 0)
  CHECK(db::get<Tags::L2Norm<Vector<1, Frame>>>(box) ==
        approx(4.129164564412516));
  CHECK(db::get<Tags::L2Norm<Vector<2, Frame>>>(box) ==
        approx(5.103920062069938));
  CHECK(db::get<Tags::L2Norm<Vector<3, Frame>>>(box) ==
        approx(6.48459713474939));
  // rank (0, 1)
  CHECK(db::get<Tags::L2Norm<Covector<1, Frame>>>(box) == approx(4.));
  CHECK(db::get<Tags::L2Norm<Covector<2, Frame>>>(box) ==
        approx(4.47213595499958));
  CHECK(db::get<Tags::L2Norm<Covector<3, Frame>>>(box) ==
        approx(12.806248474865697));
  // rank (0, 2)
  CHECK(db::get<Tags::L2Norm<Metric<1, Frame>>>(box) == approx(5.));
  CHECK(db::get<Tags::L2Norm<Metric<2, Frame>>>(box) ==
        approx(3.605551275463989));
  CHECK(db::get<Tags::L2Norm<Metric<3, Frame>>>(box) ==
        approx(6.324555320336759));
  // rank (2, 0)
  CHECK(db::get<Tags::L2Norm<InverseMetric<1, Frame>>>(box) == approx(0.2));
  CHECK(db::get<Tags::L2Norm<InverseMetric<2, Frame>>>(box) ==
        approx(1.8027756377319946));
  CHECK(db::get<Tags::L2Norm<InverseMetric<3, Frame>>>(box) ==
        approx(0.4787135538781691));

  // Check tag names
  using Tag = MyScalar;
  CHECK(Tags::PointwiseL2Norm<Tag>::name() ==
        "PointwiseL2Norm(" + db::tag_name<Tag>() + ")");
  CHECK(Tags::PointwiseL2NormCompute<Tag>::name() ==
        "PointwiseL2Norm(" + db::tag_name<Tag>() + ")");
  CHECK(Tags::L2Norm<Tag>::name() == "L2Norm(" + db::tag_name<Tag>() + ")");
  CHECK(Tags::L2NormCompute<Tag>::name() ==
        "L2Norm(" + db::tag_name<Tag>() + ")");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.EagerMath.Norms",
                  "[DataStructures][Unit]") {
  test_l2_norm_tag<Frame::Grid>();
  test_l2_norm_tag<Frame::Inertial>();
}
