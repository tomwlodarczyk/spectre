// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/MassFlux.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

namespace hydro {
namespace {
template <size_t Dim, typename Frame, typename DataType>
void test_mass_flux(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(&mass_flux<DataType, Dim, Frame>,
                                    "TestFunctions", "mass_flux",
                                    {{{-10.0, 10.0}}}, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Hydro.MassFlux",
                  "[Unit][Hydro]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/Hydro/");
  const DataVector dv(5);
  test_mass_flux<1, Frame::Inertial>(dv);
  test_mass_flux<1, Frame::Grid>(dv);
  test_mass_flux<2, Frame::Inertial>(dv);
  test_mass_flux<2, Frame::Grid>(dv);
  test_mass_flux<3, Frame::Inertial>(dv);
  test_mass_flux<3, Frame::Grid>(dv);

  test_mass_flux<1, Frame::Inertial>(0.0);
  test_mass_flux<1, Frame::Grid>(0.0);
  test_mass_flux<2, Frame::Inertial>(0.0);
  test_mass_flux<2, Frame::Grid>(0.0);
  test_mass_flux<3, Frame::Inertial>(0.0);
  test_mass_flux<3, Frame::Grid>(0.0);

  // Check compute item works correctly in DataBox
  CHECK(Tags::MassFluxCompute<DataVector, 2, Frame::Inertial>::name() ==
        "MassFlux");
  Scalar<DataVector> rho{{{DataVector{5, 1.0}}}};
  tnsr::I<DataVector, 3> velocity{
      {{DataVector{5, 0.25}, DataVector{5, 0.1}, DataVector{5, 0.35}}}};
  Scalar<DataVector> lorentz{{{DataVector{5, 0.2}}}};
  Scalar<DataVector> lapse{{{DataVector{5, 0.3}}}};
  tnsr::I<DataVector, 3> shift{
      {{DataVector{5, 0.1}, DataVector{5, 0.2}, DataVector{5, 0.3}}}};
  Scalar<DataVector> sqrt_det_g{{{DataVector{5, 0.25}}}};
  const auto box = db::create<
      db::AddSimpleTags<Tags::RestMassDensity<DataVector>,
                        Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>,
                        Tags::LorentzFactor<DataVector>,
                        ::gr::Tags::Lapse<DataVector>,
                        ::gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                        ::gr::Tags::SqrtDetSpatialMetric<DataVector>>,
      db::AddComputeTags<
          Tags::MassFluxCompute<DataVector, 3, Frame::Inertial>>>(
      rho, velocity, lorentz, lapse, shift, sqrt_det_g);
  CHECK(db::get<Tags::MassFlux<DataVector, 3, Frame::Inertial>>(box) ==
        mass_flux(rho, velocity, lorentz, lapse, shift, sqrt_det_g));
}
}  // namespace hydro
