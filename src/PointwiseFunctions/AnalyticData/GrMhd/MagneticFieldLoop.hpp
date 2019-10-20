// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/MakeArray.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma:  no_include <pup.h>

/// \cond
namespace PUP {
class er;  // IWYU pragma: keep
}  // namespace PUP
/// \endcond

namespace grmhd {
namespace AnalyticData {

/*!
 * \brief Analytic initial data for an advecting magnetic field loop.
 *
 * This test, originally proposed in \cite Gardiner2005hy and presented in a
 * slightly modified form by \cite Mignone2010br, a region with annular
 * cross section with the specified `InnerRadius` and `OuterRadius` is given a
 * non-zero azimuthal magnetic field of constant magnitude `MagFieldStrength`
 * with zero magnetic field outside the loop.  Inside the `InnerRadius` the
 * magnetic field strength falls to zero quadratically. The loop is embedded in
 * an ideal fluid with the given `AdiabaticIndex`, `RestMassDensity` and
 * `Pressure` with a uniform `AdvectionVelocity`.  The magnetic field loop
 * should advect across the grid, maintaining its shape and strength, as long
 * as the magnetic pressure is negligible compared to the thermal pressure.
 *
 * This test diagnoses how well the evolution scheme preserves the no-monopole
 * condition, as well as the diffusivity of the scheme.
 *
 * The standard test setup is done on \f$x \in [-1,1]\f$, \f$y \in [-0.5,
 * 0.5]\f$, with periodic boundary conditions and with the following values
 * given for the options:
 * -  InnerRadius: 0.06
 * -  OuterRadius: 0.3
 * -  RestMassDensity: 1.0
 * -  Pressure: 1.0
 * -  AdvectionVelocity: [0.08164965809277261, 0.040824829046386304, 0.0]
 * -  MagFieldStrength: 0.001
 * -  AdiabaticIndex: 1.66666666666666667
 *
 */
class MagneticFieldLoop : public MarkAsAnalyticData {
 public:
  using equation_of_state_type = EquationsOfState::IdealFluid<true>;

  /// The pressure throughout the fluid.
  struct Pressure {
    using type = double;
    static constexpr OptionString help = {
        "The constant pressure throughout the fluid."};
    static type lower_bound() { return 0.0; }
  };

  /// The rest mass density throughout the fluid.
  struct RestMassDensity {
    using type = double;
    static constexpr OptionString help = {
        "The constant density throughout the fluid."};
    static type lower_bound() { return 0.0; }
  };

  /// The adiabatic index for the ideal fluid.
  struct AdiabaticIndex {
    using type = double;
    static constexpr OptionString help = {
        "The adiabatic index for the ideal fluid."};
    static type lower_bound() { return 1.0; }
  };

  /// The fluid velocity.
  struct AdvectionVelocity {
    using type = std::array<double, 3>;
    static constexpr OptionString help = {"The advection velocity."};
    static type lower_bound() { return {{-1.0, -1.0, -1.0}}; }
    static type upper_bound() { return {{1.0, 1.0, 1.0}}; }
  };

  /// The strength of the magnetic field.
  struct MagFieldStrength {
    using type = double;
    static constexpr OptionString help = {
        "The magnitude of the magnetic field."};
    static type lower_bound() { return 0.0; }
  };

  /// The inner radius of the magnetic loop.
  struct InnerRadius {
    using type = double;
    static constexpr OptionString help = {
        "The inner radius of the magnetic loop."};
    static type lower_bound() { return 0.0; }
  };

  /// The outer radius of the magnetic loop.
  struct OuterRadius {
    using type = double;
    static constexpr OptionString help = {
        "The outer radius of the magnetic loop."};
    static type lower_bound() { return 0.0; }
  };

  using options =
      tmpl::list<Pressure, RestMassDensity, AdiabaticIndex, AdvectionVelocity,
                 MagFieldStrength, InnerRadius, OuterRadius>;
  static constexpr OptionString help = {
      "Periodic advection of a magnetic field loop in Minkowski."};

  MagneticFieldLoop() = default;
  MagneticFieldLoop(const MagneticFieldLoop& /*rhs*/) = delete;
  MagneticFieldLoop& operator=(const MagneticFieldLoop& /*rhs*/) = delete;
  MagneticFieldLoop(MagneticFieldLoop&& /*rhs*/) noexcept = default;
  MagneticFieldLoop& operator=(MagneticFieldLoop&& /*rhs*/) noexcept = default;
  ~MagneticFieldLoop() = default;

  MagneticFieldLoop(double pressure, double rest_mass_density,
                    double adiabatic_index,
                    const std::array<double, 3>& advection_velocity,
                    double magnetic_field_magnitude, double inner_radius,
                    double outer_radius, const OptionContext& context = {});

  explicit MagneticFieldLoop(CkMigrateMessage* /*unused*/) noexcept {}

  // @{
  /// Retrieve the GRMHD variables at a given position.
  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/) const
      noexcept
      -> tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<hydro::Tags::Pressure<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::SpatialVelocity<
                     DataType, 3, Frame::Inertial>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<
          hydro::Tags::SpatialVelocity<DataType, 3, Frame::Inertial>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::MagneticField<
                     DataType, 3, Frame::Inertial>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<
          hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const
      noexcept
      -> tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>;
  // @}

  /// Retrieve a collection of hydrodynamic variables at position x
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<Tags...> /*meta*/) const noexcept {
    static_assert(sizeof...(Tags) > 1,
                  "The generic template will recurse infinitely if only one "
                  "tag is being retrieved.");
    return {tuples::get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  /// Retrieve the metric variables
  template <typename DataType, typename Tag>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataType, 3>& x,
                                     tmpl::list<Tag> /*meta*/) const noexcept {
    constexpr double dummy_time = 0.0;
    return background_spacetime_.variables(x, dummy_time, tmpl::list<Tag>{});
  }

  const EquationsOfState::IdealFluid<true>& equation_of_state() const noexcept {
    return equation_of_state_;
  }

  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept;  //  NOLINT

 private:
  double pressure_ = std::numeric_limits<double>::signaling_NaN();
  double rest_mass_density_ = std::numeric_limits<double>::signaling_NaN();
  double adiabatic_index_ = std::numeric_limits<double>::signaling_NaN();
  std::array<double, 3> advection_velocity_ =
      make_array<3>(std::numeric_limits<double>::signaling_NaN());
  double magnetic_field_magnitude_ =
      std::numeric_limits<double>::signaling_NaN();
  double inner_radius_ = std::numeric_limits<double>::signaling_NaN();
  double outer_radius_ = std::numeric_limits<double>::signaling_NaN();

  EquationsOfState::IdealFluid<true> equation_of_state_{};
  gr::Solutions::Minkowski<3> background_spacetime_{};

  friend bool operator==(const MagneticFieldLoop& lhs,
                         const MagneticFieldLoop& rhs) noexcept;

  friend bool operator!=(const MagneticFieldLoop& lhs,
                         const MagneticFieldLoop& rhs) noexcept;
};

}  // namespace AnalyticData
}  // namespace grmhd
