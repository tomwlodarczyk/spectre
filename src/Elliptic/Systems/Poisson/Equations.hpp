// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Domain/FaceNormal.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"  // IWYU pragma: keep
#include "Options/Options.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tags::deriv
// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables

/// \cond
class DataVector;
template <size_t>
class Mesh;
namespace Tags {
template <typename>
struct Normalized;
}  // namespace Tags
namespace LinearSolver {
namespace Tags {
template <typename>
struct Operand;
}  // namespace Tags
}  // namespace LinearSolver
namespace Poisson {
namespace Tags {
struct Field;
template <size_t>
struct AuxiliaryField;
}  // namespace Tags
}  // namespace Poisson
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Poisson {

/*!
 * \brief Compute the fluxes \f$F^i=\partial_i u(x)\f$ for the Poisson
 * equation on a flat metric in Cartesian coordinates.
 */
template <size_t Dim>
void euclidean_fluxes(
    gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> flux_for_field,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& field_gradient) noexcept;

/*!
 * \brief Compute the fluxes \f$F^i_j=\delta^i_j u(x)\f$ for the auxiliary
 * field in the first-order formulation of the Poisson equation.
 *
 * \see Poisson::FirstOrderSystem
 */
template <size_t Dim>
void auxiliary_fluxes(gsl::not_null<tnsr::Ij<DataVector, Dim, Frame::Inertial>*>
                          flux_for_gradient,
                      const Scalar<DataVector>& field) noexcept;

/*!
 * \brief Compute the fluxes \f$F^i_A\f$ for the Poisson equation on a flat
 * metric in Cartesian coordinates.
 *
 * \see Poisson::FirstOrderSystem
 */
template <size_t Dim>
struct EuclideanFluxes {
  using argument_tags = tmpl::list<>;
  static void apply(
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          flux_for_field,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          field_gradient) noexcept {
    euclidean_fluxes(flux_for_field, field_gradient);
  }
  static void apply(
      const gsl::not_null<tnsr::Ij<DataVector, Dim, Frame::Inertial>*>
          flux_for_gradient,
      const Scalar<DataVector>& field) noexcept {
    auxiliary_fluxes(flux_for_gradient, field);
  }
  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

/*!
 * \brief Compute the sources \f$S_A\f$ for the Poisson equation.
 *
 * \see Poisson::FirstOrderSystem
 */
struct Sources {
  using argument_tags = tmpl::list<>;
  static void apply(const gsl::not_null<Scalar<DataVector>*> source_for_field,
                    const Scalar<DataVector>& /*field*/) noexcept {
    get(*source_for_field) = 0.;
  }
};

/*!
 * \brief The internal penalty flux for the first oder formulation of the
 * Poisson equation.
 *
 * \details For the sourced equation this is \f$-\frac{\nabla u_\mathrm{int} +
 * \nabla u_\mathrm{ext}}{2} + \sigma \left(u_\mathrm{int} -
 * u_\mathrm{ext}\right)\f$ and for the auxiliary equation it is \f$\frac{
 * u_\mathrm{int} + u_\mathrm{ext}}{2}\f$. The penalty factor \f$\sigma\f$ is
 * responsible for removing zero eigenmodes and impacts the conditioning of the
 * linear operator to be solved. It can be chosen as
 * \f$\sigma=C\frac{N_\mathrm{points}^2}{h}\f$ where \f$N_\mathrm{points}\f$ is
 * the number of collocation points (i.e. the polynomial degree plus 1),
 * \f$h\f$ is a measure of the element size in inertial coordinates and \f$C\leq
 * 1\f$ is a free parameter (see e.g. \cite HesthavenWarburton, section 7.2).
 */
template <size_t Dim>
struct FirstOrderInternalPenaltyFlux {
 public:
  struct PenaltyParameter {
    using type = double;
    // Currently this is used as the full prefactor to the penalty term. When it
    // becomes possible to compute a measure of the size $h$ of an element and
    // the number of collocation points $p$ on both sides of the mortar, this
    // should be changed to be just the parameter multiplying $\frac{p^2}{h}$.
    static constexpr OptionString help = {
        "The prefactor to the penalty term of the flux."};
  };
  using options = tmpl::list<PenaltyParameter>;
  static constexpr OptionString help = {
      "Computes the internal penalty flux for a Poisson system."};
  static std::string name() noexcept { return "InternalPenalty"; }

  FirstOrderInternalPenaltyFlux() = default;
  explicit FirstOrderInternalPenaltyFlux(double penalty_parameter)
      : penalty_parameter_(penalty_parameter) {}

  // clang-tidy: non-const reference
  void pup(PUP::er& p) noexcept { p | penalty_parameter_; }  // NOLINT

  struct NormalTimesFieldFlux : db::SimpleTag {
    using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
    static std::string name() noexcept { return "NormalTimesFieldFlux"; }
  };

  struct NormalDotGradFieldFlux : db::SimpleTag {
    using type = Scalar<DataVector>;
    static std::string name() noexcept { return "NormalDotGradFieldFlux"; }
  };

  // These tags are sliced to the interface of the element and passed to
  // `package_data` to provide the data needed to compute the numerical fluxes.
  using argument_tags =
      tmpl::list<LinearSolver::Tags::Operand<Tags::Field>,
                 ::Tags::div<::Tags::Flux<
                     LinearSolver::Tags::Operand<::Tags::deriv<
                         Tags::Field, tmpl::size_t<Dim>, Frame::Inertial>>,
                     tmpl::size_t<Dim>, Frame::Inertial>>,
                 ::Tags::Normalized<::Tags::UnnormalizedFaceNormal<Dim>>>;

  // This is the data needed to compute the numerical flux.
  // `SendBoundaryFluxes` calls `package_data` to store these tags in a
  // Variables. Local and remote values of this data are then combined in the
  // `()` operator.
  using package_tags = tmpl::list<LinearSolver::Tags::Operand<Tags::Field>,
                                  NormalTimesFieldFlux, NormalDotGradFieldFlux>;

  // Following the packaged_data pointer, this function expects as arguments the
  // types in `argument_tags`.
  void package_data(gsl::not_null<Variables<package_tags>*> packaged_data,
                    const Scalar<DataVector>& field,
                    const tnsr::i<DataVector, Dim, Frame::Inertial>& grad_field,
                    const tnsr::i<DataVector, Dim, Frame::Inertial>&
                        interface_unit_normal) const noexcept;

  // This function combines local and remote data to the numerical fluxes.
  // The numerical fluxes as not-null pointers are the first arguments. The
  // other arguments are the packaged types for the interior side followed by
  // the packaged types for the exterior side.
  void operator()(
      gsl::not_null<Scalar<DataVector>*> numerical_flux_for_field,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          numerical_flux_for_auxiliary_field,
      const Scalar<DataVector>& field_interior,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          normal_times_field_interior,
      const Scalar<DataVector>& normal_dot_grad_field_interior,
      const Scalar<DataVector>& field_exterior,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          minus_normal_times_field_exterior,
      const Scalar<DataVector>& minus_normal_dot_grad_field_exterior) const
      noexcept;

  // This function computes the boundary contributions from Dirichlet boundary
  // conditions. This data is what remains to be added to the boundaries when
  // homogeneous (i.e. zero) boundary conditions are assumed in the calculation
  // of the numerical fluxes, but we wish to impose inhomogeneous (i.e. nonzero)
  // boundary conditions. Since this contribution does not depend on the
  // numerical field values, but only on the Dirichlet boundary data, it may be
  // added as contribution to the source of the elliptic systems. Then, it
  // remains to solve the homogeneous problem with the modified source.
  // The first arguments to this function are the boundary contributions to
  // compute as not-null pointers, in the order they appear in the
  // `system::fields_tag`. They are followed by the field values of the tags in
  // `system::primal_fields`. The last argument is the
  // normalized unit covector to the element face.
  void compute_dirichlet_boundary(
      gsl::not_null<Scalar<DataVector>*> numerical_flux_for_field,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          numerical_flux_for_auxiliary_field,
      const Scalar<DataVector>& dirichlet_field,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal)
      const noexcept;

 private:
  double penalty_parameter_{};
};

}  // namespace Poisson
