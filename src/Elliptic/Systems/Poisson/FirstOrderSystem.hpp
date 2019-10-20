// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Poisson::FirstOrderSystem

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Elliptic/Systems/Poisson/Equations.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
template <typename>
class Variables;
}  // namespace Tags

namespace LinearSolver {
namespace Tags {
template <typename>
struct Operand;
}  // namespace Tags
}  // namespace LinearSolver
/// \endcond

namespace Poisson {

/*!
 * \brief The Poisson equation formulated as a set of coupled first-order PDEs.
 *
 * \details This system formulates the Poisson equation \f$-\Delta_\gamma u(x) =
 * f(x)\f$ on a background metric \f$\gamma_{ij}\f$ as the set of coupled
 * first-order PDEs
 *
 * \f[
 * -\frac{1}{\sqrt{\gamma}} \partial_i \sqrt{\gamma}\gamma^{ij} v_j(x) = f(x) \\
 * -\partial_i u(x) + v_i(x) = 0
 * \f]
 *
 * where we have chosen the field gradient as an auxiliary variable \f$v_i\f$.
 * This scheme also goes by the name of _mixed_ or _flux_ formulation (see e.g.
 * \cite Arnold2002). The reason for the latter name is that we can write the
 * set of coupled first-order PDEs in flux-form
 *
 * \f[
 * -\partial_i F^i_A + S_A = f_A(x)
 * \f]
 *
 * by choosing the fluxes and sources in terms of the system variables
 * \f$u(x)\f$ and \f$v_i(x)\f$ as
 *
 * \f{align*}
 * F^i_u &= \sqrt{\gamma}\gamma^{ij} v_j(x) \\
 * S_u &= 0 \\
 * f_u &= \sqrt{\gamma} f(x) \\
 * F^i_{v_j} &= u \delta^i_j \\
 * S_{v_j} &= v_j \\
 * f_{v_j} &= 0 \text{.}
 * \f}
 *
 * Note that we use the system variables to index the fluxes and sources, which
 * we also do in the code by using DataBox tags.
 * Also note that we have defined the _fixed sources_ \f$f_A\f$ as those source
 * terms that are independent of the system variables.
 *
 * The field gradient \f$v_i\f$ is treated on the same footing as the field
 * \f$u\f$ in this first-order formulation. This allows us to make use of the DG
 * architecture developed for coupled first-order hyperbolic PDEs in flux-form,
 * in particular the flux communication and lifting code. It does, however,
 * introduce auxiliary degrees of freedom that can be avoided in the
 * second-order (or _primal_) formulation. Furthermore, the linear operator that
 * represents the DG discretization for this system is not symmetric. This
 * property further increase the computational cost (see \ref LinearSolverGroup)
 * and is remedied in the second-order formulation.
 */
template <size_t Dim>
struct FirstOrderSystem {
 private:
  using field = Tags::Field;
  using field_gradient =
      ::Tags::deriv<field, tmpl::size_t<Dim>, Frame::Inertial>;

 public:
  static constexpr size_t volume_dim = Dim;

  // The physical fields to solve for
  using primal_fields = tmpl::list<field>;
  using auxiliary_fields = tmpl::list<field_gradient>;
  using fields_tag =
      ::Tags::Variables<tmpl::append<primal_fields, auxiliary_fields>>;

  // The variables to compute bulk contributions and fluxes for.
  using variables_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
  using primal_variables =
      db::wrap_tags_in<LinearSolver::Tags::Operand, primal_fields>;
  using auxiliary_variables =
      db::wrap_tags_in<LinearSolver::Tags::Operand, auxiliary_fields>;

  using fluxes = EuclideanFluxes<Dim>;
  using sources = Sources;

  // The tag of the operator to compute magnitudes on the manifold, e.g. to
  // normalize vectors on the faces of an element
  template <typename Tag>
  using magnitude_tag = ::Tags::EuclideanMagnitude<Tag>;
};
}  // namespace Poisson
