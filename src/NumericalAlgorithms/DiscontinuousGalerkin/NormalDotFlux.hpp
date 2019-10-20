// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Domain/FaceNormal.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
// IWYU pragma: no_forward_declare Variables
// IWYU pragma: no_forward_declare Tags::Flux
/// \endcond

/*!
 * \brief Contract a surface normal covector with the first index of a flux
 * tensor
 *
 * \details
 * Returns \f$n_i F^i_{j\ldots}\f$, where the flux tensor \f$F\f$ must have an
 * upper spatial first index and may have arbitray extra indices.
 */
template <size_t VolumeDim, typename Fr, typename Symm,
          typename... RemainingIndices,
          typename ResultTensor = Tensor<DataVector, tmpl::pop_front<Symm>,
                                         index_list<RemainingIndices...>>>
void normal_dot_flux(
    const gsl::not_null<ResultTensor*> normal_dot_flux,
    const tnsr::i<DataVector, VolumeDim, Fr>& normal,
    const Tensor<DataVector, Symm,
                 index_list<SpatialIndex<VolumeDim, UpLo::Up, Fr>,
                            RemainingIndices...>>& flux_tensor) noexcept {
  for (auto it = normal_dot_flux->begin(); it != normal_dot_flux->end(); it++) {
    const auto result_indices = normal_dot_flux->get_tensor_index(it);
    *it = get<0>(normal) * flux_tensor.get(prepend(result_indices, size_t{0}));
    for (size_t d = 1; d < VolumeDim; d++) {
      *it += normal.get(d) * flux_tensor.get(prepend(result_indices, d));
    }
  }
}

namespace Tags {

/// \ingroup ConservativeGroup
/// \ingroup DataBoxTagsGroup
/// \brief Prefix computing a boundary unit normal vector dotted into
/// the flux from a flux on the boundary.
template <typename Tag, size_t VolumeDim, typename Fr>
struct NormalDotFluxCompute : db::add_tag_prefix<NormalDotFlux, Tag>,
                              db::ComputeTag {
  using base = db::add_tag_prefix<NormalDotFlux, Tag>;

 private:
  using flux_tag = db::add_tag_prefix<Flux, Tag, tmpl::size_t<VolumeDim>, Fr>;
  using normal_tag =
      Tags::Normalized<Tags::UnnormalizedFaceNormal<VolumeDim, Fr>>;

 public:
  static auto function(const db::const_item_type<flux_tag>& flux,
                       const db::const_item_type<normal_tag>& normal) noexcept {
    using tags_list = typename db::const_item_type<Tag>::tags_list;
    auto result = make_with_value<
        ::Variables<db::wrap_tags_in<NormalDotFlux, tags_list>>>(flux, 0.);

    tmpl::for_each<tags_list>([&result, &flux,
                               &normal ](auto local_tag) noexcept {
      using tensor_tag = tmpl::type_from<decltype(local_tag)>;
      normal_dot_flux(make_not_null(&get<NormalDotFlux<tensor_tag>>(result)),
                      normal,
                      get<Flux<tensor_tag, tmpl::size_t<VolumeDim>, Fr>>(flux));
    });
    return result;
  }
  using argument_tags = tmpl::list<flux_tag, normal_tag>;
};
}  // namespace Tags
