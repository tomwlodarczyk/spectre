// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

#include "Evolution/Systems/RadiationTransport/Tags.hpp"

class DataVector;

/// Namespace for all radiation transport algorithms
namespace RadiationTransport {
/// Namespace for the grey-M1 radiation transport scheme
namespace M1Grey {
/// %Tags for the evolution of neutrinos using a grey M1 scheme.
namespace Tags {

/// The characteristic speeds
struct CharacteristicSpeeds : db::SimpleTag {
  using type = std::array<DataVector, 4>;
  static std::string name() noexcept { return "CharacteristicSpeeds"; }
};

/// The densitized energy density of neutrinos of a given species
/// \f${\tilde E}\f$
template <typename Fr, class Species>
struct TildeE : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept {
    return Frame::prefix<Fr>() + "TildeE_" + neutrinos::get_name(Species{});
  }
};

/// The densitized momentum density of neutrinos of a given species
/// \f${\tilde S_i}\f$
template <typename Fr, class Species>
struct TildeS : db::SimpleTag {
  using type = tnsr::i<DataVector, 3, Fr>;
  static std::string name() noexcept {
    return Frame::prefix<Fr>() + "TildeS_" + neutrinos::get_name(Species{});
  }
};

/// The densitized pressure tensor of neutrinos of a given species
/// \f${\tilde P^{ij}}\f$
/// computed from \f${\tilde E}\f$, \f${\tilde S_i}\f$ using the M1 closure
template <typename Fr, class Species>
struct TildeP : db::SimpleTag {
  using type = tnsr::II<DataVector, 3, Fr>;
  static std::string name() noexcept {
    return Frame::prefix<Fr>() + "TildeP_" + neutrinos::get_name(Species{});
  }
};

/// The upper index momentum density of a neutrino species.
/// This tag does not know the species of neutrinos being used.
/// \f${\tilde S^i}\f$
template <typename Fr>
struct TildeSVector : db::SimpleTag {
  using type = tnsr::I<DataVector, 3, Fr>;
  static std::string name() noexcept {
    return Frame::prefix<Fr>() + "TildeSVector";
  }
};

/// The M1 closure factor of neutrinos of
/// a given species \f${\xi}\f$
template <class Species>
struct ClosureFactor : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept {
    return "ClosureFactor_" + neutrinos::get_name(Species{});
  }
};

/// The fluid-frame densitized energy density of neutrinos of
/// a given species \f${\tilde J}\f$
template <class Species>
struct TildeJ : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept {
    return "TildeJ_" + neutrinos::get_name(Species{});
  }
};

/// The normal component of the fluid-frame momentum density of neutrinos of
/// a given species \f${\tilde H}^a t_a\f$
template <class Species>
struct TildeHNormal : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept {
    return "TildeHNormal_" + neutrinos::get_name(Species{});
  }
};

/// The spatial components of the fluid-frame momentum density of neutrinos of
/// a given species \f${\tilde H}^a {\gamma}_{ia}\f$
template <typename Fr, class Species>
struct TildeHSpatial : db::SimpleTag {
  using type = tnsr::i<DataVector, 3, Fr>;
  static std::string name() noexcept {
    return Frame::prefix<Fr>() + "TildeHSpatial_" +
           neutrinos::get_name(Species{});
  }
};

}  // namespace Tags
}  // namespace M1Grey
}  // namespace RadiationTransport
