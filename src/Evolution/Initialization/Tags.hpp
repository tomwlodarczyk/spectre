// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Time/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Initialization {
/// \ingroup InitializationGroup
/// \brief %Tags used during initialization of parallel components.
namespace Tags {
struct InitialTime : db::SimpleTag {
  static std::string name() noexcept { return "InitialTime"; }
  using type = double;
  using option_tags = tmpl::list<OptionTags::InitialTime>;

  static double create_from_options(const double initial_time) noexcept {
    return initial_time;
  }
};

struct InitialTimeDelta : db::SimpleTag {
  static std::string name() noexcept { return "InitialTimeDelta"; }
  using type = double;
  using option_tags = tmpl::list<OptionTags::InitialTimeStep>;

  static double create_from_options(const double initial_time_step) noexcept {
    return initial_time_step;
  }
};

template <bool UsingLocalTimeStepping>
struct InitialSlabSize : db::SimpleTag {
  static std::string name() noexcept { return "InitialSlabSize"; }
  using type = double;
  using option_tags = tmpl::list<OptionTags::InitialSlabSize>;

  static double create_from_options(const double initial_slab_size) noexcept {
    return initial_slab_size;
  }
};

template <>
struct InitialSlabSize<false> : db::SimpleTag {
  static std::string name() noexcept { return "InitialSlabSize"; }
  using type = double;
  using option_tags = tmpl::list<OptionTags::InitialTimeStep>;

  static double create_from_options(const double initial_time_step) noexcept {
    return std::abs(initial_time_step);
  }
};
}  // namespace Tags
}  // namespace Initialization
