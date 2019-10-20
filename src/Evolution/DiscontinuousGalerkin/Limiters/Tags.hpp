// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Options/Options.hpp"

namespace OptionTags {
/*!
 * \ingroup OptionGroupsGroup
 * \brief Holds the `OptionTags::Limiter` option in the input file
 */
struct LimiterGroup {
  static std::string name() noexcept { return "Limiter"; }
  static constexpr OptionString help = "Options for limiting troubled cells";
};

/*!
 * \ingroup OptionTagsGroup
 * \brief The global cache tag that retrieves the parameters for the limiter
 * from the input file
 */
template <typename LimiterType>
struct Limiter {
  static std::string name() noexcept { return option_name<LimiterType>(); }
  static constexpr OptionString help = "Options for the limiter";
  using type = LimiterType;
  using group = LimiterGroup;
};
}  // namespace OptionTags

namespace Tags {
/*!
 * \brief The global cache tag for the limiter
 */
template <typename LimiterType>
struct Limiter : db::SimpleTag {
  static std::string name() noexcept { return "Limiter"; }
  using type = LimiterType;
  using option_tags = tmpl::list<::OptionTags::Limiter<LimiterType>>;
  static LimiterType create_from_options(const LimiterType& limiter) noexcept {
    return limiter;
  }
};
}  // namespace Tags
