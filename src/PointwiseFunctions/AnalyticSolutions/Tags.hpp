// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "Options/Options.hpp"
#include "Parallel/Serialize.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace OptionTags {
/// \ingroup OptionGroupsGroup
/// Holds the `OptionTags::AnalyticSolution` option in the input file
struct AnalyticSolutionGroup {
  static std::string name() noexcept { return "AnalyticSolution"; }
  static constexpr Options::String help =
      "Analytic solution used for the initial data and errors";
};

/// \ingroup OptionTagsGroup
/// The analytic solution, with the type of the analytic solution set as the
/// template parameter
template <typename SolutionType>
struct AnalyticSolution {
  static std::string name() noexcept {
    return Options::name<SolutionType>();
  }
  static constexpr Options::String help = "Options for the analytic solution";
  using type = SolutionType;
  using group = AnalyticSolutionGroup;
};
/// \ingroup OptionTagsGroup
/// The boundary condition to be applied at all external boundaries.
template <typename BoundaryConditionType>
struct BoundaryCondition {
  static constexpr Options::String help = "Boundary condition to be used";
  using type = BoundaryConditionType;
};
}  // namespace OptionTags

namespace Tags {
/// Can be used to retrieve the analytic solution from the cache without having
/// to know the template parameters of AnalyticSolution.
struct AnalyticSolutionBase : AnalyticSolutionOrData {};

/// Base tag with which to retrieve the BoundaryConditionType
struct BoundaryConditionBase : db::BaseTag {};

/// \ingroup OptionTagsGroup
/// The analytic solution, with the type of the analytic solution set as the
/// template parameter
template <typename SolutionType>
struct AnalyticSolution : AnalyticSolutionBase, db::SimpleTag {
  using type = SolutionType;
  using option_tags = tmpl::list<::OptionTags::AnalyticSolution<SolutionType>>;

  static constexpr bool pass_metavariables = false;
  static SolutionType create_from_options(
      const SolutionType& analytic_solution) noexcept {
    return deserialize<type>(serialize<type>(analytic_solution).data());
  }
};
/// \ingroup OptionTagsGroup
/// The boundary condition to be applied at all external boundaries.
template <typename BoundaryConditionType>
struct BoundaryCondition : BoundaryConditionBase, db::SimpleTag {
  using type = BoundaryConditionType;
  using option_tags =
      tmpl::list<::OptionTags::BoundaryCondition<BoundaryConditionType>>;

  static constexpr bool pass_metavariables = false;
  static BoundaryConditionType create_from_options(
      const BoundaryConditionType& boundary_condition) noexcept {
    return boundary_condition;
  }
};

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating the analytic solution value for a quantity
///
/// \snippet AnalyticSolutions/Test_Tags.cpp analytic_name
template <typename Tag>
struct Analytic : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};

/*!
 * \brief Prefix indicating the error of a value represented by `Tag`
 */
template <typename Tag>
struct Error : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};
}  // namespace Tags
