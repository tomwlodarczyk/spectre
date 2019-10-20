// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Parallel/PhaseDependentActionList.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

namespace Parallel {
namespace detail {
template <class Action, class = cpp17::void_t<>>
struct get_inbox_tags_from_action {
  using type = tmpl::list<>;
};

template <class Action>
struct get_inbox_tags_from_action<Action,
                                  cpp17::void_t<typename Action::inbox_tags>> {
  using type = typename Action::inbox_tags;
};
}  // namespace detail

/*!
 * \ingroup ParallelGroup
 * \brief Given a list of Actions, get a list of the unique inbox tags
 */
template <class ActionsList>
using get_inbox_tags = tmpl::remove_duplicates<tmpl::join<tmpl::transform<
    ActionsList, detail::get_inbox_tags_from_action<tmpl::_1>>>>;

namespace detail {
// ParallelStruct is a metavariables, component, or action struct
template <class ParallelStruct, class = cpp17::void_t<>>
struct get_const_global_cache_tags_from_parallel_struct {
  using type = tmpl::list<>;
};

template <class ParallelStruct>
struct get_const_global_cache_tags_from_parallel_struct<
    ParallelStruct,
    cpp17::void_t<typename ParallelStruct::const_global_cache_tags>> {
  using type = typename ParallelStruct::const_global_cache_tags;
};

template <class Component>
struct get_const_global_cache_tags_from_pdal {
  using type = tmpl::join<tmpl::transform<
      tmpl::flatten<tmpl::transform<
          typename Component::phase_dependent_action_list,
          get_action_list_from_phase_dep_action_list<tmpl::_1>>>,
      get_const_global_cache_tags_from_parallel_struct<tmpl::_1>>>;
};
}  // namespace detail

/*!
 * \ingroup ParallelGroup
 * \brief Given a list of Actions, get a list of the unique tags specified in
 * the actions' `const_global_cache_tags` aliases.
 */
template <class ActionsList>
using get_const_global_cache_tags_from_actions =
    tmpl::remove_duplicates<tmpl::join<tmpl::transform<
        ActionsList,
        detail::get_const_global_cache_tags_from_parallel_struct<tmpl::_1>>>>;

/*!
 * \ingroup ParallelGroup
 * \brief Given the metavariables, get a list of the unique tags that will
 * specify the items in the ConstGlobalCache.
 */
template <typename Metavariables>
using get_const_global_cache_tags =
    tmpl::remove_duplicates<tmpl::flatten<tmpl::list<
        typename detail::get_const_global_cache_tags_from_parallel_struct<
            Metavariables>::type,
        tmpl::transform<
            typename Metavariables::component_list,
            detail::get_const_global_cache_tags_from_parallel_struct<tmpl::_1>>,
        tmpl::transform<
            typename Metavariables::component_list,
            detail::get_const_global_cache_tags_from_pdal<tmpl::_1>>>>>;

namespace detail {
template <typename PhaseAction>
struct get_initialization_actions_list {
  using type = tmpl::list<>;
};

template <typename PhaseType, typename InitializationActionsList>
struct get_initialization_actions_list<Parallel::PhaseActions<
    PhaseType, PhaseType::Initialization, InitializationActionsList>> {
  using type = InitializationActionsList;
};
}  // namespace detail

/// \ingroup ParallelGroup
/// \brief Given the phase dependent action list, return the list of
/// actions in the Initialization phase (or an empty list if the Initialization
/// phase is absent from the phase dependent action list)
template <typename PhaseDepActionList>
using get_initialization_actions_list = tmpl::flatten<tmpl::transform<
    PhaseDepActionList, detail::get_initialization_actions_list<tmpl::_1>>>;

namespace detail {
template <typename Action, typename = cpp17::void_t<>>
struct get_initialization_tags_from_action {
  using type = tmpl::list<>;
};

template <typename Action>
struct get_initialization_tags_from_action<
    Action, cpp17::void_t<typename Action::initialization_tags>> {
  using type = typename Action::initialization_tags;
};

template <typename Action, typename = cpp17::void_t<>>
struct get_initialization_tags_to_keep_from_action {
  using type = tmpl::list<>;
};

template <typename Action>
struct get_initialization_tags_to_keep_from_action<
    Action, cpp17::void_t<typename Action::initialization_tags_to_keep>> {
  using type = typename Action::initialization_tags_to_keep;
};
}  // namespace detail

/// \ingroup ParallelGroup
/// \brief Given a list of initialization actions, and possibly a list of tags
/// needed for allocation of an array component, returns a list of the
/// unique initialization_tags for all the actions (and the allocate function).
template <typename InitializationActionsList,
          typename AllocationTagsList = tmpl::list<>>
using get_initialization_tags = tmpl::remove_duplicates<tmpl::flatten<
    tmpl::list<AllocationTagsList,
               tmpl::transform<
                   InitializationActionsList,
                   detail::get_initialization_tags_from_action<tmpl::_1>>>>>;

/// \ingroup ParallelGroup
/// \brief Given a list of initialization actions, returns a list of
/// the unique initialization_tags_to_keep for all the actions.  These
/// are the tags that are not removed from the DataBox after
/// initialization.
template <typename InitializationActionsList>
using get_initialization_tags_to_keep =
    tmpl::remove_duplicates<tmpl::flatten<tmpl::transform<
        InitializationActionsList,
        detail::get_initialization_tags_to_keep_from_action<tmpl::_1>>>>;

namespace detail {
template <typename InitializationTag>
struct get_option_tags_from_initialization_tag {
  using type = typename InitializationTag::option_tags;
};
}  // namespace detail

/// \ingroup ParallelGroup
/// \brief Given a list of initialization tags, returns a list of the
/// unique option tags required to construct them.
template <typename InitializationTagsList>
using get_option_tags = tmpl::remove_duplicates<tmpl::flatten<tmpl::transform<
    InitializationTagsList,
    detail::get_option_tags_from_initialization_tag<tmpl::_1>>>>;

namespace detail {
template <typename Tag, typename... OptionTags, typename... OptionTagsForTag>
typename Tag::type create_initialization_item_from_options(
    const tuples::TaggedTuple<OptionTags...>& options,
    tmpl::list<OptionTagsForTag...> /*meta*/) noexcept {
  return Tag::create_from_options(tuples::get<OptionTagsForTag>(options)...);
}
}  // namespace detail

/// \ingroup ParallelGroup
/// \brief Given a list of tags and a tagged tuple containing items
/// created from input options, return a tagged tuple of items constructed
/// by calls to create_from_options for each tag in the list.
template <typename... Tags, typename... OptionTags>
tuples::TaggedTuple<Tags...> create_from_options(
    const tuples::TaggedTuple<OptionTags...>& options,
    tmpl::list<Tags...> /*meta*/) noexcept {
  return {detail::create_initialization_item_from_options<Tags>(
      options, typename Tags::option_tags{})...};
}

/// \cond
namespace Algorithms {
struct Singleton;
struct Array;
struct Group;
struct Nodegroup;
}  // namespace Algorithms

template <class ChareType>
struct get_array_index;

template <>
struct get_array_index<Parallel::Algorithms::Singleton> {
  template <class ParallelComponent>
  using f = cpp17::void_type;
};

template <>
struct get_array_index<Parallel::Algorithms::Array> {
  template <class ParallelComponent>
  using f = typename ParallelComponent::array_index;
};

template <>
struct get_array_index<Parallel::Algorithms::Group> {
  template <class ParallelComponent>
  using f = int;
};

template <>
struct get_array_index<Parallel::Algorithms::Nodegroup> {
  template <class ParallelComponent>
  using f = int;
};

template <typename ParallelComponent>
using proxy_from_parallel_component =
    typename ParallelComponent::chare_type::template cproxy<
        ParallelComponent,
        typename get_array_index<typename ParallelComponent::chare_type>::
            template f<ParallelComponent>>;

template <typename ParallelComponent>
using index_from_parallel_component =
    typename ParallelComponent::chare_type::template ckindex<
        ParallelComponent,
        typename get_array_index<typename ParallelComponent::chare_type>::
            template f<ParallelComponent>>;

template <class ParallelComponent, class... Args>
struct charm_types_with_parameters {
  using cproxy =
      typename ParallelComponent::chare_type::template cproxy<ParallelComponent,
                                                              Args...>;
  using cbase =
      typename ParallelComponent::chare_type::template cbase<ParallelComponent,
                                                             Args...>;
  using algorithm =
      typename ParallelComponent::chare_type::template algorithm_type<
          ParallelComponent, Args...>;
  using ckindex = typename ParallelComponent::chare_type::template ckindex<
      ParallelComponent, Args...>;
};
/// \endcond
}  // namespace Parallel
