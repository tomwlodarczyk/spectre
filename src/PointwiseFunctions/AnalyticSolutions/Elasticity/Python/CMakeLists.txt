# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY PyElasticitySolutions)

spectre_python_add_module(
  Elasticity
  LIBRARY_NAME ${LIBRARY}
  MODULE_PATH "PointwiseFunctions/AnalyticSolutions"
  SOURCES
  Bindings.cpp
  HalfSpaceMirror.cpp
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PUBLIC
  ElasticitySolutions
  ${GSL_LIBRARIES}
  )
