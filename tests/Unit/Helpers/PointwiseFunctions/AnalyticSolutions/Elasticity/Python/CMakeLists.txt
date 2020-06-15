# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyElasticitySolutionsTestHelpers")

spectre_python_add_module(
  Elasticity
  LIBRARY_NAME ${LIBRARY}
  MODULE_PATH "TestHelpers/AnalyticSolutions"
  SOURCES
  ApplyDgOperator.cpp
  Bindings.cpp
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PUBLIC
    DataStructures
    Domain
    DomainCreators
    EllipticDgTestHelpers
    Elasticity
    ElasticitySolutions
    ${BLAS_LIBRARIES}
    ${GSL_LIBRARIES}
  )

  spectre_python_add_dependencies(
    ${LIBRARY}
    PyDataStructures
    PyDomain
    PyDomainCreators
    PyElasticitySolutions
    )
