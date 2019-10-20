# Distributed under the MIT License.
# See LICENSE.txt for details.

# Get the current working branch and commit hash
execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND git describe --abbrev=0 --always --tags
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

file(WRITE
  "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
  "CMake Version: ${CMAKE_VERSION}\n"
  "SpECTRE Version: ${SpECTRE_VERSION}\n"
  "SpECTRE branch: ${GIT_BRANCH}\n"
  "SpECTRE hash: ${GIT_COMMIT_HASH}\n"
  "Hostname: ${HOSTNAME}\n"
  "Host system: ${CMAKE_HOST_SYSTEM}\n"
  "Host system version: ${CMAKE_HOST_SYSTEM_VERSION}\n"
  "Host system processor: ${CMAKE_HOST_SYSTEM_PROCESSOR}\n"
  "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}\n"
  "CMAKE_CXX_LINK_FLAGS: ${CMAKE_CXX_LINK_FLAGS}\n"
  "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}\n"
  "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}\n"
  "CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}\n"
  "CMAKE_C_LINK_FLAGS: ${CMAKE_C_LINK_FLAGS}\n"
  "CMAKE_C_FLAGS_DEBUG: ${CMAKE_C_FLAGS_DEBUG}\n"
  "CMAKE_C_FLAGS_RELEASE: ${CMAKE_C_FLAGS_RELEASE}\n"
  "CMAKE_Fortran_FLAGS: ${CMAKE_Fortran_FLAGS}\n"
  "CMAKE_Fortran_LINK_FLAGS: ${CMAKE_Fortran_LINK_FLAGS}\n"
  "CMAKE_Fortran_FLAGS_DEBUG: ${CMAKE_Fortran_FLAGS_DEBUG}\n"
  "CMAKE_Fortran_FLAGS_RELEASE: ${CMAKE_Fortran_FLAGS_RELEASE}\n"
  "CMAKE_EXE_LINKER_FLAGS: ${CMAKE_EXE_LINKER_FLAGS}\n"
  "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}\n"
  "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}\n"
  "CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}\n"
  "CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}\n"
  "CMAKE_C_COMPILER_VERSION: ${CMAKE_C_COMPILER_VERSION}\n"
  "CMAKE_Fortran_COMPILER: ${CMAKE_Fortran_COMPILER}\n"
  "CMAKE_Fortran_COMPILER_VERSION: ${CMAKE_Fortran_COMPILER_VERSION}\n"
  "Python Version: ${PYTHON_VERSION_STRING}\n"
  )
