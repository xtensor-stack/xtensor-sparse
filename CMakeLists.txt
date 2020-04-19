cmake_minimum_required(VERSION 3.1)
project(xtensor-sparse)

set(XTENSOR_SPARSE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Versionning
# ===========

file(STRINGS
     "${XTENSOR_SPARSE_INCLUDE_DIR}/xtensor-sparse/xtensor_sparse_config.hpp"
     xtensor_sparse_version_defines
     REGEX "#define XTENSOR_SPARSE_VERSION_(MAJOR|MINOR|PATCH)")
foreach(ver ${xtensor_sparse_version_defines})
    if(ver MATCHES "#define XTENSOR_SPARSE_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(XTENSOR_SPARSE_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif()
endforeach()
set(${PROJECT_NAME}_VERSION
    ${XTENSOR_SPARSE_VERSION_MAJOR}.${XTENSOR_SPARSE_VERSION_MINOR}.${XTENSOR_SPARSE_VERSION_PATCH})
message(STATUS "Building xtensor-sparse v${${PROJECT_NAME}_VERSION}")

# Dependencies
# ============

set(xtensor_REQUIRED_VERSION 0.21.4)
if(TARGET xtensor)
    set(xtensor_VERSION ${XTENSOR_VERSION_MAJOR}.${XTENSOR_VERSION_MINOR}.${XTENSOR_VERSION_PATCH})
    # Note: This is not SEMVER compatible comparison
    if( NOT ${xtENSOR_VERSION} VERSION_GREATER_EQUAL ${xtENSOR_REQUIRED_VERSION})
        message(ERROR "Mismatch xtensor versions. Found '${xtensor_VERSION}' but requires: '${xtensor_REQUIRED_VERSION}'")
    else()
        message(STATUS "Found xtensor v${xtensor_VERSION}")
    endif()
else()
    find_package(xtensor ${xtensor_REQUIRED_VERSION} REQUIRED)
    message(STATUS "Found xtensor: ${xtensor_INCLUDE_DIRS}/xtensor")
endif()

# Build
# =====

set(XTENSOR_SPARSE_HEADERS
    ${XTENSOR_SPARSE_INCLUDE_DIR}/xtensor-sparse/xcoo_container.hpp
    ${XTENSOR_SPARSE_INCLUDE_DIR}/xtensor-sparse/xcoo_array.hpp
    ${XTENSOR_SPARSE_INCLUDE_DIR}/xtensor-sparse/xcoo_tensor.hpp
    ${XTENSOR_SPARSE_INCLUDE_DIR}/xtensor-sparse/xmap_tensor.hpp
    ${XTENSOR_SPARSE_INCLUDE_DIR}/xtensor-sparse/xmap_array.hpp
    ${XTENSOR_SPARSE_INCLUDE_DIR}/xtensor-sparse/xmap_container.hpp
    ${XTENSOR_SPARSE_INCLUDE_DIR}/xtensor-sparse/xsparse_container.hpp
    ${XTENSOR_SPARSE_INCLUDE_DIR}/xtensor-sparse/xsparse_reference.hpp
    ${XTENSOR_SPARSE_INCLUDE_DIR}/xtensor-sparse/xtensor_sparse_config.hpp
)

add_library(xtensor-sparse INTERFACE)

target_include_directories(xtensor-sparse INTERFACE
    $<BUILD_INTERFACE:${XTENSOR_SPARSE_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>)

target_compile_features(xtensor-sparse INTERFACE cxx_std_14)

target_link_libraries(xtensor-sparse INTERFACE xtensor)

OPTION(XTENSOR_ENABLE_ASSERT "xtensor bound check" OFF)
OPTION(XTENSOR_CHECK_DIMENSION "xtensor dimension check" OFF)
OPTION(BUILD_TESTS "xtensor-sparse test suite" OFF)
OPTION(DOWNLOAD_GTEST "build gtest from downloaded sources" OFF)
OPTION(CPP17 "enables C++17" OFF)
OPTION(CPP20 "enables C++20 (experimental)" OFF)
OPTION(XTENSOR_SPARSE_DISABLE_EXCEPTIONS "Disable C++ exceptions" OFF)

if(DOWNLOAD_GTEST OR GTEST_SRC_DIR)
    set(BUILD_TESTS ON)
endif()

if(XTENSOR_ENABLE_ASSERT OR XTENSOR_CHECK_DIMENSION)
    add_definitions(-DXTENSOR_ENABLE_ASSERT)
endif()

if(XTENSOR_CHECK_DIMENSION)
    add_definitions(-DXTENSOR_ENABLE_CHECK_DIMENSION)
endif()

if(BUILD_TESTS)
    add_subdirectory(test)
endif()

# Installation
# ============

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS xtensor-sparse
        EXPORT ${PROJECT_NAME}-targets)

# Makes the project importable from the build directory
export(EXPORT ${PROJECT_NAME}-targets
       FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")

install(FILES ${XTENSOR_SPARSE_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/xtensor-sparse)

set(XTENSOR_SPARSE_CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}" CACHE
    STRING "install path for xtensor-sparseConfig.cmake")

configure_package_config_file(${PROJECT_NAME}Config.cmake.in
                              "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                              INSTALL_DESTINATION ${XTENSOR_SPARSE_CMAKECONFIG_INSTALL_DIR})

# xtensor-sparse is header-only and does not depend on the architecture.
# Remove CMAKE_SIZEOF_VOID_P from xtensor-sparseConfigVersion.cmake so that an
# xtensor-soarseConfig.cmake
# generated for a 64 bit target can be used for 32 bit targets and vice versa.
set(_XTENSOR_SPARSE_CMAKE_SIZEOF_VOID_P ${CMAKE_SIZEOF_VOID_P})
unset(CMAKE_SIZEOF_VOID_P)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
                                 VERSION ${${PROJECT_NAME}_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
set(CMAKE_SIZEOF_VOID_P ${_XTENSOR_SPARSE_CMAKE_SIZEOF_VOID_P})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
              DESTINATION ${XTENSOR_SPARSE_CMAKECONFIG_INSTALL_DIR})
install(EXPORT ${PROJECT_NAME}-targets
        FILE ${PROJECT_NAME}Targets.cmake
        DESTINATION ${XTENSOR_SPARSE_CMAKECONFIG_INSTALL_DIR})


