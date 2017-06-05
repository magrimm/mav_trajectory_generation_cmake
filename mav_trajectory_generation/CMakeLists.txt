cmake_minimum_required(VERSION 3.1)

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------

# ----------------------------------------
# Eigen
# ----------------------------------------
find_package(Eigen3 QUIET)
include_directories(${EIGEN3_INCLUDE_DIR})

# ----------------------------------------
# Google Log
# ----------------------------------------
find_package(Glog REQUIRED QUIET)
include_directories(${GLOG_INCLUDE_DIRS})

# ----------------------------------------
# Nlopt
# ----------------------------------------
find_package(NLOPT REQUIRED QUIET)
include_directories(${NLOPT_INCLUDE_DIRS})

# ------------------------------------------------------------------------------
# Internal includes
# ------------------------------------------------------------------------------
include_directories(include)

# ------------------------------------------------------------------------------
# Library sources
# ------------------------------------------------------------------------------
set(SOURCES
  src/motion_defines.cpp
  src/polynomial.cpp
  src/rpoly.cpp
  src/segment.cpp
  src/timing.cpp
  src/trajectory.cpp
  src/vertex.cpp
)

# ------------------------------------------------------------------------------
# Library building
# ------------------------------------------------------------------------------
add_library(mav_trajectory_generation SHARED ${SOURCES})
target_link_libraries(mav_trajectory_generation ${GLOG_LIBRARIES})
target_link_libraries(mav_trajectory_generation ${NLOPT_LIBRARIES})

# ------------------------------------------------------------------------------
# Build Tests
# ------------------------------------------------------------------------------
#if(BUILD_TESTS)
#  add_subdirectory(tests)
#endif()

# --------------------------------------------------------------------------------
# Install
# --------------------------------------------------------------------------------
generate_export_header(mav_trajectory_generation)
set_property(TARGET mav_trajectory_generation PROPERTY VERSION ${${PROJECT_NAME}_VERSION})
set_property(TARGET mav_trajectory_generation PROPERTY SOVERSION 1)
set_property(TARGET mav_trajectory_generation PROPERTY INTERFACE_mav_trajectory_generation_MAJOR_VERSION 1)
set_property(TARGET mav_trajectory_generation APPEND PROPERTY COMPATIBLE_INTERFACE_STRING mav_trajectory_generation_MAJOR_VERSION)

install(TARGETS mav_trajectory_generation EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include)
install(
  FILES polynomial.h "${CMAKE_CURRENT_BINARY_DIR}/polynomial_export.h"
  DESTINATION include
  COMPONENT mav_trajectory_generation)
