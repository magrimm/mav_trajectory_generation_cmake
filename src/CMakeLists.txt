cmake_minimum_required(VERSION 3.1)

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------

# ----------------------------------------
# Google Log
# ----------------------------------------
find_package(Glog REQUIRED QUIET)
include_directories(${GLOG_INCLUDE_DIRS})

# ------------------------------------------------------------------------------
# Internal includes
# ------------------------------------------------------------------------------
include_directories(.)

# ------------------------------------------------------------------------------
# Executable sources
# ------------------------------------------------------------------------------
set(SOURCES
  polynomial_timing_evaluation.cpp
)

# ------------------------------------------------------------------------------
# Executable building
# ------------------------------------------------------------------------------
add_executable(${PROJECT_NAME}_bin ${SOURCES})
target_link_libraries(${PROJECT_NAME}_bin ${GLOG_LIBRARIES})
target_link_libraries(${PROJECT_NAME}_bin mav_trajectory_generation)

# --------------------------------------------------------------------------------
# Install
# --------------------------------------------------------------------------------
install(TARGETS ${PROJECT_NAME}_bin
RUNTIME DESTINATION bin
LIBRARY DESTINATION lib
ARCHIVE DESTINATION lib
)
