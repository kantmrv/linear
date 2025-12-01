cmake_minimum_required(VERSION 3.18)
project(linear)

# C++20 required for the library
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Python
find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

# Fetch nanobind if not already present
include(FetchContent)
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG v2.0.0
)
FetchContent_MakeAvailable(nanobind)

# Include paths for C++ headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)

# Create the Python extension module
nanobind_add_module(
  _linear_impl              # Internal C++ module name
  STABLE_ABI                # Use Python stable ABI for compatibility
  NB_STATIC                 # Static linkage for nanobind
  bindings.cpp              # Source file
)

# Install the module
install(TARGETS _linear_impl LIBRARY DESTINATION linear)
