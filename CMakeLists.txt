cmake_minimum_required(VERSION 3.28)

set(CMAKE_CPP_COMPILER g++)
# set(CMAKE_C_COMPILER   gcc)
set(CMAKE_CXX_STANDARD 20)

project(cTorch VERSION 0.0.1
               DESCRIPTION "C++/CUDA implimentation of CNN"
               LANGUAGES CXX)

# message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")


# add_library(cxx_cTorch INTERFACE)

# target_include_directories(cxx_cTorch INTERFACE ${PROJECT_SOURCE_DIR})


# target_compile_options(cxx_cTorch INTERFACE
#                         -Wall -Wpedantic -Wextra)

foreach(dir IN ITEMS ${PROJECT_NAME} test)
    add_subdirectory(${CMAKE_SOURCE_DIR}/${dir})
endforeach()