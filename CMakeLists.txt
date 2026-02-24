cmake_minimum_required(VERSION 3.5.2)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(directory_test)
set(CMAKE_BUILD_TYPE Release)

# Use the active Python environment and NumPy headers instead of hardcoded paths.
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

#However, the file(GLOB...) allows for wildcard additions:
file(GLOB SOURCES "*.cpp")

#Generate the shared library from the sources
add_executable(app ${SOURCES})

# Link to final executable
target_include_directories(app PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(app PRIVATE Python3::Python)

