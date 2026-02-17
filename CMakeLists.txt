cmake_minimum_required(VERSION 3.18)

project(cburgers_sciml LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

# Discover Python (interpreter, dev headers/libs, and NumPy) from the active environment.
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

message(STATUS "Python3 executable: ${Python3_EXECUTABLE}")
message(STATUS "Python3 include dirs: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3 NumPy include dirs: ${Python3_NumPy_INCLUDE_DIRS}")

add_executable(app app.cpp)

target_include_directories(app PRIVATE
  ${Python3_INCLUDE_DIRS}
  ${Python3_NumPy_INCLUDE_DIRS}
)

target_link_libraries(app PRIVATE Python3::Python)

# Keep runtime Python modules next to the executable so embedded Python can import them.
set(PYTHON_RUNTIME_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/build/python_module.py
  ${CMAKE_CURRENT_SOURCE_DIR}/build/ml_module.py
)

foreach(py_file IN LISTS PYTHON_RUNTIME_FILES)
  if(EXISTS "${py_file}")
    get_filename_component(py_name "${py_file}" NAME)
    configure_file("${py_file}" "${CMAKE_CURRENT_BINARY_DIR}/${py_name}" COPYONLY)
  endif()
endforeach()
