cmake_minimum_required(VERSION 3.20)
project(validation_cxx20_smoke LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(validation_cxx20_smoke main.cpp)
if(MSVC)
  target_compile_options(validation_cxx20_smoke PRIVATE /W4 /WX)
else()
  target_compile_options(validation_cxx20_smoke PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

enable_testing()
add_test(NAME validation_cxx20_smoke_runs COMMAND validation_cxx20_smoke)
