cmake_minimum_required(VERSION 3.20)
project(validation_c17_smoke LANGUAGES C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_executable(validation_c17_smoke main.c)
if(MSVC)
  target_compile_options(validation_c17_smoke PRIVATE /W4 /WX)
else()
  target_compile_options(validation_c17_smoke PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

enable_testing()
add_test(NAME validation_c17_smoke_runs COMMAND validation_c17_smoke)
