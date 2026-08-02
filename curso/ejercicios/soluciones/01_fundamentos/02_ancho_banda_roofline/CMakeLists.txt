cmake_minimum_required(VERSION 3.20)
project(exercise_01_fundamentos_02_ancho_banda_roofline LANGUAGES C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_library(roofline src/roofline.c)
target_include_directories(roofline PUBLIC include)
if(MSVC)
  target_compile_options(roofline PRIVATE /W4)
else()
  target_compile_options(roofline PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(exercise_02_ancho_banda_roofline tests/test_roofline.c)
target_link_libraries(exercise_02_ancho_banda_roofline PRIVATE roofline)
if(MSVC)
  target_compile_options(exercise_02_ancho_banda_roofline PRIVATE /W4)
else()
  target_compile_options(exercise_02_ancho_banda_roofline PRIVATE -Wall -Wextra -Wpedantic)
  target_link_libraries(exercise_02_ancho_banda_roofline PRIVATE m)
endif()

# Herramienta de medición manual: no participa en ctest (ver rúbrica del
# curso, docs/PLANEACION_CURSO.md) porque el rendimiento solo se evalúa
# después de superar las pruebas de corrección de exercise_02_ancho_banda_roofline.
add_executable(roofline_benchmark src/benchmark.c)
target_link_libraries(roofline_benchmark PRIVATE roofline)
if(MSVC)
  target_compile_options(roofline_benchmark PRIVATE /W4)
else()
  target_compile_options(roofline_benchmark PRIVATE -Wall -Wextra -Wpedantic)
  target_link_libraries(roofline_benchmark PRIVATE m)
endif()

enable_testing()
add_test(NAME ancho_banda_roofline_tests COMMAND exercise_02_ancho_banda_roofline)
