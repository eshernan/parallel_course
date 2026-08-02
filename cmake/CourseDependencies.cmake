include_guard(GLOBAL)

include(CheckCSourceCompiles)

if(COURSE_STRICT_VERSIONS)
  if(NOT CMAKE_C_COMPILER_ID STREQUAL "GNU" OR NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(FATAL_ERROR "El curso requiere GCC/G++ ${COURSE_GCC_VERSION}; se detectó ${CMAKE_C_COMPILER_ID}/${CMAKE_CXX_COMPILER_ID}.")
  endif()
  if(NOT CMAKE_C_COMPILER_VERSION VERSION_EQUAL "${COURSE_GCC_VERSION}")
    message(FATAL_ERROR "Se requiere GCC ${COURSE_GCC_VERSION}; se detectó ${CMAKE_C_COMPILER_VERSION}.")
  endif()
  if(NOT CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL "${COURSE_GCC_VERSION}")
    message(FATAL_ERROR "Se requiere G++ ${COURSE_GCC_VERSION}; se detectó ${CMAKE_CXX_COMPILER_VERSION}.")
  endif()
endif()

find_package(Threads REQUIRED)

add_library(course_compile_options INTERFACE)
target_compile_features(course_compile_options INTERFACE c_std_${COURSE_C_STANDARD} cxx_std_${COURSE_CXX_STANDARD})
target_compile_options(course_compile_options INTERFACE
  $<$<COMPILE_LANGUAGE:C,CXX>:-Wall>
  $<$<COMPILE_LANGUAGE:C,CXX>:-Wextra>
  $<$<COMPILE_LANGUAGE:C,CXX>:-Wpedantic>
  $<$<COMPILE_LANGUAGE:C,CXX>:-Wconversion>
  $<$<COMPILE_LANGUAGE:C,CXX>:-Wshadow>
)
target_link_libraries(course_compile_options INTERFACE Threads::Threads)

if(COURSE_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS C CXX)
  set(CMAKE_REQUIRED_LIBRARIES OpenMP::OpenMP_C)
  check_c_source_compiles("#include <omp.h>
    #if !defined(_OPENMP) || _OPENMP < 202111
    #error OpenMP 5.2 required
    #endif
    int main(void) { return omp_get_max_threads() < 1; }" COURSE_HAS_OPENMP_52)
  unset(CMAKE_REQUIRED_LIBRARIES)
  if(NOT COURSE_HAS_OPENMP_52)
    message(FATAL_ERROR "El compilador no anuncia compatibilidad OpenMP 5.2 (_OPENMP >= 202111).")
  endif()
  add_library(course_openmp INTERFACE)
  target_link_libraries(course_openmp INTERFACE OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
endif()

if(COURSE_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS C CXX)
  execute_process(
    COMMAND "${MPIEXEC_EXECUTABLE}" --version
    OUTPUT_VARIABLE course_mpiexec_version
    ERROR_VARIABLE course_mpiexec_version
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(COURSE_STRICT_VERSIONS AND NOT course_mpiexec_version MATCHES "${COURSE_MPI_VERSION}")
    message(FATAL_ERROR "Se requiere ${COURSE_MPI_IMPLEMENTATION} ${COURSE_MPI_VERSION}. Salida de mpiexec: ${course_mpiexec_version}")
  endif()
  set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_C)
  check_c_source_compiles("#include <mpi.h>
    #if MPI_VERSION < ${COURSE_MPI_STANDARD_VERSION}
    #error MPI 5 required
    #endif
    int main(void) { return 0; }" COURSE_HAS_MPI_5)
  unset(CMAKE_REQUIRED_LIBRARIES)
  if(NOT COURSE_HAS_MPI_5)
    message(FATAL_ERROR "La implementación encontrada no anuncia MPI ${COURSE_MPI_STANDARD_VERSION}.x.")
  endif()
  add_library(course_mpi INTERFACE)
  target_link_libraries(course_mpi INTERFACE MPI::MPI_C MPI::MPI_CXX)
endif()

if(COURSE_ENABLE_CUDA)
  find_package(CUDAToolkit ${COURSE_CUDA_VERSION} REQUIRED)
  if(COURSE_STRICT_VERSIONS AND
     (CUDAToolkit_VERSION VERSION_LESS "${COURSE_CUDA_VERSION}" OR
      NOT CUDAToolkit_VERSION VERSION_LESS "13.1"))
    message(FATAL_ERROR "Se requiere CUDA Toolkit ${COURSE_CUDA_VERSION}.x; se detectó ${CUDAToolkit_VERSION}.")
  endif()
  add_library(course_cuda INTERFACE)
  target_link_libraries(course_cuda INTERFACE CUDA::cudart)
endif()

if(COURSE_ENABLE_NOTEBOOKS)
  find_package(Python3 ${COURSE_PYTHON_VERSION} EXACT REQUIRED COMPONENTS Interpreter)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/config/validate_notebook_stack.py"
    RESULT_VARIABLE course_notebook_stack_status
    OUTPUT_VARIABLE course_notebook_stack_output
    ERROR_VARIABLE course_notebook_stack_output
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT course_notebook_stack_status EQUAL 0)
    message(FATAL_ERROR "Entorno de notebooks inválido: ${course_notebook_stack_output}")
  endif()
  message(STATUS "${course_notebook_stack_output}")
endif()
