cmake_minimum_required (VERSION 2.6)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CPP_COMPILER g++)

project(parallel_algorithms)
add_executable(for_each for_each.cpp)
add_executable(sort sort.cpp)
add_executable(thread_check thread_check.cpp)

IF(WIN32)
ELSE(WIN32)
  find_package (Threads)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
  target_link_libraries (for_each ${CMAKE_THREAD_LIBS_INIT})
  target_link_libraries (sort ${CMAKE_THREAD_LIBS_INIT})
  target_link_libraries (thread_check ${CMAKE_THREAD_LIBS_INIT})
ENDIF(WIN32)



