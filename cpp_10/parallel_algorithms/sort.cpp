////////////////////////////////////////////////////////////////////////////////
//
// (C) Andy Thomason 2016
// Simple example of a parallel sort
//
// Note that C++17 and SYCL will bring std::sort with parallel semantics soon.
//
// How it works:
//
// We use std::partition to swap elements in the array into chunks.
// We recursively partition these chunks until they are small enough to be sorted by std::sort
//   on many threads.
// Finally, we run NumCPUs asyncs to sort the individual chunks.
//
// This may be improved by using a bucket sort (provided the data is random)
//   and a second partition for equal elements (provided the data has repeats).
//
// Ask yourself, does my data need to be fully sorted?
//
////////////////////////////////////////////////////////////////////////////////

#include "sort.hpp"
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>

int main() {
  std::ranlux24 gen;
  std::vector<int> x(0x100000);

  printf("par::sort\n");
  for (int i = 0; i != 5; ++i) {
    gen.seed();
    std::generate(x.begin(), x.end(), [&gen]() { return (int)gen(); });
    auto start = std::chrono::high_resolution_clock::now();
    par::sort(x.begin(), x.end(), [](int a, int b) { return a < b; });
    auto end = std::chrono::high_resolution_clock::now();
    printf("%10dns\n", (int)std::chrono::nanoseconds(end - start).count());
  }

  printf("std::sort\n");
  for (int i = 0; i != 5; ++i) {
    gen.seed();
    std::generate(x.begin(), x.end(), [&gen]() { return (int)gen(); });
    auto start = std::chrono::high_resolution_clock::now();
    std::sort(x.begin(), x.end(), [](int a, int b) { return a < b; });
    auto end = std::chrono::high_resolution_clock::now();
    printf("%10dns\n", (int)std::chrono::nanoseconds(end - start).count());
  }
}

