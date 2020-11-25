////////////////////////////////////////////////////////////////////////////////
//
// (C) Andy Thomason 2016
//
// Simple example of a for_each iterator
//
// Note that C++17 and SYCL will bring std::for_each with parallel semantics soon.
//
// How it works:
//
// We create NumCPUs asyncs (one per hardware thread typically) and use an atomic
//   variable to count blocks of ChunkSize elements which we iterate sequentially.
//
// The iterator must be random-accessible as we need to jump forward.
//
////////////////////////////////////////////////////////////////////////////////

#include "for_each.hpp"
#include <vector>
#include <algorithm>
#include <chrono>

int main() {
  std::vector<int> x(0x100000);

  printf("par::for_each\n");
  for (int i = 0; i != 5; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    par::for_each(x.begin(), x.end(), [](int &r) { r = 0; });
    auto end = std::chrono::high_resolution_clock::now();
    printf("%10dns\n", (int)std::chrono::nanoseconds(end - start).count());
  }

  printf("\nstd::for_each\n");
  for (int i = 0; i != 5; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    std::for_each(x.begin(), x.end(), [](int &r) { r = 0; });
    auto end = std::chrono::high_resolution_clock::now();
    printf("%10dns\n", (int)std::chrono::nanoseconds(end - start).count());
  }
}


