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

#include <atomic>
#include <array>
#include <future>

namespace par {
  template <class Fn, class FwdIt, int ChunkSize=0x1000, int NumCPUs=8>
  void for_each(FwdIt b, FwdIt e, Fn func) {
    size_t size = e - b;
    std::atomic<size_t> start{0};
    std::array<std::future<void>, NumCPUs> futures;
    for (auto &f : futures) {
      f = std::async(std::launch::async, [b, size, &start, &func]() {
        for (;;) {
          size_t idx = start.fetch_add(ChunkSize);
          if (idx >= size) break;
          auto cb = b + idx;
          auto ce = b + std::min(idx + ChunkSize, size);
          for (auto p = cb; p != ce; ++p) func(p[0]);
        }
      });
    }
    for (auto &f : futures) f.wait();
  }
}

