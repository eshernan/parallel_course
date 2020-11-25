////////////////////////////////////////////////////////////////////////////////
//
// (C) Andy Thomason 2016
//
// Example of a low-overhead thread check.
//
// Mutexes, even if implemented as spin locks, can be very costly.
//
// It is common to use excessive mutexes to solve unexpected multithreaded crashes
// in code.
//
// Another method is to replace selected mutexes with a thread safety check.
// This will report if the objects protected really are accessed by separate threads.
//
////////////////////////////////////////////////////////////////////////////////

#include "thread_check.hpp"
#include <mutex>
#include <future>
#include <vector>

class A {
  std::vector<int> ints_;
  std::mutex mtx_;
public:
  void push(int a) {
    // replace this with std::lock_guard in release builds
    par::thread_check<A> lock(mtx_);
    ints_.push_back(a);
  }
};

class B {
  std::vector<float> floats_;
  std::mutex mtx_;
public:
  void push(float a) {
    // replace this with std::lock_guard in release builds
    par::thread_check<B> lock(mtx_);
    floats_.push_back(a);
  }
};

int main() {
  // A is used by both Asyncs and so needs the mutex.
  A a;
  auto fn = [&a]() {
    // B is local to the Async and so does not need the mutex.
    B b;
    for (int i = 0; i != 10000; ++i) {
      a.push(1);
      b.push(1.0f);
    }
  };

  // Run both asyncs.
  auto t1 = std::async(std::launch::async, fn);
  auto t2 = std::async(std::launch::async, fn);

  t1.wait(); t2.wait();

  // Report results.
  par::thread_check<A>::report();
  par::thread_check<B>::report();
}

