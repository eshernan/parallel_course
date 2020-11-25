////////////////////////////////////////////////////////////////////////////////
//
// (C) Andy Thomason 2016
//
// Example of a thread check.
//
// Mutexes, even if implemented as spin locks, can be very costly.
//
// It is common to use excessive mutexes to solve unexpected multithreaded crashes
// in code.
//
////////////////////////////////////////////////////////////////////////////////

#include <mutex>
#include <stdio.h>

// Very simple example of a class to detect data races.
// This class replaces std::lock_guard and reports if a mutex is needed
namespace par {
  template<class Type>
  class thread_check {
    std::mutex &mtx_;
  public:
    thread_check(std::mutex &mtx) : mtx_(mtx) {
      // try the lock. If the lock is contended, mark required.
      if (!mtx.try_lock()) {
        mutex_required(true);
        mtx.lock();
      }
    }

    ~thread_check() {
      mtx_.unlock();
    }

    static bool mutex_required(bool set=false) {
      static bool value = false;
      bool ret = value;
      if (set) value = true;
      return ret;
    }

    static void report() {
      const char *name = typeid(Type).name();
      if (mutex_required()) {
        printf("%s needs a mutex\n", name);
      } else {
        printf("%s does not need a mutex\n", name);
      }
    }
  };
}

