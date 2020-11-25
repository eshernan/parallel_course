

#include <chrono>
#include <atomic>
#include <future>
#include <array>
#include <mutex>


class nolock {
public:
  void lock() {
  }
  void unlock() {
  }
};

class spinlock {
  std::atomic_flag flag_{false};
public:
  void lock() {
    while (flag_.test_and_set(std::memory_order_acquire)) ;
  }
  void unlock() {
    flag_.clear(std::memory_order_release);
  }
};

template <class LockType>
class locker {
  LockType &lock_;
public:
  locker(LockType &lock) : lock_(lock) { lock_.lock(); }
  ~locker() { lock_.unlock(); }
};

auto now() { return std::chrono::high_resolution_clock::now(); }

// crude growable array
// see http://www.stroustrup.com/lock-free-vector.pdf for an elegant growable array.
template< typename Type, typename SizeType, typename LockType >
class vector {
  static const int capacity_ = 0x100000;
  Type *values_ = nullptr;
  SizeType size_{0};
  LockType lock_;
public:
  vector() : values_(new Type[capacity_]) {}
  ~vector() { delete[] values_; }

  void push_back(const Type &x) {
    locker<LockType> lck(lock_);
    int dest = size_++;
    if (dest < capacity_) values_[dest] = x;
  }

  const Type &operator[](int idx) { return values_[idx]; }

  int size() const { return size_; };
};

template <class Fn>
void time(Fn fn, const char *text) {
  auto start = now();

  {
    std::array<std::future<void>, 8 > futures;
    for ( auto &f: futures ) f = std::async(std::launch::async, fn);
  }

  int ns = (int)std::chrono::nanoseconds(now() - start).count();
  printf("%20s: t=%dns\n", text, ns);
}

int main() {
  vector<int, std::atomic<int>, nolock> vec1;
  time([&vec1] {
    for (int i = 0; i != 0x100000 / 8; ++i) {
      vec1.push_back(i);
    }
  }, "atomic");
  printf("atomic version: size=%x\n", vec1.size());

  vector<int, int, spinlock> vec2;
  time([&vec2] {
    for (int i = 0; i != 0x100000 / 8; ++i) {
      vec2.push_back(i);
    }
  }, "spinlock");
  printf("spinlock version: size=%x\n", vec2.size());

  vector<int, volatile int, nolock> vec3;
  time([&vec3] {
    for (int i = 0; i != 0x100000 / 8; ++i) {
      vec3.push_back(i);
    }
  }, "single thread");
  printf("single thread version: size=%x\n", vec2.size());
}
