#include <array>
#include <iostream>
#include <numeric>

int main() {
  constexpr std::array<int, 4> values{2, 0, 2, 6};
  const auto sum = std::reduce(values.begin(), values.end());
  std::cout << "Programacion Paralela 2026: C++20 disponible.\n";
  return sum == 10 ? 0 : 1;
}
