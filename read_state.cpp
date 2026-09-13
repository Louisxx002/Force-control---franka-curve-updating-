#include <franka/robot.h>
#include <chrono>
#include <iomanip>
#include <iostream>

template <class A> void array(const A& a) {
  std::cout << "[";
  for (size_t i=0;i<a.size();++i) std::cout << (i ? "," : "") << a[i];
  std::cout << "]";
}
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  try {
    franka::Robot robot(argv[1]);
    auto s = robot.readOnce();
    std::cout << std::setprecision(17) << "{\"O_T_EE\":";
    array(s.O_T_EE);
    std::cout << ",\"F_T_EE\":"; array(s.F_T_EE);
    std::cout << ",\"q\":"; array(s.q);
    std::cout << ",\"dq\":"; array(s.dq);
    std::cout << ",\"O_F_ext_hat_K\":"; array(s.O_F_ext_hat_K);
    std::cout << ",\"m_ee\":" << s.m_ee << ",\"m_load\":" << s.m_load;
    std::cout << ",\"robot_mode\":" << static_cast<int>(s.robot_mode);
    std::cout << ",\"robot_time_s\":" << s.time.toSec() << "}\n";
  } catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }
}
