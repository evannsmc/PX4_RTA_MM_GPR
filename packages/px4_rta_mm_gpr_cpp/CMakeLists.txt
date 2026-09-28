cmake_minimum_required(VERSION 3.8)
project(px4_rta_mm_gpr_cpp)

set(CMAKE_CXX_STANDARD 20)  # std::atomic<std::shared_ptr<T>>
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(mocap_msgs REQUIRED)
find_package(px4_rta_mm_gpr_msgs REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

add_executable(rta_fast_loop src/rta_fast_loop.cpp)
target_include_directories(rta_fast_loop PRIVATE include)
target_link_libraries(rta_fast_loop Eigen3::Eigen)
ament_target_dependencies(rta_fast_loop rclcpp px4_msgs mocap_msgs px4_rta_mm_gpr_msgs)

# ROS-free equivalence checker for the control law (driven by test/test_control_law.py)
add_executable(control_law_check src/control_law_check.cpp)
target_include_directories(control_law_check PRIVATE include)
target_link_libraries(control_law_check Eigen3::Eigen)

install(TARGETS rta_fast_loop control_law_check DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME} OPTIONAL)

ament_package()
