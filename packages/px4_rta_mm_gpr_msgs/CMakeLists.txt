cmake_minimum_required(VERSION 3.8)
project(px4_rta_mm_gpr_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PlannerStatus.msg"
  "msg/RtaPlan.msg"
  "msg/RtaGains.msg"
  "msg/ControlTick.msg"
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
