cmake_minimum_required(VERSION 3.14)
project(quadrotor_mpc LANGUAGES CXX C)

# Default to Release (-O3) — solver needs it to meet 10ms budget at 100 Hz
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Dependencies
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(nlohmann_json 3.2 REQUIRED)

# acados
if(DEFINED ENV{ACADOS_SOURCE_DIR})
    set(ACADOS_DIR $ENV{ACADOS_SOURCE_DIR})
else()
    set(ACADOS_DIR "$ENV{HOME}/acados")
endif()
message(STATUS "ACADOS_DIR = ${ACADOS_DIR}")

set(ACADOS_INCLUDE_DIR ${ACADOS_DIR}/include)
set(ACADOS_LIB_DIR     ${ACADOS_DIR}/lib)

# ROS2 (optional — only needed for SiL targets)
find_package(ament_cmake QUIET)
find_package(ament_index_cpp QUIET)
find_package(rclcpp QUIET)
find_package(nav_msgs QUIET)
find_package(sensor_msgs QUIET)
find_package(std_msgs QUIET)
find_package(std_srvs QUIET)
find_package(geometry_msgs QUIET)
find_package(quadrotor_msgs QUIET)
find_package(mav_manager_srv QUIET)

if(rclcpp_FOUND AND quadrotor_msgs_FOUND)
    set(HAS_ROS2 TRUE)
    message(STATUS "ROS2 found — SiL targets will be built")
else()
    set(HAS_ROS2 FALSE)
    message(STATUS "ROS2 NOT found — only MiL targets will be built")
endif()

# Common library (quaternion, RK4, params)
add_library(quadrotor_common STATIC
    src/common/quaternion_algebra.cpp
    src/common/integrators.cpp
)
target_include_directories(quadrotor_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ACADOS_INCLUDE_DIR}
    ${ACADOS_INCLUDE_DIR}/acados
    ${ACADOS_INCLUDE_DIR}/blasfeo/include
    ${ACADOS_INCLUDE_DIR}/hpipm/include
)
target_link_libraries(quadrotor_common PUBLIC Eigen3::Eigen)

# Trajectory library (Lissajous, arc-length, attitude ref, planner loader)
add_library(quadrotor_trajectory STATIC
    src/trajectory/lissajous.cpp
    src/trajectory/arc_length.cpp
    src/trajectory/attitude_reference.cpp
    src/trajectory/planner_ref.cpp
)
target_link_libraries(quadrotor_trajectory PUBLIC
    quadrotor_common
    nlohmann_json::nlohmann_json
)

# MuJoCo ROS2 interface (shared by all SiL controllers)
if(HAS_ROS2)
    add_library(mujoco_interface STATIC
        src/mujoco/mujoco_interface.cpp
        src/mujoco/sil_protocol.cpp
    )
    target_link_libraries(mujoco_interface PUBLIC
        quadrotor_common
    )
    ament_target_dependencies(mujoco_interface PUBLIC
        rclcpp nav_msgs sensor_msgs std_msgs std_srvs geometry_msgs quadrotor_msgs
    )
endif()

# NMPC solver (generated C code — added after running ocp_generation/)
set(NMPC_CODEGEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/c_generated_code_nmpc)
if(EXISTS ${NMPC_CODEGEN_DIR}/acados_solver_quadrotor_nmpc.c)
    message(STATUS "NMPC generated code found at ${NMPC_CODEGEN_DIR}")

    file(GLOB NMPC_MODEL_SRC   ${NMPC_CODEGEN_DIR}/quadrotor_nmpc_model/*.c)
    file(GLOB NMPC_COST_SRC    ${NMPC_CODEGEN_DIR}/quadrotor_nmpc_cost/*.c)
    file(GLOB NMPC_CONSTR_SRC  ${NMPC_CODEGEN_DIR}/quadrotor_nmpc_constraints/*.c)

    add_library(nmpc_solver STATIC
        ${NMPC_CODEGEN_DIR}/acados_solver_quadrotor_nmpc.c
        ${NMPC_MODEL_SRC}
        ${NMPC_COST_SRC}
        ${NMPC_CONSTR_SRC}
    )
    target_include_directories(nmpc_solver PUBLIC
        ${NMPC_CODEGEN_DIR}
        ${ACADOS_INCLUDE_DIR}
        ${ACADOS_INCLUDE_DIR}/acados
        ${ACADOS_INCLUDE_DIR}/blasfeo/include
        ${ACADOS_INCLUDE_DIR}/hpipm/include
    )
    target_link_libraries(nmpc_solver PUBLIC
        ${ACADOS_LIB_DIR}/libacados.so
        ${ACADOS_LIB_DIR}/libblasfeo.so
        ${ACADOS_LIB_DIR}/libhpipm.so
        m
    )

    # NMPC MiL simulation (standalone, no ROS2)
    add_executable(nmpc_sim
        src/nmpc/nmpc_sim.cpp
        src/nmpc/nmpc_controller.cpp
    )
    target_link_libraries(nmpc_sim PRIVATE
        quadrotor_common
        quadrotor_trajectory
        nmpc_solver
    )

    # NMPC SiL simulation (MuJoCo + ROS2)
    if(HAS_ROS2)
        add_executable(nmpc_sil
            src/nmpc/nmpc_sil.cpp
            src/nmpc/nmpc_controller.cpp
        )
        target_link_libraries(nmpc_sil
            quadrotor_common
            quadrotor_trajectory
            nmpc_solver
            mujoco_interface
        )
        ament_target_dependencies(nmpc_sil
            rclcpp nav_msgs sensor_msgs std_msgs std_srvs geometry_msgs quadrotor_msgs
        )
    endif()

else()
    message(STATUS "NMPC generated code NOT found — run ocp_generation/generate_nmpc_ocp.py first")
endif()

# MPCC solver (generated C code — added after running ocp_generation/)
set(MPCC_CODEGEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/c_generated_code_mpcc)
if(EXISTS ${MPCC_CODEGEN_DIR}/acados_solver_quadrotor_mpcc.c)
    message(STATUS "MPCC generated code found at ${MPCC_CODEGEN_DIR}")

    file(GLOB MPCC_MODEL_SRC   ${MPCC_CODEGEN_DIR}/quadrotor_mpcc_model/*.c)
    file(GLOB MPCC_COST_SRC    ${MPCC_CODEGEN_DIR}/quadrotor_mpcc_cost/*.c)
    file(GLOB MPCC_CONSTR_SRC  ${MPCC_CODEGEN_DIR}/quadrotor_mpcc_constraints/*.c)

    add_library(mpcc_solver STATIC
        ${MPCC_CODEGEN_DIR}/acados_solver_quadrotor_mpcc.c
        ${MPCC_MODEL_SRC}
        ${MPCC_COST_SRC}
        ${MPCC_CONSTR_SRC}
    )
    target_include_directories(mpcc_solver PUBLIC
        ${MPCC_CODEGEN_DIR}
        ${ACADOS_INCLUDE_DIR}
        ${ACADOS_INCLUDE_DIR}/acados
        ${ACADOS_INCLUDE_DIR}/blasfeo/include
        ${ACADOS_INCLUDE_DIR}/hpipm/include
    )
    target_link_libraries(mpcc_solver PUBLIC
        ${ACADOS_LIB_DIR}/libacados.so
        ${ACADOS_LIB_DIR}/libblasfeo.so
        ${ACADOS_LIB_DIR}/libhpipm.so
        m
    )

    # MPCC MiL (no ROS, no MuJoCo — internal RK4 plant)
    add_executable(mpcc_mil
        src/mpcc/mpcc_mil.cpp
        src/mpcc/mpcc_controller.cpp
    )
    target_link_libraries(mpcc_mil
        quadrotor_common
        quadrotor_trajectory
        mpcc_solver
        nlohmann_json::nlohmann_json
    )
    if(HAS_ROS2)
        ament_target_dependencies(mpcc_mil ament_index_cpp)
    endif()
    target_compile_definitions(mpcc_mil PRIVATE
        QUADROTOR_RESULTS_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/results")

    # MPCC SiL simulation (MuJoCo + ROS2)
    if(HAS_ROS2)
        add_executable(mpcc_sil
            src/mpcc/mpcc_sil.cpp
            src/mpcc/mpcc_controller.cpp
        )
        target_link_libraries(mpcc_sil
            quadrotor_common
            quadrotor_trajectory
            mpcc_solver
            mujoco_interface
            nlohmann_json::nlohmann_json
        )
        ament_target_dependencies(mpcc_sil
            rclcpp ament_index_cpp nav_msgs sensor_msgs std_msgs std_srvs geometry_msgs quadrotor_msgs
        )
        target_compile_definitions(mpcc_sil PRIVATE
            QUADROTOR_RESULTS_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/results")

        # MPCC node for the integrated flight stack
        add_executable(mpcc_node
            src/mpcc/mpcc_ros2_node.cpp
            src/mpcc/mpcc_controller.cpp
        )
        target_link_libraries(mpcc_node
            quadrotor_common
            quadrotor_trajectory
            mpcc_solver
            nlohmann_json::nlohmann_json
        )
        ament_target_dependencies(mpcc_node
            rclcpp ament_index_cpp nav_msgs sensor_msgs std_msgs std_srvs geometry_msgs quadrotor_msgs mav_manager_srv
        )
        target_compile_definitions(mpcc_node PRIVATE
            QUADROTOR_RESULTS_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/results")
    endif()

else()
    message(STATUS "MPCC generated code NOT found — run ocp_generation/generate_mpcc_ocp.py first")
endif()

# DQ-MPCC solver (generated C code)
set(DQ_MPCC_CODEGEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/c_generated_code_dq_mpcc)
if(EXISTS ${DQ_MPCC_CODEGEN_DIR}/acados_solver_quadrotor_dq_mpcc.c)
    message(STATUS "DQ-MPCC generated code found at ${DQ_MPCC_CODEGEN_DIR}")

    file(GLOB DQ_MPCC_MODEL_SRC   ${DQ_MPCC_CODEGEN_DIR}/quadrotor_dq_mpcc_model/*.c)
    file(GLOB DQ_MPCC_COST_SRC    ${DQ_MPCC_CODEGEN_DIR}/quadrotor_dq_mpcc_cost/*.c)
    file(GLOB DQ_MPCC_CONSTR_SRC  ${DQ_MPCC_CODEGEN_DIR}/quadrotor_dq_mpcc_constraints/*.c)

    add_library(dq_mpcc_solver STATIC
        ${DQ_MPCC_CODEGEN_DIR}/acados_solver_quadrotor_dq_mpcc.c
        ${DQ_MPCC_MODEL_SRC}
        ${DQ_MPCC_COST_SRC}
        ${DQ_MPCC_CONSTR_SRC}
    )
    target_include_directories(dq_mpcc_solver PUBLIC
        ${DQ_MPCC_CODEGEN_DIR}
        ${ACADOS_INCLUDE_DIR}
        ${ACADOS_INCLUDE_DIR}/acados
        ${ACADOS_INCLUDE_DIR}/blasfeo/include
        ${ACADOS_INCLUDE_DIR}/hpipm/include
    )
    target_link_libraries(dq_mpcc_solver PUBLIC
        ${ACADOS_LIB_DIR}/libacados.so
        ${ACADOS_LIB_DIR}/libblasfeo.so
        ${ACADOS_LIB_DIR}/libhpipm.so
        m
    )

    # DQ-MPCC MiL (no ROS, no MuJoCo — internal RK4 plant)
    add_executable(dq_mpcc_mil
        src/dq_mpcc/dq_mpcc_mil.cpp
        src/dq_mpcc/dq_mpcc_controller.cpp
    )
    target_link_libraries(dq_mpcc_mil
        quadrotor_common
        quadrotor_trajectory
        dq_mpcc_solver
        nlohmann_json::nlohmann_json
    )
    if(HAS_ROS2)
        ament_target_dependencies(dq_mpcc_mil ament_index_cpp)
    endif()
    target_compile_definitions(dq_mpcc_mil PRIVATE
        QUADROTOR_RESULTS_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/results")

    # DQ-MPCC SiL simulation (MuJoCo + ROS2)
    if(HAS_ROS2)
        add_executable(dq_mpcc_sil
            src/dq_mpcc/dq_mpcc_sil.cpp
            src/dq_mpcc/dq_mpcc_controller.cpp
        )
        target_link_libraries(dq_mpcc_sil
            quadrotor_common
            quadrotor_trajectory
            dq_mpcc_solver
            mujoco_interface
            nlohmann_json::nlohmann_json
        )
        ament_target_dependencies(dq_mpcc_sil
            rclcpp ament_index_cpp nav_msgs sensor_msgs std_msgs std_srvs geometry_msgs quadrotor_msgs
        )
        target_compile_definitions(dq_mpcc_sil PRIVATE
            QUADROTOR_RESULTS_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/results")

        # DQ-MPCC (exact J_l^-1) node for the integrated flight stack
        add_executable(dq_mpcc_node
            src/dq_mpcc/dq_mpcc_ros2_node.cpp
            src/dq_mpcc/dq_mpcc_controller.cpp
        )
        target_link_libraries(dq_mpcc_node
            quadrotor_common
            quadrotor_trajectory
            dq_mpcc_solver
            nlohmann_json::nlohmann_json
        )
        ament_target_dependencies(dq_mpcc_node
            rclcpp ament_index_cpp nav_msgs sensor_msgs std_msgs std_srvs geometry_msgs quadrotor_msgs mav_manager_srv
        )
        target_compile_definitions(dq_mpcc_node PRIVATE
            QUADROTOR_RESULTS_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/results")
    endif()

else()
    message(STATUS "DQ-MPCC generated code NOT found — run ocp_generation/generate_dq_mpcc_ocp.py first")
endif()

# DQ-MPCC ABLATION (J_l⁻¹ = I) — separate solver, MiL only
set(DQ_MPCC_ABL_CODEGEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/c_generated_code_dq_mpcc_ablation)
if(EXISTS ${DQ_MPCC_ABL_CODEGEN_DIR}/acados_solver_quadrotor_dq_mpcc_ablation.c)
    message(STATUS "DQ-MPCC ablation generated code found at ${DQ_MPCC_ABL_CODEGEN_DIR}")

    file(GLOB DQ_MPCC_ABL_MODEL_SRC   ${DQ_MPCC_ABL_CODEGEN_DIR}/quadrotor_dq_mpcc_ablation_model/*.c)
    file(GLOB DQ_MPCC_ABL_COST_SRC    ${DQ_MPCC_ABL_CODEGEN_DIR}/quadrotor_dq_mpcc_ablation_cost/*.c)
    file(GLOB DQ_MPCC_ABL_CONSTR_SRC  ${DQ_MPCC_ABL_CODEGEN_DIR}/quadrotor_dq_mpcc_ablation_constraints/*.c)

    add_library(dq_mpcc_ablation_solver STATIC
        ${DQ_MPCC_ABL_CODEGEN_DIR}/acados_solver_quadrotor_dq_mpcc_ablation.c
        ${DQ_MPCC_ABL_MODEL_SRC}
        ${DQ_MPCC_ABL_COST_SRC}
        ${DQ_MPCC_ABL_CONSTR_SRC}
    )
    target_include_directories(dq_mpcc_ablation_solver PUBLIC
        ${DQ_MPCC_ABL_CODEGEN_DIR}
        ${ACADOS_INCLUDE_DIR}
        ${ACADOS_INCLUDE_DIR}/acados
        ${ACADOS_INCLUDE_DIR}/blasfeo/include
        ${ACADOS_INCLUDE_DIR}/hpipm/include
    )
    target_link_libraries(dq_mpcc_ablation_solver PUBLIC
        ${ACADOS_LIB_DIR}/libacados.so
        ${ACADOS_LIB_DIR}/libblasfeo.so
        ${ACADOS_LIB_DIR}/libhpipm.so
        m
    )

    add_executable(dq_mpcc_ablation_mil
        src/dq_mpcc_ablation/dq_mpcc_ablation_mil.cpp
        src/dq_mpcc_ablation/dq_mpcc_ablation_controller.cpp
    )
    target_link_libraries(dq_mpcc_ablation_mil
        quadrotor_common
        quadrotor_trajectory
        dq_mpcc_ablation_solver
        nlohmann_json::nlohmann_json
    )
    if(HAS_ROS2)
        ament_target_dependencies(dq_mpcc_ablation_mil ament_index_cpp)
    endif()
    target_compile_definitions(dq_mpcc_ablation_mil PRIVATE
        QUADROTOR_RESULTS_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/results")

    if(HAS_ROS2)
        add_executable(dq_mpcc_ablation_sil
            src/dq_mpcc_ablation/dq_mpcc_ablation_sil.cpp
            src/dq_mpcc_ablation/dq_mpcc_ablation_controller.cpp
        )
        target_link_libraries(dq_mpcc_ablation_sil
            quadrotor_common
            quadrotor_trajectory
            dq_mpcc_ablation_solver
            mujoco_interface
            nlohmann_json::nlohmann_json
        )
        ament_target_dependencies(dq_mpcc_ablation_sil
            rclcpp ament_index_cpp nav_msgs sensor_msgs std_msgs std_srvs geometry_msgs quadrotor_msgs
        )
        target_compile_definitions(dq_mpcc_ablation_sil PRIVATE
            QUADROTOR_RESULTS_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/results")

        # DQ-MPCC node for the integrated flight stack
        add_executable(dq_mpcc_ablation_node
            src/dq_mpcc_ablation/dq_mpcc_ablation_ros2_node.cpp
            src/dq_mpcc_ablation/dq_mpcc_ablation_controller.cpp
        )
        target_link_libraries(dq_mpcc_ablation_node
            quadrotor_common
            quadrotor_trajectory
            dq_mpcc_ablation_solver
            nlohmann_json::nlohmann_json
        )
        ament_target_dependencies(dq_mpcc_ablation_node
            rclcpp ament_index_cpp nav_msgs sensor_msgs std_msgs std_srvs geometry_msgs quadrotor_msgs mav_manager_srv
        )
        target_compile_definitions(dq_mpcc_ablation_node PRIVATE
            QUADROTOR_RESULTS_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/results")
    endif()
endif()

# Install targets (for colcon)
if(ament_cmake_FOUND)
    set(_install_targets "")
    foreach(_t nmpc_sil mpcc_sil mpcc_mil dq_mpcc_sil dq_mpcc_mil
              dq_mpcc_ablation_sil dq_mpcc_ablation_mil nmpc_sim
              mpcc_node dq_mpcc_node dq_mpcc_ablation_node)
        if(TARGET ${_t})
            list(APPEND _install_targets ${_t})
        endif()
    endforeach()
    if(_install_targets)
        install(TARGETS ${_install_targets}
            DESTINATION lib/${PROJECT_NAME}
        )
    endif()

    # Config YAML (for --params-file)
    install(DIRECTORY config/
        DESTINATION share/${PROJECT_NAME}/config
    )

    # Launch files (SiL and integrated flight stack)
    install(DIRECTORY launch/
        DESTINATION share/${PROJECT_NAME}/launch
    )

    # Controller activation scripts (ros2 run)
    # Renamed to drop .py for clean invocation: ros2 run quadrotor_mpc mpcc_on
    install(PROGRAMS scripts/ops/mpcc_on.py     DESTINATION lib/${PROJECT_NAME} RENAME mpcc_on)
    install(PROGRAMS scripts/ops/mpcc_off.py    DESTINATION lib/${PROJECT_NAME} RENAME mpcc_off)
    install(PROGRAMS scripts/ops/dq_mpcc_on.py  DESTINATION lib/${PROJECT_NAME} RENAME dq_mpcc_on)
    install(PROGRAMS scripts/ops/dq_mpcc_off.py DESTINATION lib/${PROJECT_NAME} RENAME dq_mpcc_off)
    install(PROGRAMS scripts/ops/dq_ablation_on.py  DESTINATION lib/${PROJECT_NAME} RENAME dq_ablation_on)
    install(PROGRAMS scripts/ops/dq_ablation_off.py DESTINATION lib/${PROJECT_NAME} RENAME dq_ablation_off)
    install(PROGRAMS scripts/ops/play_multilap.py   DESTINATION lib/${PROJECT_NAME} RENAME play_multilap)

    # Resources for portable deployment (clone+build+run on any host).
    # Nodes resolve via ament_index_cpp::get_package_share_directory.
    install(DIRECTORY path_planing/
        DESTINATION share/${PROJECT_NAME}/path_planing
        FILES_MATCHING
        PATTERN "*.json"
        PATTERN "*.npy"
        PATTERN "*.npz"
        PATTERN "snapshots" EXCLUDE
        PATTERN "figures"   EXCLUDE
    )
    install(DIRECTORY tuning/
        DESTINATION share/${PROJECT_NAME}/tuning
        FILES_MATCHING
        PATTERN "*.json"
        PATTERN "manual"     EXCLUDE
        PATTERN "*.bak*"     EXCLUDE
        PATTERN "*history*"  EXCLUDE
        PATTERN "*eval_top*" EXCLUDE
    )
    # tuning/sil/manual: installed without FILES_MATCHING so symlink-install applies
    install(DIRECTORY tuning/sil/manual
        DESTINATION share/${PROJECT_NAME}/tuning/sil
    )

    ament_package()
endif()
