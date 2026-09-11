cmake_minimum_required(VERSION 3.23)

project(kvcache-manager VERSION 0.1.0 LANGUAGES CXX)

option(KVCACHE_MANAGER_BUILD_DEVICE_TESTS "Build tests that require Tenstorrent devices" OFF)
option(KVCACHE_MANAGER_BUILD_PYTHON_BINDINGS "Build the manager-owned Python bindings" OFF)

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

find_package(TT-NN CONFIG REQUIRED)

add_library(kvcache_manager_cpp SHARED)
add_library(kvcache_manager::cpp ALIAS kvcache_manager_cpp)

set_target_properties(kvcache_manager_cpp PROPERTIES EXPORT_NAME cpp OUTPUT_NAME kvcache_manager)
target_compile_features(kvcache_manager_cpp PUBLIC cxx_std_20)
target_compile_definitions(kvcache_manager_cpp PRIVATE KVCACHE_MANAGER_BUILDING_LIBRARY)
set_target_properties(
    kvcache_manager_cpp
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
)
target_sources(
    kvcache_manager_cpp
    PRIVATE
        src/client.cpp
        src/client_impl.hpp
        src/native/ttnn/bootstrap_contract.hpp
        src/native/ttnn/client_connection.cpp
        src/native/ttnn/client_lifecycle.cpp
        src/native/ttnn/client_operations_adapter.cpp
        src/native/ttnn/shared_memory_mapping.cpp
        src/native/ttnn/server_lifecycle.cpp
        src/native/ttnn/server_tiering.cpp
        src/native/ttnn/server_runtime/server.cpp
        src/native/ttnn/server_runtime/server_device_runtime.cpp
        src/native/ttnn/server_runtime/server_device_runtime_io.cpp
        src/native/ttnn/server_runtime/server_host_api.cpp
        src/native/ttnn/server_runtime/server_program.cpp
        src/native/ttnn/server_runtime/server_resources.cpp
        src/native/ttnn/server_runtime/server_storage.cpp
        src/native/ttnn/server_runtime/server_tiering.cpp
        src/server.cpp
        src/backend_bridge.hpp
        src/server_impl.hpp
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS include
        FILES
            include/kvcache_manager/client.hpp
            include/kvcache_manager/export.hpp
            include/kvcache_manager/profiles.hpp
            include/kvcache_manager/server.hpp
)
target_include_directories(kvcache_manager_cpp PRIVATE src src/native/ttnn)
target_link_libraries(kvcache_manager_cpp PUBLIC TTNN::TTNN)

if(KVCACHE_MANAGER_BUILD_PYTHON_BINDINGS)
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(nanobind CONFIG QUIET)
    if(NOT nanobind_FOUND)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
            RESULT_VARIABLE KVCACHE_MANAGER_NANOBIND_RESULT
            OUTPUT_VARIABLE KVCACHE_MANAGER_NANOBIND_CMAKE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(KVCACHE_MANAGER_NANOBIND_RESULT EQUAL 0)
            find_package(nanobind CONFIG REQUIRED PATHS "${KVCACHE_MANAGER_NANOBIND_CMAKE_DIR}")
        else()
            message(
                FATAL_ERROR
                "nanobind 2.12.0 is required; install the Python package or set nanobind_DIR"
            )
        endif()
    endif()

    set(KVCACHE_MANAGER_PYTHON_PACKAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}/python/kvcache_manager")
    file(MAKE_DIRECTORY "${KVCACHE_MANAGER_PYTHON_PACKAGE_DIR}")
    foreach(PYTHON_SOURCE IN ITEMS __init__.py _facade.py py.typed)
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/src/kvcache_manager/${PYTHON_SOURCE}"
            "${KVCACHE_MANAGER_PYTHON_PACKAGE_DIR}/${PYTHON_SOURCE}"
            COPYONLY
        )
    endforeach()

    nanobind_add_module(kvcache_manager_python NB_STATIC src/python/bindings.cpp)
    set_target_properties(
        kvcache_manager_python
        PROPERTIES
            OUTPUT_NAME _native
            LIBRARY_OUTPUT_DIRECTORY "${KVCACHE_MANAGER_PYTHON_PACKAGE_DIR}"
            INSTALL_RPATH "\$ORIGIN/.libs"
    )
    target_compile_features(kvcache_manager_python PRIVATE cxx_std_20)
    target_link_libraries(kvcache_manager_python PRIVATE kvcache_manager::cpp)

    install(
        TARGETS kvcache_manager_python
        LIBRARY DESTINATION kvcache_manager
        COMPONENT python
    )
    install(
        TARGETS kvcache_manager_cpp
        LIBRARY DESTINATION kvcache_manager/.libs
        COMPONENT python
    )
    install(
        FILES
            src/kvcache_manager/__init__.py
            src/kvcache_manager/_facade.py
            src/kvcache_manager/py.typed
        DESTINATION kvcache_manager
        COMPONENT python
    )
endif()
install(
    TARGETS kvcache_manager_cpp
    EXPORT kvcache-manager-targets
    FILE_SET api DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(
    EXPORT kvcache-manager-targets
    FILE kvcache-manager-targets.cmake
    NAMESPACE kvcache_manager::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kvcache-manager
)

configure_package_config_file(
    cmake/kvcache-manager-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/kvcache-manager-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kvcache-manager
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/kvcache-manager-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/kvcache-manager-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/kvcache-manager-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kvcache-manager
)

if(BUILD_TESTING)
    add_executable(
        kvcache_manager_cpp_facade_test
        src/client.cpp
        src/server.cpp
        src/native/ttnn/server_lifecycle.cpp
        src/native/ttnn/server_tiering.cpp
        tests/cpp/fake_client_impl.cpp
        tests/cpp/fake_backend.cpp
        tests/cpp/test_facade.cpp
    )
    target_compile_features(kvcache_manager_cpp_facade_test PRIVATE cxx_std_20)
    target_compile_definitions(kvcache_manager_cpp_facade_test PRIVATE KVCACHE_MANAGER_BUILDING_LIBRARY)
    target_include_directories(
        kvcache_manager_cpp_facade_test
        PRIVATE
            include
            src
            src/native/ttnn
            tests/cpp
    )
    target_link_libraries(kvcache_manager_cpp_facade_test PRIVATE TTNN::TTNN)
    add_test(NAME kvcache_manager_cpp_facade COMMAND kvcache_manager_cpp_facade_test)

    if(KVCACHE_MANAGER_BUILD_PYTHON_BINDINGS)
        add_test(
            NAME kvcache_manager_python_native_import
            COMMAND
                "${CMAKE_COMMAND}" -E env
                "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:$ENV{PYTHONPATH}"
                "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_import_smoke.py"
        )
    endif()

    if(KVCACHE_MANAGER_BUILD_DEVICE_TESTS)
        add_executable(kvcache_manager_cpp_device_ping_smoke tests/cpp/device_ping_smoke.cpp)
        target_compile_features(kvcache_manager_cpp_device_ping_smoke PRIVATE cxx_std_20)
        target_link_libraries(kvcache_manager_cpp_device_ping_smoke PRIVATE kvcache_manager::cpp)
        add_test(NAME kvcache_manager_cpp_device_ping COMMAND kvcache_manager_cpp_device_ping_smoke)
        set_tests_properties(
            kvcache_manager_cpp_device_ping
            PROPERTIES
                LABELS device
                ENVIRONMENT TT_METAL_INSPECTOR=0
        )

        add_executable(
            kvcache_manager_cpp_static_intermesh_smoke
            tests/cpp/static_intermesh_smoke.cpp
        )
        target_compile_features(kvcache_manager_cpp_static_intermesh_smoke PRIVATE cxx_std_20)
        target_link_libraries(
            kvcache_manager_cpp_static_intermesh_smoke
            PRIVATE
                kvcache_manager::cpp
        )
    endif()
endif()
