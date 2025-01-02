# Build on Windows

set(TORCH_XPU_OPS_LIBRARIES)
set(SYCL_LINK_LIBRARIES_KEYWORD PRIVATE)


if(BUILD_SEPARATE_OPS)

  add_library(
    torch_xpu_ops
    STATIC
    ${ATen_XPU_CPP_SRCS})
  set(PATH_TO_TORCH_XPU_OPS_ATEN_LIB \"torch_xpu_ops_aten.dll\")
  target_compile_options(torch_xpu_ops PRIVATE -DPATH_TO_TORCH_XPU_OPS_ATEN_LIB=${PATH_TO_TORCH_XPU_OPS_ATEN_LIB})

  add_library(
    torch_xpu_ops_aten
    SHARED
    ${ATen_XPU_NATIVE_CPP_SRCS}
    ${ATen_XPU_GEN_SRCS})
  install(TARGETS torch_xpu_ops_aten DESTINATION "${TORCH_INSTALL_LIB_DIR}")
  target_compile_definitions(torch_xpu_ops_aten PRIVATE TORCH_XPU_BUILD_MAIN_LIB)
  target_link_libraries(torch_xpu_ops_aten PUBLIC torch_xpu)
  target_link_libraries(torch_xpu_ops_aten PUBLIC torch_cpu)
  target_link_libraries(torch_xpu_ops_aten PUBLIC c10)

  foreach(sycl_src ${ATen_XPU_SYCL_SRCS})
    get_filename_component(name ${sycl_src} NAME_WLE REALPATH)
    set(sycl_lib torch-xpu-ops-sycl-${name})
    sycl_add_library(
      ${sycl_lib}
      SHARED
      SYCL_SOURCES ${sycl_src})
    target_link_libraries(torch_xpu_ops_aten PUBLIC ${sycl_lib})
    list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_lib})

    # Decouple with PyTorch cmake definition.
    install(TARGETS ${sycl_lib} DESTINATION "${TORCH_INSTALL_LIB_DIR}")
  endforeach()
  list(APPEND TORCH_XPU_OPS_LIBRARIES torch_xpu_ops)
  list(APPEND TORCH_XPU_OPS_LIBRARIES torch_xpu_ops_aten)
elseif(BUILD_SPLIT_KERNEL_LIB)

  add_library(
    torch_xpu_ops
    STATIC
    ${ATen_XPU_CPP_SRCS})
  set(PATH_TO_TORCH_XPU_OPS_ATEN_LIB \"torch_xpu_ops_aten.dll\")
  target_compile_options(torch_xpu_ops PRIVATE -DPATH_TO_TORCH_XPU_OPS_ATEN_LIB=${PATH_TO_TORCH_XPU_OPS_ATEN_LIB})

  add_library(
    torch_xpu_ops_aten
    SHARED
    ${ATen_XPU_NATIVE_CPP_SRCS}
    ${ATen_XPU_GEN_SRCS})
  install(TARGETS torch_xpu_ops_aten DESTINATION "${TORCH_INSTALL_LIB_DIR}")
  target_compile_definitions(torch_xpu_ops_aten PRIVATE TORCH_XPU_BUILD_MAIN_LIB)
  target_link_libraries(torch_xpu_ops_aten PUBLIC torch_xpu)
  target_link_libraries(torch_xpu_ops_aten PUBLIC torch_cpu)
  target_link_libraries(torch_xpu_ops_aten PUBLIC c10)

  # Split SYCL kernels into 4 libraries as categories 1) Unary+Binary 2) Reduce 3) Foreach 4) Others.
  set(ATen_XPU_SYCL_UNARY_BINARY_SRCS)
  set(ATen_XPU_SYCL_REDUCE_SRCS)
  set(ATen_XPU_SYCL_FOREACH_SRCS)
  set(ATen_XPU_SYCL_OTHERS_SRCS)
  foreach(sycl_src ${ATen_XPU_SYCL_SRCS})
    string(REGEX MATCH "Binary" IS_BINARY ${sycl_src})
    string(REGEX MATCH "Unary" IS_UNARY ${sycl_src})
    string(REGEX MATCH "Pow" IS_POW ${sycl_src})
    string(REGEX MATCH "Copy" IS_COPY ${sycl_src})
    string(REGEX MATCH "Reduce" IS_REDUCE ${sycl_src})
    string(REGEX MATCH "Activation" IS_ACTIVATION ${sycl_src})
    string(REGEX MATCH "Foreach" IS_FOREACH ${sycl_src})

    if(NOT IS_FOREACH STREQUAL "")
      list(APPEND ATen_XPU_SYCL_FOREACH_SRCS ${sycl_src})
    elseif(NOT IS_REDUCE STREQUAL "")
      list(APPEND ATen_XPU_SYCL_REDUCE_SRCS ${sycl_src})
    elseif(NOT IS_UNARY STREQUAL "" OR NOT IS_BINARY STREQUAL "")
      list(APPEND ATen_XPU_SYCL_UNARY_BINARY_SRCS ${sycl_src})
    elseif(NOT IS_COPY STREQUAL "" OR NOT IS_POW STREQUAL "")
      list(APPEND ATen_XPU_SYCL_UNARY_BINARY_SRCS ${sycl_src})
    elseif(NOT IS_ACTIVATION STREQUAL "")
      list(APPEND ATen_XPU_SYCL_UNARY_BINARY_SRCS ${sycl_src})
    else()
      list(APPEND ATen_XPU_SYCL_OTHERS_SRCS ${sycl_src})
    endif()
  endforeach()
  # Unary binary kernel lib
  set(sycl_unary_binary_lib torch_xpu_ops_sycl_unary_binary_kernels)
  sycl_add_library(
    ${sycl_unary_binary_lib}
    SHARED
    SYCL_SOURCES ${ATen_XPU_SYCL_UNARY_BINARY_SRCS})
  target_compile_definitions(${sycl_unary_binary_lib} PRIVATE TORCH_XPU_BUILD_MAIN_LIB)
  target_link_libraries(torch_xpu_ops_aten PUBLIC ${sycl_unary_binary_lib})
  target_link_libraries(${sycl_unary_binary_lib} PUBLIC torch_xpu)
  list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_unary_binary_lib})

  # Decouple with PyTorch cmake definition.
  install(TARGETS ${sycl_unary_binary_lib} DESTINATION "${TORCH_INSTALL_LIB_DIR}")


  # Reduce kernel lib
  set(sycl_reduce_lib torch_xpu_ops_sycl_reduce_kernels)
  sycl_add_library(
    ${sycl_reduce_lib}
    SHARED
    SYCL_SOURCES ${ATen_XPU_SYCL_REDUCE_SRCS})
  target_compile_definitions(${sycl_reduce_lib} PRIVATE TORCH_XPU_BUILD_MAIN_LIB)
  target_link_libraries(torch_xpu_ops_aten PUBLIC ${sycl_reduce_lib})
  target_link_libraries(${sycl_reduce_lib} PUBLIC torch_xpu)
  list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_reduce_lib})

  # Decouple with PyTorch cmake definition.
  install(TARGETS ${sycl_reduce_lib} DESTINATION "${TORCH_INSTALL_LIB_DIR}")


  # Foreach kernel lib
  set(sycl_foreach_lib torch_xpu_ops_sycl_foreach_kernels)
  sycl_add_library(
    ${sycl_foreach_lib}
    SHARED
    SYCL_SOURCES ${ATen_XPU_SYCL_FOREACH_SRCS})
  target_compile_definitions(${sycl_foreach_lib} PRIVATE TORCH_XPU_BUILD_MAIN_LIB)
  target_link_libraries(torch_xpu_ops_aten PUBLIC ${sycl_foreach_lib})
  target_link_libraries(${sycl_foreach_lib} PUBLIC torch_xpu)
  list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_foreach_lib})

  # Decouple with PyTorch cmake definition.
  install(TARGETS ${sycl_foreach_lib} DESTINATION "${TORCH_INSTALL_LIB_DIR}")

  
  # Other kernel lib
  set(sycl_lib torch_xpu_ops_sycl_kernels)
  sycl_add_library(
    ${sycl_lib}
    SHARED
    SYCL_SOURCES ${ATen_XPU_SYCL_OTHERS_SRCS})
  target_compile_definitions(${sycl_lib} PRIVATE TORCH_XPU_BUILD_MAIN_LIB)
  target_link_libraries(torch_xpu_ops_aten PUBLIC ${sycl_lib})
  target_link_libraries(${sycl_lib} PUBLIC torch_xpu)
  list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_lib})

  # Decouple with PyTorch cmake definition.
  install(TARGETS ${sycl_lib} DESTINATION "${TORCH_INSTALL_LIB_DIR}")

  list(APPEND TORCH_XPU_OPS_LIBRARIES torch_xpu_ops)
  list(APPEND TORCH_XPU_OPS_LIBRARIES torch_xpu_ops_aten)
else()
  # Internal file name is decided by the target name. On windows, torch_xpu_ops_sycl_kernels
  # is too long in device code linkage command.
  sycl_add_library(
    xpu-sycl
    STATIC
    CXX_SOURCES  ${ATen_XPU_CPP_SRCS} ${ATen_XPU_NATIVE_CPP_SRCS} ${ATen_XPU_GEN_SRCS}
    SYCL_SOURCES ${ATen_XPU_SYCL_SRCS})
  add_library(torch_xpu_ops ALIAS xpu-sycl)
  set_target_properties(xpu-sycl PROPERTIES OUTPUT_NAME torch_xpu_ops)
  set(SYCL_TARGET xpu-sycl)

  install(TARGETS xpu-sycl DESTINATION "${TORCH_INSTALL_LIB_DIR}")
  list(APPEND TORCH_XPU_OPS_LIBRARIES xpu-sycl)
endif()
set(SYCL_LINK_LIBRARIES_KEYWORD)

foreach(lib ${TORCH_XPU_OPS_LIBRARIES})
  # Align with PyTorch compile options PYTORCH_SRC_DIR/cmake/public/utils.cmake
  torch_compile_options(${lib})
  target_compile_options_if_supported(${lib} "-Wno-deprecated-copy")
  target_compile_options(${lib} PRIVATE ${TORCH_XPU_OPS_FLAGS})

  target_include_directories(${lib} PUBLIC ${TORCH_XPU_OPS_INCLUDE_DIRS})
  target_include_directories(${lib} PUBLIC ${ATen_XPU_INCLUDE_DIRS})
  target_include_directories(${lib} PUBLIC ${SYCL_INCLUDE_DIR})

  target_link_libraries(${lib} PUBLIC ${SYCL_LIBRARY})
endforeach()
