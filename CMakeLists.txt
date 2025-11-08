cmake_minimum_required(VERSION 3.16)
project(bgremover-lite LANGUAGES CXX)

# ==== C++ standard ====
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==== Find OpenCV with CUDA support ====
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui dnn)
message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

# ==== Detect CUDA Automatically ====
find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND)
    set(CUDA_AVAILABLE TRUE)
    message(STATUS "✅ CUDA detected: ${CUDAToolkit_VERSION}")
    
    # Check for CUDA runtime library
    find_library(CUDA_RUNTIME_LIBRARY
        NAMES cudart
        HINTS ${CUDAToolkit_LIBRARY_DIRS} ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    )
    if(CUDA_RUNTIME_LIBRARY)
        message(STATUS "✅ CUDA runtime library found: ${CUDA_RUNTIME_LIBRARY}")
    else()
        message(WARNING "⚠️ CUDA runtime library not found")
        set(CUDA_AVAILABLE FALSE)
    endif()
else()
    set(CUDA_AVAILABLE FALSE)
    message(WARNING "⚠️ CUDA not detected — building CPU-only version")
endif()

# ==== ONNX Runtime download ====
set(ONNXRUNTIME_VERSION "1.19.0")
set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/onnxruntime")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib")

# Only download if header doesn’t exist
if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
  file(MAKE_DIRECTORY ${ONNXRUNTIME_DIR})
  message(STATUS "💾 Downloading ONNX Runtime ${ONNXRUNTIME_VERSION}...")

  if(CUDA_AVAILABLE)
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz")
  else()
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz")
  endif()

  set(ONNXRUNTIME_ARCHIVE "${CMAKE_BINARY_DIR}/onnxruntime.tgz")
  file(DOWNLOAD ${ONNXRUNTIME_URL} ${ONNXRUNTIME_ARCHIVE} SHOW_PROGRESS TIMEOUT 300 STATUS DOWNLOAD_STATUS)

  if(NOT DOWNLOAD_STATUS EQUAL 0)
    message(FATAL_ERROR "❌ Failed to download ONNX Runtime from ${ONNXRUNTIME_URL}")
  endif()

  file(ARCHIVE_EXTRACT INPUT ${ONNXRUNTIME_ARCHIVE} DESTINATION ${CMAKE_BINARY_DIR})

  # Determine extracted folder name dynamically
  file(GLOB EXTRACTED_DIRS "${CMAKE_BINARY_DIR}/onnxruntime-linux-*")
  list(GET EXTRACTED_DIRS 0 EXTRACTED_DIR)

  file(COPY "${EXTRACTED_DIR}/" DESTINATION "${ONNXRUNTIME_DIR}/")
  file(REMOVE ${ONNXRUNTIME_ARCHIVE})
  file(REMOVE_RECURSE ${EXTRACTED_DIR})

  message(STATUS "✅ ONNX Runtime downloaded to: ${ONNXRUNTIME_DIR}")
endif()

# ==== Verify ONNX Runtime ====
if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "❌ ONNX Runtime headers not found after download.")
endif()

# ==== Create imported ONNX Runtime target ====
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
)

# ==== Executables ====
function(add_bgremover_target TARGET SRC_FILES)
    add_executable(${TARGET} ${SRC_FILES})
    target_include_directories(${TARGET} PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${ONNXRUNTIME_INCLUDE_DIR}
    )
    target_link_libraries(${TARGET} PRIVATE
        ${OpenCV_LIBS}
        onnxruntime
    )
    target_compile_options(${TARGET} PRIVATE -O3 -ffast-math -march=native)
endfunction()

# CPU version
add_bgremover_target(bgremover ${CMAKE_SOURCE_DIR}/main.cpp)

# GPU version — only if CUDA available
if(CUDA_AVAILABLE)
    add_bgremover_target(bgremover_gpu ${CMAKE_SOURCE_DIR}/main_gpu.cpp)
    target_compile_definitions(bgremover_gpu PRIVATE USE_CUDA=1)
    target_link_libraries(bgremover_gpu PRIVATE 
        ${CUDA_RUNTIME_LIBRARY}
        cudadevrt  # CUDA device runtime
    )
    # Enable CUDA compilation for this target
    enable_language(CUDA)
    set_target_properties(bgremover_gpu PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
    # Add CUDA include directories
    target_include_directories(bgremover_gpu PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    message(STATUS "🚀 Building GPU-accelerated version with CUDA and ONNX Runtime GPU backend")
else()
    message(STATUS "🧩 Building CPU-only executables")
endif()

# Test utilities
add_bgremover_target(test_model ${CMAKE_SOURCE_DIR}/test_model.cpp)
add_bgremover_target(debug_colors ${CMAKE_SOURCE_DIR}/debug_colors.cpp)
add_bgremover_target(test_performance ${CMAKE_SOURCE_DIR}/test_performance.cpp)

# GPU test utilities
if(CUDA_AVAILABLE)
    add_bgremover_target(test_gpu ${CMAKE_SOURCE_DIR}/test_gpu.cpp)
    target_link_libraries(test_gpu PRIVATE 
        ${CUDA_RUNTIME_LIBRARY}
        cudadevrt
    )
    target_include_directories(test_gpu PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    enable_language(CUDA)
    set_target_properties(test_gpu PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
endif()

# ==== Optional RPATH setup (so ONNX Runtime lib is found at runtime) ====
set_target_properties(bgremover bgremover_gpu test_model debug_colors test_performance
    PROPERTIES BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}"
)
