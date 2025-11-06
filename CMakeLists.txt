cmake_minimum_required(VERSION 3.16)
project(bgremover-lite LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)

# Set CUDA availability to FALSE for now
set(CUDA_AVAILABLE FALSE)
message(STATUS "CUDA support disabled - building CPU-only version")

# Download ONNX Runtime if not already present
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/onnxruntime/include/onnxruntime_cxx_api.h")
  message(STATUS "Downloading ONNX Runtime...")
  
  # Detect platform
  if(WIN32)
    message(FATAL_ERROR "ONNX Runtime download is currently only supported on Linux x64. 
Please install ONNX Runtime manually for Windows or modify this script.")
  elseif(APPLE)
    message(FATAL_ERROR "ONNX Runtime download is currently only supported on Linux x64. 
Please install ONNX Runtime manually for macOS or modify this script.")
  endif()
  
  # For GPU support, try to download CUDA-enabled version
  set(ONNXRUNTIME_VERSION "1.19.0")
  
  if(CUDA_AVAILABLE)
    # Try CUDA-enabled version first
    set(ONNXRUNTIME_CUDA_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz")
    set(ONNXRUNTIME_CUDA_ARCHIVE "${CMAKE_BINARY_DIR}/onnxruntime_cuda.tgz")
    
    message(STATUS "Attempting to download CUDA-enabled ONNX Runtime...")
    file(DOWNLOAD ${ONNXRUNTIME_CUDA_URL} ${ONNXRUNTIME_CUDA_ARCHIVE} 
         SHOW_PROGRESS TIMEOUT 300 STATUS CUDA_DOWNLOAD_STATUS)
    
    if(CUDA_DOWNLOAD_STATUS EQUAL 0)
      message(STATUS "CUDA-enabled ONNX Runtime downloaded successfully!")
      set(ONNXRUNTIME_ARCHIVE ${ONNXRUNTIME_CUDA_ARCHIVE})
      set(EXTRACTED_DIR "onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}")
    else()
      message(STATUS "CUDA-enabled ONNX Runtime download failed, falling back to CPU version")
      set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz")
      set(ONNXRUNTIME_ARCHIVE "${CMAKE_BINARY_DIR}/onnxruntime.tgz")
      file(DOWNLOAD ${ONNXRUNTIME_URL} ${ONNXRUNTIME_ARCHIVE} 
           SHOW_PROGRESS TIMEOUT 300)
      set(EXTRACTED_DIR "onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}")
    endif()
  else()
    # CPU-only version
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz")
    set(ONNXRUNTIME_ARCHIVE "${CMAKE_BINARY_DIR}/onnxruntime.tgz")
    file(DOWNLOAD ${ONNXRUNTIME_URL} ${ONNXRUNTIME_ARCHIVE} 
         SHOW_PROGRESS TIMEOUT 300)
    set(EXTRACTED_DIR "onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}")
  endif()
  
  set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/onnxruntime")
  
  # Create the target directory
  file(MAKE_DIRECTORY ${ONNXRUNTIME_DIR})
  
  # Extract the archive using CMake's built-in capabilities
  message(STATUS "Extracting ONNX Runtime archive...")
  file(ARCHIVE_EXTRACT INPUT ${ONNXRUNTIME_ARCHIVE}
                     DESTINATION ${CMAKE_BINARY_DIR})
  
  # Copy the extracted directory to the source directory
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy_directory 
            "${CMAKE_BINARY_DIR}/${EXTRACTED_DIR}" 
            "${ONNXRUNTIME_DIR}"
  )
  
  # Clean up
  file(REMOVE ${ONNXRUNTIME_ARCHIVE})
  file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/${EXTRACTED_DIR}")
  
  message(STATUS "ONNX Runtime downloaded and extracted to ${ONNXRUNTIME_DIR}")
endif()

# Set ONNX Runtime paths
set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/onnxruntime")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib")

message(STATUS "ONNX Runtime dir: ${ONNXRUNTIME_DIR}")
message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime lib: ${ONNXRUNTIME_LIB_DIR}")

# Check if the header file exists
if(EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
  message(STATUS "ONNX Runtime headers found")
else()
  message(FATAL_ERROR "ONNX Runtime headers not found in ${ONNXRUNTIME_INCLUDE_DIR}")
endif()

# Set the library name (just the name, not the full path, for direct linking)
set(ONNXRUNTIME_LIB onnxruntime)

# Include directories
include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE_DIR}
)

# Link directories
link_directories(${ONNXRUNTIME_LIB_DIR})

# Add the main CPU executable
add_executable(bgremover ${CMAKE_SOURCE_DIR}/main.cpp)
target_link_libraries(bgremover ${OpenCV_LIBS} ${ONNXRUNTIME_LIB})

# Add GPU executable (ONNX GPU acceleration) 
add_executable(bgremover_gpu ${CMAKE_SOURCE_DIR}/main_gpu.cpp)
target_link_libraries(bgremover_gpu ${OpenCV_LIBS} ${ONNXRUNTIME_LIB})

# Set optimization flags
if(CUDA_AVAILABLE)
  message(STATUS "Building GPU-accelerated version (ONNX Runtime CUDA)")
  target_compile_options(bgremover_gpu PRIVATE -O3 -ffast-math -march=native)
else()
  message(STATUS "CUDA not available - building CPU optimized version")
  target_compile_options(bgremover_gpu PRIVATE -O3 -ffast-math)
endif()

# Add a test executable for model verification
add_executable(test_model ${CMAKE_SOURCE_DIR}/test_model.cpp)
target_link_libraries(test_model ${ONNXRUNTIME_LIB})

# Add a color debug test
add_executable(debug_colors ${CMAKE_SOURCE_DIR}/debug_colors.cpp)
target_link_libraries(debug_colors ${OpenCV_LIBS})

# Add a performance test
add_executable(test_performance ${CMAKE_SOURCE_DIR}/test_performance.cpp)
target_link_libraries(test_performance ${OpenCV_LIBS})

# Add GPU performance test (commented out due to linking issues)
# add_executable(test_gpu ${CMAKE_SOURCE_DIR}/test_gpu.cpp)
# target_link_libraries(test_gpu ${OpenCV_LIBS})