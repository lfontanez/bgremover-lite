cmake_minimum_required(VERSION 3.16)

# Suppress FindCUDA deprecation warnings
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 NEW)
endif()

project(bgremover-lite LANGUAGES CXX)

# ==================================================================================
# U²-Net Model Download Configuration
# ==================================================================================

# Include the comprehensive model download module
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ModelDownload)

# Model download options
option(U2NET_DOWNLOAD_MODELS "Download U²-Net models during build" ON)
option(U2NET_OFFLINE_MODE "Use only cached models (no downloads)" OFF)
option(U2NET_CLEAN_CACHE "Clean model cache before downloading" OFF)
set(U2NET_MODEL_CACHE_DIR "" CACHE PATH "Directory to cache downloaded models")

# Set default cache directory if not specified
if(NOT U2NET_MODEL_CACHE_DIR)
    if(DEFINED ENV{XDG_CACHE_HOME})
        set(U2NET_MODEL_CACHE_DIR "$ENV{XDG_CACHE_HOME}/u2net" CACHE PATH "Directory to cache downloaded models" FORCE)
    else()
        set(U2NET_MODEL_CACHE_DIR "$ENV{HOME}/.cache/u2net" CACHE PATH "Directory to cache downloaded models" FORCE)
    endif()
endif()

message(STATUS "📦 Model cache directory: ${U2NET_MODEL_CACHE_DIR}")

if(U2NET_DOWNLOAD_MODELS)
    message(STATUS "📦 U²-Net model download enabled")
    
    if(U2NET_OFFLINE_MODE)
        message(STATUS "🔒 Offline mode: Using cached models only")
    endif()
    
    # Clean cache if requested
    if(U2NET_CLEAN_CACHE)
        u2net_clean_cache()
        message(STATUS "🧹 Model cache cleaned")
    endif()
    
    # Download required models
    message(STATUS "📥 Downloading U²-Net models...")
    u2net_download_all_models()
    
    # Verify model integrity with enhanced reporting
    message(STATUS "=== Model Integrity Verification ===")
    set(MODELS_OK TRUE)
    set(MODEL_VERIFICATION_RESULTS "")
    foreach(model_name u2net u2netp)
        u2net_verify_model_integrity(${model_name} verified)
        if(NOT verified)
            set(MODELS_OK FALSE)
            set(MODEL_VERIFICATION_RESULTS "${MODEL_VERIFICATION_RESULTS}❌ ${model_name}: FAILED\n")
            message(WARNING "❌ Model ${model_name} verification failed")
        else()
            set(MODEL_VERIFICATION_RESULTS "${MODEL_VERIFICATION_RESULTS}✅ ${model_name}: VERIFIED\n")
            message(STATUS "✅ Model ${model_name} verified")
        endif()
    endforeach()
    
    # Enhanced error reporting with fallback to warnings
    if(NOT MODELS_OK)
        message(WARNING "Model verification results:\n${MODEL_VERIFICATION_RESULTS}")
        
        # Check which specific models failed
        string(FIND "${MODEL_VERIFICATION_RESULTS}" "u2net.onnx: FAILED" u2net_failed_pos)
        string(FIND "${MODEL_VERIFICATION_RESULTS}" "u2netp.onnx: FAILED" u2netp_failed_pos)
        
        set(critical_failure FALSE)
        set(optional_failure FALSE)
        
        if(u2net_failed_pos GREATER -1)
            set(critical_failure TRUE)
        endif()
        if(u2netp_failed_pos GREATER -1)
            set(optional_failure TRUE)
        endif()
        
        if(critical_failure)
            if(U2NET_OFFLINE_MODE)
                message(FATAL_ERROR "❌ Critical model integrity verification failed in offline mode. The main u2net.onnx model is required. Check your cached models or disable offline mode.")
            else()
                message(FATAL_ERROR "❌ Critical model integrity verification failed. The main u2net.onnx model is required. Possible issues:\n1. Network connection problems\n2. Corrupted download - try U2NET_CLEAN_CACHE=ON\n3. Model source unavailable - check internet connection")
            endif()
        elseif(optional_failure)
            message(WARNING "⚠️ Optional model verification failed. Build will continue but some features may be limited.")
            message(STATUS "💡 Tip: Try re-running with U2NET_CLEAN_CACHE=ON to redownload models")
        else()
            message(WARNING "⚠️ Model verification failed but continuing build. Runtime may have issues.")
        endif()
    endif()
    
    # Set model paths for use in executables with enhanced feedback
    u2net_get_model_path("u2net" U2NET_MODEL_PATH)
    if(U2NET_MODEL_PATH)
        add_definitions(-DU2NET_MODEL_PATH="${U2NET_MODEL_PATH}")
        message(STATUS "✅ U²-Net model path: ${U2NET_MODEL_PATH}")
    else()
        message(WARNING "⚠️  U²-Net model path not found after download")
    endif()
    
    u2net_get_model_path("u2netp" U2NETP_MODEL_PATH)
    if(U2NETP_MODEL_PATH)
        add_definitions(-DU2NETP_MODEL_PATH="${U2NETP_MODEL_PATH}")
        message(STATUS "✅ U²-Net Portrait model path: ${U2NETP_MODEL_PATH}")
    else()
        message(WARNING "⚠️  U²-Net Portrait model path not found after download")
    endif()
    
    # Final model availability check
    u2net_models_available(models_available)
    if(models_available)
        message(STATUS "🎉 All required models are available and verified!")
    else()
        message(WARNING "⚠️  Some models are still missing after download process")
    endif()
    
else()
    message(STATUS "📦 U²-Net model download disabled")
    message(STATUS "   Manual model placement required:")
    message(STATUS "   Expected locations:")
    message(STATUS "   - models/u2net.onnx")
    message(STATUS "   - models/u2netp.onnx")
    
    # Check for existing manual model placement with improved symlink handling
    set(MANUAL_MODELS_FOUND TRUE)
    
    # Enhanced helper function to check file existence with proper symlink handling
    macro(CHECK_MODEL_EXISTS model_path result_var)
        set(${result_var} FALSE)
        
        # First check if the path exists (handles both files and symlinks)
        if(EXISTS "${model_path}")
            set(${result_var} TRUE)
            get_filename_component(model_name "${model_path}" NAME)
            message(STATUS "✅ ${model_name}: Path exists")
            
            # Check if it's a symlink and get the real path
            if(IS_SYMLINK "${model_path}")
                file(REAL_PATH "${model_path}" real_path)
                message(STATUS "   📋 ${model_name}: Symlink detected")
                message(STATUS "   🔗 Symlink: ${model_path}")
                message(STATUS "   🎯 Target: ${real_path}")
                
                # Verify the target file actually exists and is readable
                if(EXISTS "${real_path}")
                    file(SIZE "${real_path}" file_size)
                    if(file_size GREATER 0)
                        message(STATUS "   ✅ ${model_name}: Valid symlink (${file_size} bytes)")
                    else()
                        message(WARNING "   ⚠️  ${model_name}: Symlink target is empty")
                        set(${result_var} FALSE)
                    endif()
                else()
                    message(WARNING "   ❌ ${model_name}: Symlink target does not exist: ${real_path}")
                    set(${result_var} FALSE)
                endif()
            else()
                # Regular file
                file(SIZE "${model_path}" file_size)
                if(file_size GREATER 0)
                    message(STATUS "   📁 ${model_name}: Regular file (${file_size} bytes)")
                else()
                    message(WARNING "   ⚠️  ${model_name}: File exists but is empty")
                    set(${result_var} FALSE)
                endif()
            endif()
        else()
            get_filename_component(model_name "${model_path}" NAME)
            message(WARNING "⚠️  Manual model placement: ${model_name} not found at ${model_path}")
            
            # Try to find the file in the current directory as fallback
            get_filename_component(parent_dir "${model_path}" PATH)
            get_filename_component(file_name "${model_path}" NAME)
            string(REPLACE "${CMAKE_SOURCE_DIR}/" "" relative_path "${model_path}")
            if(EXISTS "${CMAKE_SOURCE_DIR}/${file_name}")
                message(STATUS "   🔍 Found fallback: ${CMAKE_SOURCE_DIR}/${file_name}")
                set(${result_var} TRUE)
                file(REAL_PATH "${CMAKE_SOURCE_DIR}/${file_name}" real_path)
                message(STATUS "   📋 Using fallback: ${file_name} -> ${real_path}")
            endif()
        endif()
    endmacro()
    
    CHECK_MODEL_EXISTS("${CMAKE_SOURCE_DIR}/models/u2net.onnx" U2NET_EXISTS)
    if(NOT U2NET_EXISTS)
        set(MANUAL_MODELS_FOUND FALSE)
    endif()
    
    CHECK_MODEL_EXISTS("${CMAKE_SOURCE_DIR}/models/u2netp.onnx" U2NETP_EXISTS)
    if(NOT U2NETP_EXISTS)
        message(WARNING "⚠️  Manual model placement: u2netp.onnx not found (optional)")
    endif()
    
    if(MANUAL_MODELS_FOUND)
        message(STATUS "✅ Manual model placement detected")
    endif()
    
    # Enhanced macro to set model paths with comprehensive validation
    macro(SET_MODEL_PATH model_path definition_name)
        set(validated_path "")
        set(validation_status "NOT_FOUND")
        
        # Check if the path exists
        if(EXISTS "${model_path}")
            get_filename_component(model_name "${model_path}" NAME)
            
            # Handle symlinks
            if(IS_SYMLINK "${model_path}")
                file(REAL_PATH "${model_path}" real_path)
                message(STATUS "🔗 Processing symlink: ${model_name}")
                message(STATUS "   Link: ${model_path}")
                message(STATUS "   Target: ${real_path}")
                
                # Validate the target file
                if(EXISTS "${real_path}")
                    file(SIZE "${real_path}" target_size)
                    if(target_size GREATER 0)
                        # Read a small portion to verify the file is readable
                        file(READ "${real_path}" test_content LENGTH_MAX 1024)
                        if(test_content)
                            set(validated_path "${model_path}")
                            set(validation_status "VALID_SYMLINK")
                            message(STATUS "✅ ${model_name}: Valid symlink (${target_size} bytes)")
                        else()
                            message(WARNING "❌ ${model_name}: Symlink target is not readable")
                        endif()
                    else()
                        message(WARNING "❌ ${model_name}: Symlink target is empty (${target_size} bytes)")
                    endif()
                else()
                    message(WARNING "❌ ${model_name}: Symlink target does not exist: ${real_path}")
                endif()
            else()
                # Regular file validation
                file(SIZE "${model_path}" file_size)
                if(file_size GREATER 0)
                    file(READ "${model_path}" test_content LENGTH_MAX 1024)
                    if(test_content)
                        set(validated_path "${model_path}")
                        set(validation_status "VALID_FILE")
                        message(STATUS "✅ ${model_name}: Valid file (${file_size} bytes)")
                    else()
                        message(WARNING "❌ ${model_name}: File exists but is not readable")
                    endif()
                else()
                    message(WARNING "❌ ${model_name}: File exists but is empty (${file_size} bytes)")
                endif()
            endif()
        else()
            get_filename_component(model_name "${model_path}" NAME)
            message(WARNING "⚠️  ${model_name}: Path not found: ${model_path}")
            
            # Try fallback locations
            get_filename_component(file_name "${model_path}" NAME)
            if(EXISTS "${CMAKE_SOURCE_DIR}/${file_name}")
                message(STATUS "🔄 Using fallback: ${CMAKE_SOURCE_DIR}/${file_name}")
                file(REAL_PATH "${CMAKE_SOURCE_DIR}/${file_name}" real_path)
                file(SIZE "${real_path}" fallback_size)
                if(fallback_size GREATER 0)
                    set(validated_path "${CMAKE_SOURCE_DIR}/${file_name}")
                    set(validation_status "FALLBACK")
                    message(STATUS "✅ ${model_name}: Using fallback file (${fallback_size} bytes)")
                endif()
            endif()
        endif()
        
        # Set the definition if we have a validated path
        if(validated_path)
            add_definitions(-D${definition_name}="${validated_path}")
            
            # Log the final result
            if(validation_status STREQUAL "VALID_SYMLINK")
                file(REAL_PATH "${validated_path}" real_target)
                message(STATUS "🎯 ${definition_name} set to symlink: ${validated_path} -> ${real_target}")
            elseif(validation_status STREQUAL "FALLBACK")
                message(STATUS "🎯 ${definition_name} set to fallback: ${validated_path}")
            else()
                message(STATUS "🎯 ${definition_name} set to: ${validated_path}")
            endif()
        else()
            message(WARNING "⚠️  ${definition_name}: No valid model file found - build may fail at runtime")
        endif()
        
        # Clean up variables
        unset(validated_path)
        unset(validation_status)
        unset(test_content)
    endmacro()
    
    # Process model paths with enhanced error handling
    message(STATUS "=== Manual Model Path Resolution ===")
    SET_MODEL_PATH("${CMAKE_SOURCE_DIR}/models/u2net.onnx" "U2NET_MODEL_PATH")
    SET_MODEL_PATH("${CMAKE_SOURCE_DIR}/models/u2netp.onnx" "U2NETP_MODEL_PATH")
    message(STATUS "=== End Manual Model Path Resolution ===")
endif()

# Create models directory if it doesn't exist with enhanced error handling
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/models")
    file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/models")
    if(EXISTS "${CMAKE_SOURCE_DIR}/models")
        message(STATUS "📁 Created models directory: ${CMAKE_SOURCE_DIR}/models")
    else()
        message(FATAL_ERROR "❌ Failed to create models directory: ${CMAKE_SOURCE_DIR}/models")
        message(STATUS "   Please ensure you have write permissions in the project directory")
    endif()
else()
    # Directory exists, check if it's writable
    file(WRITE "${CMAKE_SOURCE_DIR}/models/.cmake_test_write" "test")
    if(EXISTS "${CMAKE_SOURCE_DIR}/models/.cmake_test_write")
        file(REMOVE "${CMAKE_SOURCE_DIR}/models/.cmake_test_write")
        message(STATUS "📁 Models directory exists and is writable: ${CMAKE_SOURCE_DIR}/models")
    else()
        message(WARNING "⚠️  Models directory exists but may not be writable: ${CMAKE_SOURCE_DIR}/models")
    endif()
endif()

# Enhanced model information display
message(STATUS "=== Model Configuration Summary ===")
if(U2NET_DOWNLOAD_MODELS)
    message(STATUS "📦 Model Download: ENABLED")
    message(STATUS "📂 Cache Directory: ${U2NET_MODEL_CACHE_DIR}")
    message(STATUS "🔒 Offline Mode: ${U2NET_OFFLINE_MODE}")
    message(STATUS "🧹 Clean Cache: ${U2NET_CLEAN_CACHE}")
    
    # Get all model paths using the comprehensive function
    u2net_get_all_model_paths(model_paths)
    if(model_paths)
        message(STATUS "📋 Model Paths:")
        foreach(model_path ${model_paths})
            # Parse model_path format: "name=path"
            string(REPLACE "=" " " model_parts ${model_path})
            list(GET model_parts 0 model_name)
            list(GET model_parts 1 path)
            message(STATUS "   ✅ ${model_name}: ${path}")
        endforeach()
    endif()
    
    # Check availability with detailed reporting
    u2net_models_available(models_available)
    if(models_available)
        message(STATUS "🎯 Model Status: ALL AVAILABLE")
    else()
        message(STATUS "⚠️  Model Status: SOME MISSING")
    endif()
else()
    message(STATUS "📦 Model Download: DISABLED (Manual placement)")
    message(STATUS "📂 Expected Directory: ${CMAKE_SOURCE_DIR}/models")
    
    # Check manual placement with detailed status
    set(MANUAL_CHECK_RESULTS "")
    # Enhanced model status checking with comprehensive symlink handling
    set(MANUAL_CHECK_RESULTS "")
    
    macro(ADD_MODEL_STATUS model_path model_name is_required)
        set(found_status "NOT_FOUND")
        set(status_icon "❌")
        set(details "")
        
        # Check if path exists
        if(EXISTS "${model_path}")
            if(IS_SYMLINK "${model_path}")
                file(REAL_PATH "${model_path}" real_path)
                if(EXISTS "${real_path}")
                    file(SIZE "${real_path}" file_size)
                    if(file_size GREATER 0)
                        set(found_status "VALID_SYMLINK")
                        set(status_icon "✅")
                        set(details " (symlink to: ${real_path}, ${file_size} bytes)")
                    else()
                        set(found_status "BROKEN_SYMLINK")
                        set(status_icon "⚠️")
                        set(details " (broken symlink: ${real_path} is empty)")
                    endif()
                else()
                    set(found_status "BROKEN_SYMLINK")
                    set(status_icon "❌")
                    set(details " (broken symlink: ${real_path} does not exist)")
                endif()
            else()
                # Regular file
                file(SIZE "${model_path}" file_size)
                if(file_size GREATER 0)
                    set(found_status "VALID_FILE")
                    set(status_icon "✅")
                    set(details " (direct file, ${file_size} bytes)")
                else()
                    set(found_status "EMPTY_FILE")
                    set(status_icon "⚠️")
                    set(details " (empty file, ${file_size} bytes)")
                endif()
            endif()
        else()
            # Try fallback locations
            get_filename_component(file_name "${model_path}" NAME)
            if(EXISTS "${CMAKE_SOURCE_DIR}/${file_name}")
                file(SIZE "${CMAKE_SOURCE_DIR}/${file_name}" fallback_size)
                if(fallback_size GREATER 0)
                    set(found_status "FALLBACK_FILE")
                    set(status_icon "🔄")
                    set(details " (found in fallback: ${CMAKE_SOURCE_DIR}/${file_name}, ${fallback_size} bytes)")
                else()
                    set(found_status "FALLBACK_EMPTY")
                    set(status_icon "⚠️")
                    set(details " (fallback exists but is empty: ${CMAKE_SOURCE_DIR}/${file_name})")
                endif()
            else()
                if(is_required)
                    set(found_status "MISSING_REQUIRED")
                    set(status_icon "❌")
                else()
                    set(found_status "MISSING_OPTIONAL")
                    set(status_icon "⚠️")
                endif()
            endif()
        endif()
        
        # Add to results
        set(MANUAL_CHECK_RESULTS "${MANUAL_CHECK_RESULTS}${status_icon} ${model_name}: ${details}\n")
        
        # Log detailed status for debugging
        message(DEBUG "Model ${model_name}: status=${found_status}, path=${model_path}, required=${is_required}")
    endmacro()
    
    ADD_MODEL_STATUS("${CMAKE_SOURCE_DIR}/models/u2net.onnx" "u2net.onnx" TRUE)
    ADD_MODEL_STATUS("${CMAKE_SOURCE_DIR}/models/u2netp.onnx" "u2netp.onnx" FALSE)
    
    message(STATUS "Manual Model Status:\n${MANUAL_CHECK_RESULTS}")
endif()
message(STATUS "====================================")

# Show model information in verbose builds
if(CMAKE_VERBOSE_MAKEFILE)
    u2net_show_model_info()
endif()

# ==================================================================================
# Build Configuration
# ==================================================================================

# ==== C++ standard ====
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==================================================================================
# Model Configuration Summary
# ==================================================================================

if(U2NET_DOWNLOAD_MODELS)
    message(STATUS "=== U²-Net Model Download Summary ===")
    message(STATUS "Cache directory: ${U2NET_MODEL_CACHE_DIR}")
    message(STATUS "Models directory: ${CMAKE_SOURCE_DIR}/models")
    
    # Get all model paths using the new function
    u2net_get_all_model_paths(model_paths)
    foreach(model_path ${model_paths})
        string(REPLACE "=" " " model_parts ${model_path})
        list(GET model_parts 0 model_name)
        list(GET model_parts 1 path)
        message(STATUS "✅ ${model_name}: ${path}")
    endforeach()
    
    # Check availability
    u2net_models_available(models_available)
    if(models_available)
        message(STATUS "✅ All models available for building")
    else()
        message(WARNING "⚠️  Some models are missing - build may fail")
    endif()
    
    message(STATUS "=====================================")
else()
    message(STATUS "=== Manual Model Configuration ===")
    message(STATUS "Models directory: ${CMAKE_SOURCE_DIR}/models")
    
    if(EXISTS "${CMAKE_SOURCE_DIR}/models/u2net.onnx")
        message(STATUS "✅ u2net.onnx: Found")
    else()
        message(WARNING "⚠️  u2net.onnx: Missing")
    endif()
    
    if(EXISTS "${CMAKE_SOURCE_DIR}/models/u2netp.onnx")
        message(STATUS "✅ u2netp.onnx: Found")
    else()
        message(WARNING "⚠️  u2netp.onnx: Missing")
    endif()
endif()

# ==== Find OpenCV with CUDA support ====
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui dnn)
message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

# ==== Virtual Camera Support ====
# V4L2 (Video4Linux2) is part of the Linux kernel headers
# The v4l2_output.hpp header is header-only and provides virtual camera support
# No additional libraries or includes needed - V4L2 is built into the kernel
message(STATUS "✅ Virtual Camera support available on Linux (V4L2)")

# ==== Find GTK3 Development Libraries ====
find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)
if(GTK3_FOUND)
    message(STATUS "✅ GTK3 found: ${GTK3_INCLUDE_DIRS}")
    message(STATUS "GTK3 libraries: ${GTK3_LIBRARIES}")
    message(STATUS "GTK3 linker flags: ${GTK3_LDFLAGS}")
else()
    message(WARNING "⚠️ GTK3 not found - OpenCV GUI features may not work properly")
endif()

# Find additional GTK3 dependencies that might be needed
pkg_check_modules(GLIB2 REQUIRED glib-2.0)
pkg_check_modules(GIO REQUIRED gio-2.0)
pkg_check_modules(GOBJECT REQUIRED gobject-2.0)
pkg_check_modules(CAIRO REQUIRED cairo)
if(GLIB2_FOUND)
    message(STATUS "✅ GLib2 found: ${GLIB2_INCLUDE_DIRS}")
endif()
if(GIO_FOUND)
    message(STATUS "✅ GIO found: ${GIO_INCLUDE_DIRS}")
endif()
if(GOBJECT_FOUND)
    message(STATUS "✅ GObject found: ${GOBJECT_INCLUDE_DIRS}")
endif()
if(CAIRO_FOUND)
    message(STATUS "✅ Cairo found: ${CAIRO_INCLUDE_DIRS}")
endif()

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

  # Parse download status
  list(GET DOWNLOAD_STATUS 0 DOWNLOAD_STATUS_CODE)
  list(GET DOWNLOAD_STATUS 1 DOWNLOAD_STATUS_STRING)
  
  if(NOT DOWNLOAD_STATUS_CODE EQUAL 0)
    message(FATAL_ERROR "❌ Failed to download ONNX Runtime from ${ONNXRUNTIME_URL}")
    message(STATUS "   Download status: ${DOWNLOAD_STATUS_STRING}")
    message(STATUS "   Please check your internet connection and try again")
    # Clean up failed download
    if(EXISTS "${ONNXRUNTIME_ARCHIVE}")
      file(REMOVE "${ONNXRUNTIME_ARCHIVE}")
    endif()
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
    
    # Link OpenCV and ONNX Runtime libraries
    target_link_libraries(${TARGET} PRIVATE
        ${OpenCV_LIBS}
        onnxruntime
    )
    
    # Link GTK3 and related dependencies if found
    if(GTK3_FOUND)
        target_include_directories(${TARGET} PRIVATE ${GTK3_INCLUDE_DIRS})
        target_link_libraries(${TARGET} PRIVATE ${GTK3_LIBRARIES})
        target_compile_options(${TARGET} PRIVATE ${GTK3_CFLAGS_OTHER})
    endif()
    
    # Link additional GTK3 dependencies
    if(GLIB2_FOUND)
        target_include_directories(${TARGET} PRIVATE ${GLIB2_INCLUDE_DIRS})
        target_link_libraries(${TARGET} PRIVATE ${GLIB2_LIBRARIES})
    endif()
    
    if(GIO_FOUND)
        target_include_directories(${TARGET} PRIVATE ${GIO_INCLUDE_DIRS})
        target_link_libraries(${TARGET} PRIVATE ${GIO_LIBRARIES})
    endif()
    
    if(GOBJECT_FOUND)
        target_include_directories(${TARGET} PRIVATE ${GOBJECT_INCLUDE_DIRS})
        target_link_libraries(${TARGET} PRIVATE ${GOBJECT_LIBRARIES})
    endif()
    
    if(CAIRO_FOUND)
        target_include_directories(${TARGET} PRIVATE ${CAIRO_INCLUDE_DIRS})
        target_link_libraries(${TARGET} PRIVATE ${CAIRO_LIBRARIES})
    endif()
    
    # Performance optimization flags
    target_compile_options(${TARGET} PRIVATE -O3 -ffast-math -march=native)
endfunction()

# CPU version
add_bgremover_target(bgremover ${CMAKE_SOURCE_DIR}/main.cpp)

# Ensure U²-Net model is copied to build directory for CPU version
if(U2NET_MODEL_PATH)
    add_custom_command(TARGET bgremover POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${U2NET_MODEL_PATH}"
        "$<TARGET_FILE_DIR:bgremover>/models/u2net.onnx"
    )
endif()

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
    
    # Ensure U²-Net model is copied to build directory for GPU version
    if(U2NET_MODEL_PATH)
        add_custom_command(TARGET bgremover_gpu POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${U2NET_MODEL_PATH}"
            "$<TARGET_FILE_DIR:bgremover_gpu>/models/u2net.onnx"
        )
    endif()
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
