#!/usr/bin/env bash
export LANG=C.UTF-8
set -e

# =========================================================
# 🧱 Build OpenCV 4.12.0 with CUDA 12.8 + cuDNN 9 support
# =========================================================

# ======= Configuration =======
OPENCV_VERSION="4.12.0"                      # valid tag
BUILD_DIR="$HOME/opencv_build"
CONDA_ENV_NAME="opencv_cuda12"               # <-- your active Conda env name
CONDA_PREFIX=$(conda info --base)/envs/${CONDA_ENV_NAME}
INSTALL_PREFIX="${CONDA_PREFIX}"        # install inside the Conda env
LAUNCHER_PATH="$HOME/.local/bin/opencv_py"

# ===== Activate Conda environment =====
echo "🐍 Activating Conda environment: ${CONDA_ENV_NAME}"

# Initialize Conda for non-interactive shell
eval "$($(conda info --base)/bin/conda shell.bash hook)"

# Now activate the env
conda activate "${CONDA_ENV_NAME}"
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}"        # install inside the Conda env
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export PKG_CONFIG_PATH="${CONDA_PREFIX}/lib/pkgconfig"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# ===== Detect Python inside Conda =====
echo "🔍 Detecting Python inside Conda environment..."

PYTHON_BIN="$(command -v python)"
if [[ -z "$PYTHON_BIN" ]]; then
    echo "❌ ERROR: No Python interpreter found after activating ${CONDA_ENV_NAME}."
    echo "   Debug info: PATH=${PATH}"
    exit 1
fi

PYTHON_VER="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PYTHON_INCLUDE="$($PYTHON_BIN -c 'import sysconfig; print(sysconfig.get_paths().get("include", ""))')"
PYTHON_PACKAGES="$($PYTHON_BIN -c 'import sys, site, os; print(site.getsitepackages()[0] if hasattr(site, "getsitepackages") else os.path.join(sys.prefix, f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"))')"

if [[ ! -d "$PYTHON_INCLUDE" ]]; then
    PYTHON_INCLUDE="$CONDA_PREFIX/include/python${PYTHON_VER}"
fi
if [[ ! -d "$PYTHON_PACKAGES" ]]; then
    PYTHON_PACKAGES="$CONDA_PREFIX/lib/python${PYTHON_VER}/site-packages"
fi

echo "➡ Using Python: ${PYTHON_BIN} (v${PYTHON_VER})"
echo "   include:   ${PYTHON_INCLUDE}"
echo "   packages:  ${PYTHON_PACKAGES}"

# ===== Dependencies =====
echo "📦 Installing Linux build dependencies..."
sudo apt update -y
sudo apt install -y build-essential cmake git pkg-config \
    libtiff-dev libjpeg-dev libpng-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libgtk-3-dev libcanberra-gtk3-module \
    libatlas-base-dev gfortran python3-dev python3-numpy libglib2.0-dev

# ===== Workspace =====
mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"

# ===== Clone sources =====
if [ ! -d "opencv" ]; then
  git clone -b ${OPENCV_VERSION} https://github.com/opencv/opencv.git
else
  echo "✅ OpenCV source already exists."
fi

if [ ! -d "opencv_contrib" ]; then
  git clone -b ${OPENCV_VERSION} https://github.com/opencv/opencv_contrib.git
else
  echo "✅ OpenCV_contrib source already exists."
fi

# ===== Clean build =====
cd opencv
rm -rf build && mkdir build && cd build

# ===== CMake configuration (CUDA 12.8) =====
echo "⚙️  Configuring OpenCV with CUDA 12.8..."
cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
      -D CMAKE_INSTALL_RPATH="${CONDA_PREFIX}/lib" \
      -D OPENCV_EXTRA_MODULES_PATH=${BUILD_DIR}/opencv_contrib/modules \
      -D ZLIB_LIBRARY="${CONDA_PREFIX}/lib/libz.so" \
      -D ZLIB_INCLUDE_DIR="${CONDA_PREFIX}/include" \
      -D WITH_OPENCL=OFF \
      -D WITH_CUDA=ON \
      -D CUDA_ARCH_BIN=89 \
      -D WITH_CUDNN=ON \
      -D WITH_CUBLAS=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D WITH_OPENBLAS=ON \
      -D WITH_LAPACK=ON \
      -D BUILD_PROTOBUF=OFF \
      -D PROTOBUF_UPDATE_FILES=ON \
      -D OPENCV_DNN_PROTOBUF_ENABLED=ON \
      -D LAPACK_LIBRARIES="${CONDA_PREFIX}/lib/libopenblas.so" \
      -D OpenBLAS_INCLUDE_DIRS="${CONDA_PREFIX}/include" \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D CMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
      -D CUDA_HOST_COMPILER=/usr/bin/g++-12 \
      -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8 \
      -D PYTHON3_EXECUTABLE=${PYTHON_BIN} \
      -D PYTHON3_INCLUDE_DIR=${PYTHON_INCLUDE} \
      -D PYTHON3_PACKAGES_PATH=${PYTHON_PACKAGES} \
      -D BUILD_opencv_python3=ON \
      -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF \
      -D BUILD_EXAMPLES=OFF -D BUILD_opencv_apps=OFF -D BUILD_opencv_world=OFF \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D ENABLE_CCACHE=ON \
      ..

# ===== Build & install =====
echo "Building OpenCV ..."
make -j$(nproc)

echo "📀 Installing into Conda environment..."
make install
sudo ldconfig

# =========================================================
# ✅ Post‑build setup: launcher & verification
# =========================================================

echo "🧠 Detecting system GLib..."
GLIB_PATH=$(whereis libglib-2.0.so.0 | awk '{print $2}' || true)
if [ -z "$GLIB_PATH" ]; then
    echo "❌ ERROR: libglib-2.0.so.0 not found. Install 'libglib2.0-0' and retry."
    exit 1
fi
echo "🔹 Found GLib: $GLIB_PATH"

# Create launcher
mkdir -p "$(dirname "$LAUNCHER_PATH")"

echo "📝 Creating launcher script: $LAUNCHER_PATH"
cat <<EOF > "$LAUNCHER_PATH"
#!/usr/bin/env bash
# Auto‑generated OpenCV launcher for Conda + GLib preload
if [ -f "\$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "\$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "\$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "\$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate ${CONDA_ENV_NAME} 2>/dev/null || true
export LD_PRELOAD=${GLIB_PATH}
exec python "\$@"
EOF

chmod +x "$LAUNCHER_PATH"
echo "✅ Launcher ready: $LAUNCHER_PATH"

# Add to PATH (if not already)
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    export PATH="$HOME/.local/bin:$PATH"
fi

# =========================================================
# 🧪 Post‑Build Verification
# =========================================================
echo "🧪 Verifying OpenCV with CUDA 12.8/cuDNN..."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV_NAME}

LD_PRELOAD=${GLIB_PATH} python - <<'PY'
import cv2;
print("✅ OpenCV CUDA/cuDNN import successful!");
print("Version:", cv2.__version__);
n = cv2.cuda.getCudaEnabledDeviceCount();
print("CUDA devices:", n);
if n:
    props = cv2.cuda.getDeviceProperties(0)
    print("GPU:", props.name)
info = cv2.getBuildInformation()
print("cuDNN:", "YES" if "cuDNN: YES" in info else "NO");
PY

echo
echo "🚀 Done! Run GPU-enabled scripts via:"
echo "   opencv_py my_script.py"
echo
echo "👉 To export the launcher globally:"
echo "   echo 'export PATH=\$HOME/.local/bin:\$PATH' >> ~/.bashrc && source ~/.bashrc"