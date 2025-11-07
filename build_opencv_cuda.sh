#!/usr/bin/env bash
set -e

# =========================================================
# 🧱 Build OpenCV with CUDA/cuDNN support and safe runtime
# =========================================================

# ======= Configuration =======
OPENCV_VERSION="4.12.0"                # change if desired (valid tag)
BUILD_DIR="$HOME/opencv_build"
INSTALL_PREFIX="/usr/local"
CONDA_ENV_NAME="base"                  # set your conda env here
LAUNCHER_PATH="$HOME/.local/bin/opencv_py"

# ===== Detect system Python within Conda =====
echo "🔍 Detecting Python..."
PYTHON_BIN=$(which python3)
PYTHON_INCLUDE=$($PYTHON_BIN -c "from sysconfig import get_paths as g; print(g()['include'])")
PYTHON_PACKAGES=$($PYTHON_BIN -c "from sysconfig import get_paths as g; print(g()['purelib'])")

echo "➡ Using Python: ${PYTHON_BIN}"

# ===== Install system dependencies =====
echo "📦 Installing Linux build dependencies..."
sudo apt update -y
sudo apt install -y build-essential cmake git pkg-config \
    libtiff-dev libjpeg-dev libpng-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libgtk-3-dev libcanberra-gtk3-module \
    libatlas-base-dev gfortran python3-dev python3-numpy

# ===== Create build workspace =====
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# ===== Clone sources =====
if [ ! -d "opencv" ]; then
  git clone -b $OPENCV_VERSION https://github.com/opencv/opencv.git
fi
if [ ! -d "opencv_contrib" ]; then
  git clone -b $OPENCV_VERSION https://github.com/opencv/opencv_contrib.git
fi

# ===== Build OpenCV =====
cd opencv
rm -rf build && mkdir build && cd build

echo "⚙️  Running CMake configuration..."
cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
      -D OPENCV_EXTRA_MODULES_PATH=${BUILD_DIR}/opencv_contrib/modules \
      -D WITH_CUDA=ON \
      -D WITH_CUDNN=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D BUILD_opencv_python3=ON \
      -D PYTHON3_EXECUTABLE=${PYTHON_BIN} \
      -D PYTHON3_INCLUDE_DIR=${PYTHON_INCLUDE} \
      -D PYTHON3_PACKAGES_PATH=${PYTHON_PACKAGES} \
      -D ENABLE_CCACHE=ON \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_opencv_apps=OFF \
      -D WITH_NVCUVID=OFF \
      -D WITH_NVCUVENC=OFF \
      ..

echo "🏗️  Building OpenCV (this may take a while)..."
make -j"$(nproc)"
echo "📀 Installing (sudo required)..."
sudo make install
sudo ldconfig

# =========================================================
# ✅ Post‑Build Setup: configure safe runtime environment
# =========================================================

echo "🧠 Detecting system GLib..."
GLIB_PATH=$(whereis libglib-2.0.so.0 | awk '{print $2}')
if [ -z "$GLIB_PATH" ]; then
    echo "❌ ERROR: Could not locate system libglib-2.0.so.0"
    echo "Please install 'libglib2.0-0' and rerun."
    exit 1
fi
echo "🔹 Found system GLib at: $GLIB_PATH"

# Create launcher directory
mkdir -p "$(dirname "$LAUNCHER_PATH")"

echo "📝 Creating launcher script: $LAUNCHER_PATH"
cat <<EOF > "$LAUNCHER_PATH"
#!/usr/bin/env bash
# Auto-generated OpenCV launcher with proper GLib preload
# Activates conda ('${CONDA_ENV_NAME}') and runs python with correct GLib

# Conda initialization (assumed installed in ~/miniconda3 or ~/anaconda3)
if [ -f "\$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "\$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "\$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "\$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "⚠️  Conda initialization script not found — skipping activation."
fi

# Activate the environment
conda activate ${CONDA_ENV_NAME} 2>/dev/null || true

# Preload correct GLib to avoid libatspi/GLib mismatch in GTK
export LD_PRELOAD=${GLIB_PATH}

# Run Python (pass-through)
exec python3 "\$@"
EOF

chmod +x "$LAUNCHER_PATH"
echo "✅ Created launcher at: $LAUNCHER_PATH"

if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo "📋 Adding ~/.local/bin to PATH for this session..."
    export PATH="$HOME/.local/bin:$PATH"
fi

# =========================================================
# 🧪 Automated Post‑Build Test
# =========================================================
echo "🧪 Testing OpenCV in Conda environment..."

source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate ${CONDA_ENV_NAME} 2>/dev/null || true

LD_PRELOAD=${GLIB_PATH} python3 - <<'PY'
import cv2, sys
print("✅ OpenCV CUDA/cuDNN import test successful.")
print(f"Version: {cv2.__version__}")
print(f"CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
print("cuDNN:", "YES" if "cuDNN:                         YES" in cv2.getBuildInformation() else "NO")
PY

echo
echo "🚀 Done! You can now safely run OpenCV with:"
echo "  opencv_py my_script.py"
echo
echo "To make 'opencv_py' globally available, add this to ~/.bashrc:"
echo "  export PATH=\$HOME/.local/bin:\$PATH"
echo