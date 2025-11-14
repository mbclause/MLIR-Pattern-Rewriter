## MLIR InstCombine

An MLIR optimization pass that rewrites:
```
scf.index_switch (…)-> i32   followed by   arith.shli %res, %c2_i32
```

into an equivalent
```
arith.cmpi eq …, 0  +  arith.select {0,4}
```

This matches the assignment’s input/output shape and demonstrates pattern-based IR rewriting, dominance-safe insertion, and running a custom pass via an mlir-opt-style driver.

# Repo Layout

.
├─ main.cpp        # custom mlir-opt-style driver + pass registration
├─ input.mlir      # sample input IR (assignment input)
└─ README.md

# Prerequisites

Prerequisites

CMake ≥ 3.20

Ninja (recommended) or Make

Clang/LLVM toolchain (C++17)

Python 3

LLVM/MLIR built from source (you need the MLIR CMake packages)


# Install & build LLVM/MLIR (once)

```
# 1) Get sources
git clone https://github.com/llvm/llvm-project.git
cd llvm-project

# 2) Configure (Release build; MLIR enabled)
cmake -G Ninja -S llvm -B build \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_BUILD_TOOLS=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DCMAKE_BUILD_TYPE=Release

# 3) Build
cmake --build build
```

Key paths you’ll use below:

LLVM_DIR = <llvm-project>/build/lib/cmake/llvm

MLIR_DIR = <llvm-project>/build/lib/cmake/mlir

# Build this Project

If your repo already has a CMakeLists.txt, configure with the two package dirs:

```
cmake -S . -B build \
  -DLLVM_DIR=/absolute/path/to/llvm-project/build/lib/cmake/llvm \
  -DMLIR_DIR=/absolute/path/to/llvm-project/build/lib/cmake/mlir
cmake --build build

```

If you need a minimal CMakeLists.txt, this works:

```
cmake_minimum_required(VERSION 3.20)
project(mlir-instcombine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found LLVM at ${LLVM_DIR}")
message(STATUS "Found MLIR at ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_executable(mlir-opt main.cpp)
target_link_libraries(mlir-opt
  PRIVATE
    MLIRArithDialect
    MLIRSCFDialect
    MLIRFuncDialect
    MLIRParser
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRTools     # for MlirOptMain if needed in your build
)

# On some setups you may need:
# target_link_libraries(mlir-opt PRIVATE LLVMCore LLVMSupport)

```

Note: This builds a custom mlir-opt-style tool from main.cpp that already registers your --instcombine pass.

# Run

From the repo root (or wherever your binary lives):

```
# If the binary is under ./build/
./build/mlir-opt --instcombine input.mlir -o output0.mlir
```

(If you installed your binary into PATH, the same command works without ./build/.)

# Expected Output (Shape)

You should see the switch + shift-by-2 rewritten to a cmpi eq + select {0,4}, e.g.:

```
func.func @foo(%arg0: i32) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %0 = arith.cmpi eq, %arg0, %c0_i32 : i32
  %1 = arith.select %0, %c0_i32, %c4_i32 : i32
  return %1 : i32
}

```

# License

MIT

