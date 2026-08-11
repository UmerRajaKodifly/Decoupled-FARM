# Phase 1 — Stella VSLAM Dense (equirect monocular SLAM + PatchMatch)
# Compatible with Docker that lacks COPY --exclude / --parents (BuildKit experimental).
ARG CUDA=12.9.1
ARG UBUNTU=24.04

FROM nvidia/cuda:${CUDA}-devel-ubuntu${UBUNTU} AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
        git curl cmake ninja-build clang pkg-config \
        libatlas-base-dev libsuitesparse-dev binutils-dev \
        libomp-dev libopencv-dev libeigen3-dev libyaml-cpp-dev libsqlite3-dev \
        python3 python3-pip python3-opencv python3-msgpack python3-numpy \
        python3-scipy python3-dev pybind11-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --break-system-packages \
        pybind11-stubgen numba viser==1.0.24 tqdm

ENV C=clang CXX=clang++
ARG NUM_THREADS=0
WORKDIR /tmp

ARG G2O_VERSION=20241228_git
RUN git clone https://github.com/RainerKuemmerle/g2o.git --branch ${G2O_VERSION} --depth 1 && \
    cd g2o && \
    cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_UNITTESTS=OFF \
        -DG2O_USE_CHOLMOD=OFF \
        -DG2O_USE_CSPARSE=ON \
        -DG2O_USE_OPENGL=OFF \
        -DG2O_USE_OPENMP=OFF \
        -DG2O_BUILD_APPS=OFF \
        -DG2O_BUILD_EXAMPLES=OFF \
        -DG2O_BUILD_LINKED_APPS=OFF \
        && \
    cmake --build build --parallel $(nproc) && \
    cmake --install build && \
    rm -rf /tmp/g2o

ARG BACKWARD_CPP_COMMIT=5ffb2c879ebdbea3bdb8477c671e32b1c984beaa
RUN git clone https://github.com/bombela/backward-cpp.git && \
    cd backward-cpp && \
    git checkout ${BACKWARD_CPP_COMMIT} && \
    cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --parallel $(nproc) && \
    cmake --install build && \
    rm -rf /tmp/backward-cpp


FROM base AS build
# Full source tree — tools/example do not affect the C++ cmake target
COPY . /stella
WORKDIR /stella
RUN export PYTHONPATH=/stella/build/lib && \
    cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --parallel $(nproc) && \
    cmake --install build && \
    rm -rf build


FROM nvidia/cuda:${CUDA}-runtime-ubuntu${UBUNTU} AS runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
        libatlas3-base libcxsparse4 libyaml-cpp0.8 \
        libamd3 libcamd3 libcolamd3 libccolamd3 libomp5 \
        python3 python3-opencv python3-msgpack python3-numpy python3-scipy \
        python3-pip coreutils findutils \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --break-system-packages tqdm || \
       pip install --no-cache-dir tqdm || true

# Installed Stella shared libs + any installed binaries
COPY --from=build /usr/local/lib /usr/local/lib
COPY --from=build /usr/local/include /usr/local/include
RUN ldconfig

# Python tools, configs, and entrypoint from build context
COPY tools /stella/tools
COPY example /stella/example
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /stella/tools
ENTRYPOINT ["/entrypoint.sh"]
