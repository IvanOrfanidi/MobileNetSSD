FROM iorfanidi/ubuntu-20.04-gcc-cmake-git-boost:latest

LABEL maintainer="i.orfanidi@mail.ru"

COPY . /MobileNetSSD

WORKDIR /MobileNetSSD

RUN apt-get update && \
    apt-get install -y libopencv-dev && \
    rm -rf build && \
    mkdir build && \
    cd build && \
    cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release .. && \
    cmake --build .
