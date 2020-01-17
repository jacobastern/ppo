# Python Dockerfile
#
# https://github.com/dockerfile/python
#

# Pull base image.
FROM ubuntu

# Install Python.
RUN \
  apt-get update && \
  apt-get install -y \
  chromium-browser \
  libopenmpi-dev \
  openssh-server \
  python3 \
  python3-dev \
  python3-pip \
  python3-virtualenv \
  && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

RUN pip3 --no-cache-dir install \
    argparse \
    beautifulsoup4 \
    cython \
    gym \
    gym[atari] \
    imageio \
    ipykernel \
    ipyparallel \
    jupyter \
    matplotlib \
    mpi4py \
    numpy \
    opencv-python \
    pprint \
    requests \
    scipy \
    seaborn \
    selenium \
    setuptools \
    sklearn \
    sympy \
    torch \
    torchvision \
    tqdm \
    xgboost \
    h5py \
    && python3 -m ipykernel.kernelspec

# Define working directory.
WORKDIR /code

# Define default command.
CMD ["bash"]