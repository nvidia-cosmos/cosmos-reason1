FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 as base

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update -qq && apt-get install -qq -y --allow-change-held-packages \
    build-essential tzdata git openssh-server curl netcat elfutils \
    python3.10 python3.10-dev python3.10-venv python3-pip python-is-python3 \
    lsb-release gpg

# Download and add Redis GPG key
RUN curl -fsSL https://packages.redis.io/gpg  | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg && \
    chmod 644 /usr/share/keyrings/redis-archive-keyring.gpg

# Add Redis APT repository
RUN echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb  $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list

# Update package list
RUN apt-get update -qq

# Install specific Redis version
RUN apt-get install -qq -y redis-server

#################################################
## Install EFA installer
ARG EFA_INSTALLER_VERSION=1.38.1
RUN cd $HOME \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && tar -xf $HOME/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify \
    && rm -rf $HOME/aws-efa-installer

###################################################
## Install AWS-OFI-NCCL plugin
ARG AWS_OFI_NCCL_VERSION=v1.14.0
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libhwloc-dev
#Switch from sh to bash to allow parameter expansion
SHELL ["/bin/bash", "-c"]
RUN curl -OL https://github.com/aws/aws-ofi-nccl/releases/download/${AWS_OFI_NCCL_VERSION}/aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v}.tar.gz \
    && tar -xf aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v}.tar.gz \
    && cd aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v} \
    && ./configure --prefix=/opt/aws-ofi-nccl/install \
        --with-mpi=/opt/amazon/openmpi \
        --with-libfabric=/opt/amazon/efa \
        --with-cuda=/usr/local/cuda \
        --enable-platform-aws \
    && make -j $(nproc) \
    && make install \
    && cd .. \
    && rm -rf aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v} \
    && rm aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v}.tar.gz

#################################################
## Install NVIDIA GDRCopy
##
## NOTE: if `nccl-tests` or `/opt/gdrcopy/bin/sanity -v` crashes with incompatible version, ensure
## that the cuda-compat-xx-x package is the latest.
ARG GDRCOPY_VERSION=v2.4.4
RUN git clone -b ${GDRCOPY_VERSION} https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && cd /tmp/gdrcopy \
    && make prefix=/opt/gdrcopy install

RUN pip install -U pip setuptools wheel packaging
RUN pip install torch==2.6.0

COPY requirements.txt /workspace/cosmos_reason1/requirements.txt
COPY constraints.txt /workspace/cosmos_reason1/constraints.txt
RUN pip install -r /workspace/cosmos_reason1/requirements.txt

FROM base as package

COPY setup.py /workspace/cosmos_reason1/setup.py
COPY CMakeLists.txt /workspace/cosmos_reason1/CMakeLists.txt
COPY tools /workspace/cosmos_reason1/tools
COPY configs /workspace/cosmos_reason1/configs
COPY cosmos_reason1 /workspace/cosmos_reason1/cosmos_reason1

RUN cd /workspace/cosmos_reason1 && pip install -e . && cd -
