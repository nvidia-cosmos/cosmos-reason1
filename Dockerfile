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

RUN pip install -U pip setuptools wheel packaging
RUN pip install torch==2.6.0

COPY requirements.txt /workspace/cosmos_reason1/requirements.txt
RUN pip install -r /workspace/cosmos_reason1/requirements.txt

FROM base as package

COPY setup.py /workspace/cosmos_reason1/setup.py
COPY CMakeLists.txt /workspace/cosmos_reason1/CMakeLists.txt
COPY tools /workspace/cosmos_reason1/tools
COPY configs /workspace/cosmos_reason1/configs
COPY cosmos_reason1 /workspace/cosmos_reason1/cosmos_reason1

RUN cd /workspace/cosmos_reason1 && pip install -e . && cd -
