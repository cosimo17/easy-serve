FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04
ARG http_proxy
ARG https_proxy
ARG no_proxy

ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV no_proxy=${no_proxy}

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl bzip2 git libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6\
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /tmp/deps
COPY requirements.txt /tmp/deps/requirements.txt

RUN umask 0 \
    && mkdir -p /tmp/deps \
    && cd /tmp/deps \
    && curl -L https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 \
    && rm -rf Miniconda3-latest-Linux-x86_64.sh \
    && . /opt/miniconda3/bin/activate \
    && conda create -n easy-serve python=3.8 \
    && conda activate easy-serve \
    && pip install -r requirements.txt \
    && conda clean -y --all \
    && cd / \
    && rm -rf /tmp/*

ENV PATH /opt/miniconda3/envs/easy-serve/bin:$PATH
EXPOSE 9876
