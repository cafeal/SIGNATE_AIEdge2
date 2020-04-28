# FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

WORKDIR /home
ARG PYTHON_VERSION=3.6.9
ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

RUN apt-get update \
    && apt-get install -y \
         build-essential cmake git curl wget \
         libssl-dev libbz2-dev libreadline-dev libsqlite3-dev libopencv-dev

ENV PATH=/opt/python$PYTHON_VERSION/bin:$PATH
RUN git clone https://github.com/pyenv/pyenv.git /tmp/pyenv \
    && cd /tmp/pyenv/plugins/python-build \
    && ./install.sh \
    && cd ~ \
    && /usr/local/bin/python-build $PYTHON_VERSION /opt/python$PYTHON_VERSION

RUN pip install pipenv

COPY Pipfile ${WORKDIR}/Pipfile
COPY src ${WORKDIR}/src
RUN cd ${WORKDIR} \
    && pipenv install --system --skip-lock --dev

COPY dockerfiles/advanced_activations.py \
     /opt/python${PYTHON_VERSION}/lib/python3.6/site-packages/tensorflow/python/keras/layers/advanced_activations.py

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH