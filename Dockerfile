# Use the official Miniconda3 base image
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update && \
  apt-get install -y tmux \
  curl \
  wget \
  && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh \
    && mkdir /root/.conda
RUN bash Miniconda3-latest-Linux-aarch64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

WORKDIR /app

COPY environment.yml /app/

RUN conda env create -f environment.yml
RUN conda install -c anaconda gcc_linux-aarch64
RUN pip install -q emoji pythainlp==2.2.4 sefr_cut tinydb seqeval sentencepiece pydantic jsonlines
RUN pip install --no-deps thai2transformers==0.1.2


CMD ["/bin/bash"]
