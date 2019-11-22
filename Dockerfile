ARG PYTORCH="1.3"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel


RUN apt-get -y update && apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 wget \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

ENV TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
# ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all;-gencode;arch=compute_52,code=sm_52"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# Install HumanDetectionServer
RUN conda install cython -y && conda clean --all
RUN git clone https://github.com/jsong0327/HumanDetectionServer.git /HumanDetectionServer
WORKDIR /HumanDetectionServer
RUN PYTHON=python3 bash ./compile.sh
RUN pip install --no-cache-dir -e .

RUN mkdir /HumanDetectionServer/checkpoints && \
      cd /HumanDetectionServer/checkpoints && \
      wget https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth

RUN pip uninstall -y mmcv
RUN git clone https://github.com/jsong0327/mmcv.git && \
      cd mmcv
      pip install . && \
      cd .. && \
      rm -rf mmcv

