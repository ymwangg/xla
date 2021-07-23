# Use nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04 or later for CUDA.
# Warning: All ARGs placed before FROM will only be scoped up unitl FROM statement.
# https://github.com/docker/cli/blob/3c7ede6a68941f64c3a154c9a753eb7a9b1c2c3e/docs/reference/builder.md#understand-how-arg-and-from-interact
ARG base_image="debian:stretch"
FROM "${base_image}"

ARG python_version="3.6"
ARG release_version="nightly"
ARG cuda="1"
ARG cuda_compute="7.0,7.5"
ARG cxx_abi="0"
ARG PT_XLA_BRANCH
ARG AWS_CONTAINER_CREDENTIALS_RELATIVE_URI
ARG AWS_DEFAULT_REGION

RUN apt-get update
RUN apt-get install -y git sudo python-pip python3-pip
RUN apt-get install -y curl unzip && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip > /dev/null && \
    ./aws/install && \
    git config --global credential.helper '!aws codecommit credential-helper $@' && \
    git config --global credential.UseHttpPath true
RUN git clone -b hopper/main https://git-codecommit.us-west-2.amazonaws.com/v1/repos/PyTorch pytorch

# Disable CUDA for PyTorch
ENV USE_CUDA "0"

# Enable CUDA for XLA
ENV XLA_CUDA "${cuda}"
ENV TF_CUDA_COMPUTE_CAPABILITIES "${cuda_compute}"

# Whether to build torch and torch_xla libraries with CXX ABI
ENV CXX_ABI "${cxx_abi}"

COPY . /pytorch/xla
RUN if [ ! -z "${PT_XLA_BRANCH}" ]; \
  then \
    cd /pytorch && \
    rm -rf xla && \
    git clone -b "${PT_XLA_BRANCH}" --recursive https://git-codecommit.us-west-2.amazonaws.com/v1/repos/PyTorchXla xla; \
  fi

RUN cd /pytorch && bash xla/scripts/build_torch_wheels.sh ${python_version} ${release_version}

# Use conda environment on startup or when running scripts.
RUN echo "conda activate pytorch" >> ~/.bashrc
RUN echo "export TF_CPP_LOG_THREAD_ID=1" >> ~/.bashrc
ENV PATH /root/anaconda3/envs/pytorch/bin/:/root/bin:$PATH

# Define entrypoint and cmd
COPY docker/docker-entrypoint.sh /usr/local/bin
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["bash"]
