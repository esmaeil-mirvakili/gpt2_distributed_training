FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.1.0-transformers4.36.0-gpu-py310-cu121-ubuntu20.04
LABEL authors="esmaeil"

RUN apt-get update && apt-get install gcc -y
RUN pip install sagemaker-pytorch-training
RUN pip install tiktoken
