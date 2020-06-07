FROM gcr.io/mlp/gpu-setup:latest
WORKDIR /pipeline
COPY ./ ./
RUN pip3 install .
