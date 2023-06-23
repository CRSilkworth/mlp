FROM tensorflow/tensorflow:2.12.0-gpu

WORKDIR /mlp
COPY ./mlp ./mlp
COPY requirements.txt ./
COPY version.py ./
COPY setup.py ./
COPY proto_gen.py ./
COPY README.md ./

ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda/lib64
RUN pip install --upgrade pip
RUN pip install .
