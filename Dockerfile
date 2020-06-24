FROM tensorflow/tensorflow:2.1.1-gpu
FROM tensorflow/tfx:0.21.4

WORKDIR /mlp
COPY ./mlp ./mlp
COPY requirements.txt ./
COPY version.py ./
COPY setup.py ./
COPY proto_gen.py ./
COPY README.md ./
RUN pip3 install .
