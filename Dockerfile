FROM tensorflow/tfx:1.7.0

WORKDIR /mlp
COPY ./mlp ./mlp
COPY requirements.txt ./
COPY version.py ./
COPY setup.py ./
COPY proto_gen.py ./
COPY README.md ./
RUN pip3 install .
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda/lib64
