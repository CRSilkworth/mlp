FROM tensorflow/tfx:1.9.0

WORKDIR /mlp
COPY ./mlp ./mlp
COPY requirements.txt ./
COPY version.py ./
COPY setup.py ./
COPY proto_gen.py ./
COPY README.md ./

ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda/lib64
ENV pip=/opt/conda/bin/pip
ENTRYPOINT ["/opt/apache/beam/boot"]
RUN pip install .
