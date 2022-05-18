ARG mlp_version
FROM mlp/base:$mlp_version

ARG mlp_version
ENV MLP_VERSION=$mlp_version
WORKDIR /pipeline
COPY ./ ./

ENV pip=/opt/conda/bin/pip
RUN pip install .
