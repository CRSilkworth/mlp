# mlp
All the necessary setup to run tfx pipelines using either beam (primarily to run locally for testing) or kubeflow on GCP's ai platform for production. In addition, some custom components, and utilities are made available that lowers the overhead in starting a new pipeline from scratch. The idea being that you set up this repository at the beginning and clone your own pipeline definition into here or use the utilities to start a new one from scratch.
## Set up

### Prerequisites
* Update system python 3
```
# MAC OS
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
export PATH="/usr/local/bin:/usr/local/sbin:$PATH"
brew update
brew install python

# Ubuntu
sudo apt-get install python3.6
sudo apt-get install \
    build-essential libssl-dev libffi-dev \
    libxml2-dev libxslt1-dev zlib1g-dev \
    python3-pip git software-properties-common
sudo apt-get install python3.6-dev
```

* docker must be installed( [mac](https://docs.docker.com/docker-for-mac/install/), [ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/))

* Ensure that your machine is able to access the mlp github repo via ssh. [follow these instructions](https://help.github.com/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/)

### Inference only install
* Pull tensorflow serving docker image
```
docker pull tensorflow/serving:2.0.0
```

### Install for Beam
* Checkout the mlp repository and build the mlp/beam docker imagine.
```
git clone --branch v{version} git+ssh://git@github.com/CRSilkworth/mlp.git
cd mlp
build . -f Dockerfile.beam -t mlp/beam:latest
cd {mlp_project}
build . -f Dockerfile.beam -t {mlp_project}/beam:latest
```
* Note: the example project assumes data will be pulled directly from BigQuery which is not possible without a GCP project setup.

### Install for kubeflow
* Must have GCP environment setup with all the relevant apis enabled for the GCP project. (only required for kubeflow, or if you plan to run from GCP):
  * [Storage](https://console.cloud.google.com/compute/instances)
  * [BigQuery](https://console.cloud.google.com/apis/api/bigquery.googleapis.com/overview) - If that's where your raw data is being stored.
  * [AI platform](https://console.cloud.google.com/ai-platform/pipelines) - If using kubeflow
  * [Dataflow](https://console.cloud.google.com/dataflow) - If using a dataflow runner (recommended)
  * [Kubeflow pipelines](https://console.cloud.google.com/marketplace/details/google-cloud-ai-platform/kubeflow-pipelines?project=booming-cosine-217602) -may not need to explicitly set this up since it might be handled by ai platform
  * [GPU quota](https://console.cloud.google.com/iam-admin/quotas) - Must ask for whatever number of GPUs you need to use in your corresponding region.

* Get gcp credentials from [here](https://console.cloud.google.com/apis/credentials), and store the downloaded json file somewhere safe. Point the GOOGLE_APPLICATION_CREDENTIALS towards that file. You'll have to do this every time you open a new terminal:
```
export GOOGLE_APPLICATION_CREDENTIALS=path/to/secrets/{gcp_project}-{key_id}.json
```

* Checkout the ml pipelines repo.
```
git clone --branch v{version} git+ssh://git@github.com/CRSilkworth/mlp.git
cd mlp
export PYTHONPATH=$PYTHONPATH:$PWD
```

* Build the base docker image
```
docker build . -f Dockerfile.gpu -t gcr.io/mlp/base:latest
```

## Running a pipeline
### Using beam
* Ensure that the environmental variables from the installation section are set.
* Create or edit a beam pipeline file from the pipelines directory ({mlp_subproject}/pipelines/beam/). Adjust any of the input variables, i.e. the variables uppercased beginning with an underscore (e.g. \_NUM_TRAIN_STEPS) and run:
```
docker run --gpus all -it -v $PWD:/tmp -w /tmp --rm \
  path_nn/beam:latest \
  python {mlp_subproject}/pipelines/beam/{pipeline_file}
```
Outputs will be written to a directory path_nn/runs/ by default.

### Using kubeflow
* Ensure that the environmental variables from the installation section are set. Also, make sure you have successfully built the Docker.kubeflow image from above. Running:
```
docker images
```
should show an image with the REPOSITORY = 'gcr.io/mlp/base:latest'.

* cd to the base mlp directory, _you must run it from here!_
```
cd {mlp_project}
```

* Create or edit a beam pipeline file from the pipelines directory of the project/subproject you want to run (e.g. mlp/example_project/example_subproject/pipelines/beam/bigquery_to_pusher.py). Adjust any of the input variables, i.e. the variables uppercased beginning with an underscore (e.g. \_NUM_TRAIN_STEPS), or the ai_platform args if you want to change the VMs that the training process is being run on. Create the pipeline using the built in tfx tool:
```
tfx pipeline create  --endpoint {ai_platform_pipeline_endpoint} --build_target_image gcr.io/mlp/{mlp_project} --pipeline_path {mlp_subproject}/pipelines/{pipeline_file_name}
```
Where the endpoint can be taken from the [AI platform page](https://console.cloud.google.com/ai-platform/pipelines/clusters) (after enabling the api and setting up a kubeflow cluster, etc.) by clicking on the open pipelines dashboard and taking the url of the form: {hash_string}-dot-{region}.pipelines.googleusercontent.com. This creation takes a while, 30min~1 hour.

* Start a run:
```
tfx run create --pipeline_name {mlp_project}-{mlp_subproject}-{pipeline_type} --endpoint  {ai_platform_pipeline_endpoint}
```
You can get the pipeline name either from the python file used to create the pipeline or from the pipelines dashboard.

* To update an existing pipeline:
```
tfx pipeline update  --endpoint {ai_platform_pipeline_endpoint} --pipeline_path {mlp_subproject}/pipelines/kubeflow/{pipeline_file_name}
```

* To delete an existing pipeline:
```
tfx pipeline delete --pipeline_name {mlp_project}-{mlp_subproject}-{pipeline_type} --endpoint  {ai_platform_pipeline_endpoint}
```
## Create project skeleton
* A simple script is provided to give you a the skeleton of a project in order to make the process of starting a project from scratch less tedious. The script is fairly straightforward. Follow the instructions for a beam install then:
```
cd mlp
docker run --gpus all -it -v $PWD:/tmp -w /tmp --rm \
  path_nn/beam:latest \
  python create_project.py --help
```
