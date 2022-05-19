# mlp
All the necessary setup to run tfx pipelines using either beam (primarily to run locally for testing) or kubeflow on GCP for production. In addition, some custom components, and utilities are made available that lowers the overhead in starting a new pipeline from scratch. The idea being that you set up this repository at the beginning and clone your own pipeline definition into here or use the utilities to start a new one from scratch.

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
sudo apt-get install python3.7
sudo apt-get install \
    build-essential libssl-dev libffi-dev \
    libxml2-dev libxslt1-dev zlib1g-dev \
    python3-pip git software-properties-common
sudo apt-get install python3.6-dev
```

* docker must be installed( [mac](https://docs.docker.com/docker-for-mac/install/), [ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/))

* Ensure that your machine is able to access the mlp github repo via ssh. [follow these instructions](https://help.github.com/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/)

### Set up GCP Services (See README in kubeflow_deploy)
## Create project skeleton
* A simple script is provided to give you a the skeleton of a project in order to make the process of starting a project from scratch less tedious. The script is fairly straightforward. Follow the instructions for a beam install then:
```
cd mlp
docker run --gpus all -it -v $PWD:/tmp -w /tmp --rm \
  crsilkworth/mlp:latest \
  python create_project.py --help
```
## Running a pipeline
### Using beam
* Create or edit a beam pipeline file from the pipelines directory (e.g. example_project/example_subproject/pipelines/beam_bigquery_to_pusher.py). Adjust any of the input variables and run:
```
cd $MLP_PROJECT_DIR
source set_env.sh

docker run --gpus all -it \
  -v ${PWD}:/tmp_dir \
  -w /tmp_dir --rm \
  -v ~/runs/:/root/runs/ \
  -v $GOOGLE_APPLICATION_CREDENTIALS:/home/secrets/<gcp_credentials_json> \
  -e GOOGLE_APPLICATION_CREDENTIALS=/home/secrets/<gcp_credentials_json> \
  -e PYTHONPATH=/tmp_dir \
  crsilkworth/mlp:$MLP_VERSION python <path_to_pipeline_file>
```
Where
  * <gcp_credentials_json> should match the file set in set_env.sh.
  * <path_to_pipeline_file> is the path to some beam pipeline file to run (e.g. example_project/example_subproject/pipelines/beam/bigquery_to_pusher.py)

### Using kubeflow
* Use the helper script create_update_run.py to create/update pipelines and launch runs.
```
cd $MLP_PROJECT_DIR
source set_env.sh
python create_update_run.py --pipeline_path <path_to_pipeline_file>
```
