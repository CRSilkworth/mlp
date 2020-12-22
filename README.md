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
sudo apt-get install python3.6
sudo apt-get install \
    build-essential libssl-dev libffi-dev \
    libxml2-dev libxslt1-dev zlib1g-dev \
    python3-pip git software-properties-common
sudo apt-get install python3.6-dev
```

* docker must be installed( [mac](https://docs.docker.com/docker-for-mac/install/), [ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/))

* If using kubeflow, install skaffold:
```
curl -Lo skaffold https://storage.googleapis.com/skaffold/releases/latest/skaffold-linux-amd64 && \
sudo install skaffold /usr/local/bin/
```

* Ensure that your machine is able to access the mlp github repo via ssh. [follow these instructions](https://help.github.com/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/)

### Inference only install
* Pull tensorflow serving docker image
```
docker pull tensorflow/serving:<tf_serving_version>
```

### Set up GCP Services
If you plan to use kubeflow on gcp, then you will likely need to enable and set up each of these services individually. If you only want to orchestrate with beam then you may not need to set up most (or possibly any) of these services. It should be noted that the example_project uses BigQuery as a data source however.

* Must have GCP environment setup with all the relevant apis enabled for the GCP project. (only required for kubeflow, or if you plan to run from GCP):
  * [Storage](https://console.cloud.google.com/compute/instances) - For storing output from the various components.
  * [BigQuery](https://console.cloud.google.com/apis/api/bigquery.googleapis.com/overview) - If that's where your raw data is being stored.
  * [OAuth 2.0 Client](https://console.cloud.google.com/apis/credentials) - necessary to create one if you're setting up a kubeflow deployment.
  * [Kubeflow](https://www.kubeflow.org/docs/) - There are many ways to set up kubeflow. It can be quite difficult to set up as the documentation is still very much in flux.
    * [AI platform](https://console.cloud.google.com/ai-platform/pipelines) - Using the kubeflow deployed by AI platform is by far the easiest setup, however they were still working out a lot of the bugs at the time of this writing.
    * [Kubeflow on GCP](https://www.kubeflow.org/docs/gke/deploy/) - This is method that is currently being used in this documentation (v1.0).
  * [Dataflow](https://console.cloud.google.com/dataflow) - If using a dataflow runner. This allows the upstream components (like ExampleGen and Transform) automatically scale the number of workers. Recommended if you dealing with a lot of data.
  * [GPU quota](https://console.cloud.google.com/iam-admin/quotas) - Must ask for whatever number of GPUs you need to use in your corresponding region.

* To deploy a kubeflow cluster to GCP you can use the script located in mlp/kubeflow_deploy/setup.sh.
  * First install [kfctl](https://github.com/kubeflow/kfctl/releases/tag/v1.0.2) and [gcloud](https://cloud.google.com/sdk/docs/downloads-apt-get)
  ```
  curl -L https://github.com/kubeflow/kfctl/releases/download/v1.0.2/kfctl_v1.0.2-0-ga476281_linux.tar.gz
  tar -xvf kfctl_v1.0.2_<platform>.tar.gz
  sudo cp ./kfctl /usr/local/bin/

  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
  sudo apt-get install apt-transport-https ca-certificates gnupg
  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
  sudo apt-get update && sudo apt-get install google-cloud-sdk

  gcloud init
  gcloud auth login
  gcloud auth application-default login
  ```
  * Replace all variable of the form <var_name> with their relevant values.
  * Set your oauth client secrete associated with the oauth client id:
  ```
  export CLIENT_SECRET='<secret>'
  ```
  * Open up the setup.sh file and replace all the variables of the form '<var_name>' with the corresponding values.
  * Run setup script
  ```
  cd kubeflow_deploy
  source setup.sh
  ```

### Install for Beam
Beam is the simpler orchestration method. It runs directly from the local machine, and although it uses tensorflow gpu it is probably more useful as for end to end testing of the whole pipeline rather than something to use in production.

* Checkout the mlp repository and build the mlp/beam docker imagine.
```
git clone --branch v<version> git+ssh://git@github.com/CRSilkworth/mlp.git
cd mlp
docker build . -f Dockerfile.beam -t mlp/beam:latest
```
* Note: the example project assumes data will be pulled directly from BigQuery which is not possible without a GCP project setup.

### Install for kubeflow
Kubeflow is the more production ready orchestration method. It is an ML specialized kubernetes deployment that can handle scaling relatively easily, although it is really quite challenging to set up. Make sure you set up the GCP services described above.

* Checkout the mlp repo.
```
git clone --branch v<version> git+ssh://git@github.com/CRSilkworth/mlp.git

cd mlp
export PYTHONPATH=$PYTHONPATH:$PWD
```

* Build the base docker image
```
docker build . -f Dockerfile.kubeflow -t mlp/base:latest
```

## Running a pipeline
### Using beam
* Ensure that the environmental variables from the installation section are set. If you used create_project.py and filled in all the relevant information when running, these variables should be located in an file called set_env.sh:
```
cd <mlp_project>
source set_env.sh
```

* Create or edit a beam pipeline file from the pipelines directory (<mlp_subproject>/pipelines/beam/). Adjust any of the input variables, i.e. the variables uppercased beginning with an underscore (e.g. \_NUM_TRAIN_STEPS) and run:
```
docker run --gpus all -it \
  -v $(PWD):/tmp \
  -w /tmp --rm \
  -v ~/runs/:/root/runs/ \
  -v $(GOOGLE_APPLICATION_CREDENTIALS):<some_path_to_credentials_file> \
  -e GOOGLE_APPLICATION_CREDENTIALS=<some_path_to_credentials_file> \
  <mlp_project>/beam:latest \
  python <mlp_subproject>/pipelines/beam/<pipeline_file>

docker run --gpus all -it -v $PWD:/tmp -w /tmp --rm \
  <mlp_project>/beam:latest \
  python <mlp_subproject>/pipelines/beam/<pipeline_file>
```
Outputs will be written to a directory <mlp_project>/runs/ by default.

### Using kubeflow
* Ensure that the environmental variables from the installation section are set. If you used create_project.py and filled in all the relevant information when running, these variables should be located in an file called set_env.sh:
```
cd <mlp_project>
source set_env.sh
```

* Make sure you have successfully built the Docker.kubeflow image from above. Running:
```
docker images
```
should show an image with the REPOSITORY = 'gcr.io/mlp/base:latest'.

* cd to the base mlp directory, _you must run it from here!_
```
cd <mlp_project>
```

* Create or edit a kubeflow pipeline file from the pipelines directory of the project/subproject you want to run (e.g. mlp/example_project/example_subproject/pipelines/kubeflow/bigquery_to_pusher.py). Adjust any of the input variables, i.e. the variables uppercased beginning with an underscore (e.g. \_NUM_TRAIN_STEPS), or the ai_platform args if you want to change the VMs that the training process is being run on. Create the pipeline using the built in tfx tool:
```
tfx pipeline create  --endpoint $ENDPOINT --iap_client_id $CLIENT_ID --build_target_image gcr.io/$PROJECT/<mlp_project> --pipeline_path <mlp_subproject>/pipelines/<pipeline_file_name>
```

* Start a run:
```
tfx run create --pipeline_name <mlp_project>-<mlp_subproject>-<pipeline_type> --endpoint  <ai_platform_pipeline_endpoint>
```
You can get the pipeline name either from the python file used to create the pipeline or from the pipelines dashboard.

* To update an existing pipeline:
```
tfx pipeline update  --endpoint $ENDPOINT --pipeline_path <mlp_subproject>/pipelines/kubeflow/<pipeline_file_name>
```

* To delete an existing pipeline:
```
tfx pipeline delete --pipeline_name <mlp_project>-<mlp_subproject>-<pipeline_type> --endpoint  <ai_platform_pipeline_endpoint>
```
## Create project skeleton
* A simple script is provided to give you a the skeleton of a project in order to make the process of starting a project from scratch less tedious. The script is fairly straightforward. Follow the instructions for a beam install then:
```
cd mlp
docker run --gpus all -it -v $PWD:/tmp -w /tmp --rm \
  <mlp_project>/beam:latest \
  python create_project.py --help
```
