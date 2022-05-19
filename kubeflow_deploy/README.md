# Deploying a new Kubeflow Cluster
As of this writing, although kubeflow documentation is much improved, there are still numerous holes in how to set up a kubeflow cluster and google searches are of only moderate help considering the difference in everyone's environment. Having set this up several times over the course of several different versions of kubeflow I can safely say that any attempt at writing comprehensive set of instructions on how to do it is very difficult. Setting this up on one's own is surely going to require some prior experience and intuition. This README is an attempt a making the process a little smoother and faster.

## Overview
### Prerequisites
1. GCP account (since this kubeflow cluster is set up on GCP)
2. A linux machine. You can't set this up from your mac due to the bash version.
3. docker, with permissions for the current user. See [here](https://stackoverflow.com/questions/48957195/how-to-fix-docker-got-permission-denied-issue)
4. python3 (to actually test a kf pipeline), preferably 3.7-3.8. As well as the apt packages: build-essential libssl-dev libffi-dev python3.7-dev python3.8-dev  python3.7-distutils  python3.8-distutils python3-virtualenv
5. a bunch of stuff on gcp has to be [enabled]https://www.kubeflow.org/docs/distributions/gke/deploy/project-setup/
### The flow
The basic flow to get it up and running:
1. Create a project
2. Create an OAuth Client
3. Create a kubeflow management cluster
4. Create a kubeflow cluster
5. Create a gpu-node-pool (assuming you'll use gpus)
6. Adjust any IAM settings
7. Test it out

These are the general instructions one should use: https://www.kubeflow.org/docs/distributions/gke/deploy/

Most likely, one will not need to create a project (tripla-data is probably what you should use), deploy a [management cluster](https://console.cloud.google.com/kubernetes/list/overview?orgonly=true&project=tripla-data&supportedpurview=organizationId) or create an [OAuth client](https://console.cloud.google.com/apis/credentials?referrer=search&orgonly=true&project=tripla-data&supportedpurview=organizationId) since they should already exist.

If they don't already exist then follow the instructions [above](https://www.kubeflow.org/docs/distributions/gke/deploy/)

## Some customization
The main differences between this kubeflow cluster and the one described in the instructions [above](https://www.kubeflow.org/docs/distributions/gke/deploy/), aside from the issues described below are the gpus. We use nvidia-tesla-t4s while the instructions are for k80's, and we also need to add a gpu-node-pool. The modified files for this are in ml_pipelines/kubeflow_deploy/gcp-blueprints/kubeflow/common/cluster/upstream/cluster.yaml and gcp-blueprints/kubeflow/gcp-blueprints/kubeflow/containernodepool-gpu.yaml.

The environmental variables were all set in ml_pipelines/kubeflow_deploy/gcp-blueprints/kubeflow/env.sh. Don't put any sensitive info like client secrets in here. Just set those outside of a scripts.

Other than that, running through the steps described in the official documentation should do the trick. The documentation for the addition of the gpu node pool is [here](https://www.kubeflow.org/docs/distributions/gke/customizing-gke/#add-gpu-nodes-to-your-cluster)


## Some Issues with Deploying Kubeflow
1. This first command they tell you to run didn't work for me:
```
gcloud components install kubectl kustomize kpt anthoscli beta
```
I ended up just using apt-get to install most of them. Kustomize I had to grab from [here](https://kubectl.docs.kubernetes.io/installation/kustomize/binaries/):

2. kpt didn't always work for me. I tried several different install from several different sources but I couldn't get this command to work:
```
kpt pkg get https://github.com/kubeflow/gcp-blueprints.git@v1.5.0 gcp-blueprints
```
It didn't end up mattering too much. I just used the git commands instead and they worked fine:
```
git clone https://github.com/kubeflow/gcp-blueprints.git
cd gcp-blueprints
git checkout tags/v1.5.0 -b v1.5.0
```
3. There's something wrong with the gcp-blueprints v1.5.0 tag. I couldn't get the
```
bash ./pull_upstream
```
command to run, but after switching to the master branch it worked fine.
4. Make sure your version of bash is > 5.1 otherwise you'll run into errors at the install asm step
5. During the 'make apply' command to actually deploy the kubeflow cluster I got errors during the install asm step about not having a package 'jq'. I did just did
```
sudo apt-get install jq
```
and it worked fine after that.
6. It takes a long time after deploying to actually be able to see the kubeflow UI (e.g. https://kf-ml-pipelines-16.endpoints.tripla-data.cloud.goog/_/pipeline/?ns=chris#/runs/details/f82f7c86-9ba7-4641-82fa-07ccefc88ce4). maybe 30 min or so.

##Test with tfx example
After ensuring that you can get onto the kubeflow UI, you can test to see that you're actually able to run jobs. To do this you need a pipeline file which is created from a pipeline script. You can grab the files from the tfx [tutorial](https://github.com/tensorflow/tfx/tree/v1.7.1/tfx/examples/bigquery_ml). Then put taxi_utils_bqml.py in the root of the gcp bucket that is referenced in taxi_pipeline_kubeflow_gcp_bqml.py. Along with a bucket, you need to fill in various information like gcp project, some bigquery dataset, etc. Then go to some directory:
```
virtualenv venv_temp -p python3.8
source venv_temp/bin/activate
pip install tfx
pip install kfp
python taxi_pipeline_kubeflow_gcp_bqml.py
```
This last commands creates a file called chicago_taxi_pipeline_kubeflow_gcp.tar.gz. This is the pipeline file that is uploaded to kubeflow and tells it all the steps to run. Also if you get some command about pyfarmhash when trying to install tfx then make sure you installed all the python packages above in the prerequisites.

When using the kfp cli tool you need to provide a fair amount of information:
* endpoint: the url of the kf pipelines ui (e.g. https://kf-ml-pipelines-16.endpoints.tripla-data.cloud.goog/_/pipeline/?ns=chris#/runs/details/f82f7c86-9ba7-4641-82fa-07ccefc88ce4)
* namespace: it forced me to create this when I first logged on to the UI it could be either 'kubeflow' or 'chris'. It's possible to create your own I believe however I don't think we need or want to do this since we want several people to see all the jobs that are running.
* iap-client-id: this is one of the [oauth clients](https://console.cloud.google.com/apis/credentials?orgonly=true&project=tripla-data&supportedpurview=organizationId) that were created above. Don't store these anywhere that's public (e.g. github)
* other client id: Another oauth [client](https://console.cloud.google.com/apis/credentials?orgonly=true&project=tripla-data&supportedpurview=organizationId) that's for a desktop app. I had to create this, but you may be able to use the one that I created. Don't store these anywhere that's public (e.g. github)
* other client secret: They give you the secret when you create the [client](https://console.cloud.google.com/apis/credentials?orgonly=true&project=tripla-data&supportedpurview=organizationId). Don't store this anywhere that's public (e.g. github)

You can start a run by running:
```
 kfp --endpoint $ENDPOINT --iap-client-id $CLIENT_ID --namespace chris run submit --experiment-name test --package-file chicago_taxi_pipeline_kubeflow_gcp.tar.gz
```

## Some Issues with Using Kubeflow
1. Don't have permission to create experiments or runs from the command line (only pipelines work). It turns out you have to pass the other-client-id and secret when using kfp. This needs to be created using on the gcp console [OAuth page](https://console.cloud.google.com/apis/credentials?orgonly=true&project=tripla-data&supportedpurview=organizationId) of a desktop client type. The first time you try to use it, you need to authenticate in a web browser. I also ran into another issue where it wouldn't let me authenticate because my application didn't meet google's security requirements. Turns out I needed to make the [oauth consent screen](https://console.cloud.google.com/apis/credentials/consent?orgonly=true&project=tripla-data&supportedpurview=organizationId) external but in the testing status (rather than published or in prod?)
2. Any jobs that use dataflow fail with
```
apitools.base.py.exceptions.HttpForbiddenError: HttpError accessing <
https://dataflow.googleapis.com/v1b3/projects/tripla-data/locations/asia-east1/jobs?alt=json
>: response: <{'vary': 'Origin, X-Origin, Referer', 'content-type': 'application/json; charset=UTF-8', 'date': 'Mon, 02 May 2022 05:08:08 GMT', 'server': 'ESF', 'cache-control': 'private', 'x-xss-protection': '0', 'x-frame-options': 'SAMEORIGIN', 'x-content-type-options': 'nosniff', 'transfer-encoding': 'chunked', 'status': '403', 'content-length': '482', '-content-encoding': 'gzip'}>, content <{
  "error": {
    "code": 403,
    "message": "(bbc378e81298bba9): Current user cannot act as service account 165456087439-compute@developer.gserviceaccount.com. Enforced by Org Policy constraint constraints/dataflow.enforceComputeDefaultServiceAccountCheck.
https://cloud.google.com/iam/docs/service-accounts-actas
 Causes: (bbc378e81298b5b9): Current user cannot act as service account 165456087439-compute@developer.gserviceaccount.com.",
    "status": "PERMISSION_DENIED"
  }
}
```
Turns out the service account kf-ml-pipelines-16-user@tripla-data.iam.gserviceaccount.com needed to be given ServiceAccountUser permissions.
