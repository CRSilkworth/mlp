export MLP_VERSION='0.6.0'
export PROJECT_VERSION='0.0.0'

# Set your default project/compute zone
export PROJECT='__gcp_project__'
export ZONE='__gcp_zone__'

gcloud config set project ${PROJECT}
gcloud config set compute/zone ${ZONE}

# If using AI Platform the endpoint would be something like
# 1e9deb537390ca22-dot-asia-east1.pipelines.googleusercontent.com
# Or if just using kubeflow on gcp without AI Platform it would be like
# https://kf_deployment_name.endpoints.tripla-data.cloud.goog/pipeline
export ENDPOINT="kubeflow_endpoint"
export NAMESPACE="kubeflow"

export MLP_PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export PYTHONPATH=$PYTHONPATH:$MLP_PROJECT_DIR

source custom_env.sh
source $VENV_DIR/bin/activate
