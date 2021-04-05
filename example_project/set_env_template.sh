# Set the path to gcp credentials, your gcp project, and default zone.

# The json file should correspond to the credentials that either created
# the kubeflow deployment or have permissions to upload new pipelines etc.
# They are typically associated with some service account from here.
# https://console.cloud.google.com/apis/credentials
export GOOGLE_APPLICATION_CREDENTIALS="__gcp_credentials_json__"

# Set your default project/compute zone
export PROJECT='__gcp_project__'
export ZONE='__gcp_zone__'

gcloud config set project ${PROJECT}
gcloud config set compute/zone ${ZONE}

# Set your IAP OAuth client id, secret and endpoint that's tied to your
# kubeflow deployment

# The client id and secret are typically set when you create your kubeflow
# deployment. They can be found here under OAuth 2.0 Client IDs.
# https://console.cloud.google.com/apis/credentials
export CLIENT_ID='__iap_client_id__'
export CLIENT_SECRET='__iap_client_secret__'

# If using AI Platform the endpoint would be something like
# 1e9deb537390ca22-dot-asia-east1.pipelines.googleusercontent.com
# Or if just using kubeflow on gcp without AI Platform it would be like
# https://kf_deployment_name.endpoints.tripla-data.cloud.goog/pipeline
export ENDPOINT="kubeflow_endpoint"
export NAMESPACE="kubeflow"

export MLP_PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export PYTHONPATH=$PYTHONPATH:$MLP_PROJECT_DIR

source $MLP_PROJECT_DIR/venv/bin/activate
