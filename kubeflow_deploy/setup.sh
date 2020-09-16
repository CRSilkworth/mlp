
# Must have project owner role. Set up service account from here:
# https://console.cloud.google.com/apis/credentials
export GOOGLE_APPLICATION_CREDENTIALS='<path_to_gcp_service_account_json>'

# e.g. username@gmail.com
export USER_EMAIL='<user_email_associated_with_gcp_account>'

# Set the gcp project id and zone. If you want to use gpus make sure you select
# a zone which has access to them. You can find this out by running:
# gcloud compute accelerator-types list
export PROJECT='<gcp_project_name>'
export ZONE='<zone_to_deploy_kubeflow>' # e.g. asia-east1-b

gcloud config set project ${PROJECT}
gcloud config set compute/zone ${ZONE}


export CONFIG_URI="https://raw.githubusercontent.com/kubeflow/manifests/v1.0-branch/kfdef/kfctl_gcp_iap.v1.0.2.yaml"

# Google OAuth client id and secret.
# https://www.kubeflow.org/docs/gke/deploy/oauth-setup/
export CLIENT_ID='<oauth_client_id>'
################################################################################
# ********* NOTE: YOU MUST SET CLIENT SECRET BUT DON'T PUT ON SHARE REPO *******
################################################################################
# export CLIENT_SECRET='<insert_secret_from_gcp_api_page>'

# Pick some name and directory to put evertyhing (e.g. kubeflow-deployment-1)
export KF_NAME='<deployment_name>'
export BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export KF_DIR=${BASE_DIR}/${KF_NAME}

# select gpu type (e.g. nvidia-tesla-t4, nvidia-tesla-k80, etc.)
export MACHINE_TYPE='n1-standard-8'
export GPU_TYPE='nvidia-tesla-t4'
export GPU_POOL_NAME='<some_name_for_node_pool>' # e.g. gpu-node-pool-1
export NUM_GPUS=4

# Create the deploy directory.
mkdir -p ${KF_DIR}
cd ${KF_DIR}

# Do any modifications to the default yaml file before kfctl apply.
export CONFIG_FILE='kfctl_gcp_iap.v1.0.2.yaml'
curl -L -o ${CONFIG_FILE} ${CONFIG_URI}
kfctl build -V -f ${CONFIG_FILE}

# Replace all the variables in the cluster-kubeflow file
cp ../cluster-kubeflow.yaml gcp_config/
sed -i s/'<machine_type>'/${MACHINE_TYPE}/ gcp_config/cluster-kubeflow.yaml
sed -i s/'<gpu_type>'/${GPU_TYPE}/ gcp_config/cluster-kubeflow.yaml
sed -i s/'<project>'/${PROJECT}/ gcp_config/cluster-kubeflow.yaml
sed -i s/'<user_email>'/${USER_EMAIL}/ gcp_config/cluster-kubeflow.yaml
sed -i s/'<zone>'/${ZONE}/ gcp_config/cluster-kubeflow.yaml

# Replace all the variables in the iam_bindings file
cp ../iam_bindings.yaml gcp_config/
sed -i s/'<kf_name>'/${KF_NAME}/ gcp_config/iam_bindings.yaml
sed -i s/'<project>'/${PROJECT}/ gcp_config/iam_bindings.yaml
sed -i s/'<user_email>'/${USER_EMAIL}/ gcp_config/iam_bindings.yaml

# Create the kubeflow deployment
kfctl apply -V -f ${CONFIG_FILE}

# Add gpu node pool.
gcloud container node-pools create ${GPU_POOL_NAME} \
--accelerator=type=${GPU_TYPE},count=${NUM_GPUS} \
--zone=${ZONE} --cluster=${KF_NAME} \
--num-nodes=1 --machine-type=n1-standard-4 --min-nodes=1 --max-nodes=5 --enable-autoscaling
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

# kubeflow endpoint will be http://${KF_NAME}.endpoints.${PROJECT}.cloud.goog
# and should show up in 15min - 30min
