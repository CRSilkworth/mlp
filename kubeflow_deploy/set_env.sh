# 1. Edit <placeholders>.
# 2. Other env vars are configurable, but with default values set below.

# The KF_PROJECT env var contains the Google Cloud project ID where Kubeflow
# cluster will be deployed to.
export KF_PROJECT='__gcp_project__'
export PROJECT="${KF_PROJECT}"
# You can get your project number by running this command
# (replace ${KF_PROJECT} with the actual project ID):
# gcloud projects describe --format='value(projectNumber)' "${KF_PROJECT}"
export KF_PROJECT_NUMBER=__project_number__
# ADMIN_EMAIL env var is the Kubeflow admin's email address, it should be
# consistent with login email on Google Cloud.
# Example: admin@gmail.com
export ADMIN_EMAIL='__email__'
# The MGMT_NAME env var contains the name of your management cluster created in
# management cluster setup:
# https://www.kubeflow.org/docs/distributions/gke/deploy/management-setup/
export MGMT_PROJECT='__gcp_project__'
export MGMT_NAME='__kf_manager_name__'
# The MGMTCTXT env var contains a kubectl context that connects to the management
# cluster. By default, management cluster setup creates a context named
# ${MGMT_NAME} for you.
export MGMTCTXT="${MGMT_NAME}"

######################
# NOTICE: The following env vars have default values, but they are also configurable.
######################

# KF_NAME env var is name of your new Kubeflow cluster.
# It should satisfy the following prerequisites:
# * be unique within your project, e.g. if you already deployed cluster with the
# name "kubeflow", use a different name when deploying another Kubeflow cluster.
# * start with a lowercase letter
# * only contain lowercase letters, numbers and "-"s (hyphens)
# * end with a number or a letter
# * contain no more than 24 characters
export KF_NAME='__kubeflow_name__'

# This is the url created when launch the kubeflow cluster. Should be something like https://$KF_NAME.$KF_PROJECT.cloud.goog/_/pipeline/
export ENDPOINT='__endpoint__'
# Default values for managed storage used by Kubeflow Pipelines (KFP), you can
# override as you like.
# The CloudSQL instance and Cloud Storage bucket instance are created during
# deployment, so you should make sure their names are not used before.
export CLOUDSQL_NAME="${KF_NAME}-kfp"
# Note, Cloud Storage bucket name needs to be globally unique across projects.
# So we default to a name related to ${KF_PROJECT}.
export BUCKET_NAME="${KF_PROJECT}-kfp"
# LOCATION can either be a zone or a region, that determines whether the deployed
# Kubeflow cluster is a zonal cluster or a regional cluster.
# Specify LOCATION as a region like the following line to create a regional Kubeflow cluster.
# export LOCATION=us-central1
export LOCATION='__location__'
export ZONE='__zone__'
export REGION='__region__'
# Anthos Service Mesh version label
export ASM_LABEL=asm-1104-6


source custom_env.sh
