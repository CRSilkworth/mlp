# Stores all the sensitive and custom enviromental variables
# WARNING: MAKE SURE THE FILLED OUT VERSION DOES NOT GET UPLOADED TO GITHUB


# Set the path to gcp credentials, your gcp project, and default zone.

# The json file should correspond to the credentials that either created
# the kubeflow deployment or have permissions to upload new pipelines etc.
# They are typically associated with some service account from here.
# https://console.cloud.google.com/apis/credentials
export GOOGLE_APPLICATION_CREDENTIALS="__gcp_credentials_json__"


# Set your IAP OAuth client id, secret and endpoint that's tied to your
# kubeflow deployment

# The client id and secret are typically set when you create your kubeflow
# deployment. They can be found here under OAuth 2.0 Client IDs.
# https://console.cloud.google.com/apis/credentials
export CLIENT_ID='__iap_client_id__'
export CLIENT_SECRET='__iap_client_secret__'

# The other client id and secret are typically set when you create your kubeflow
# deployment. They allow you to use the kubeflow api. They can be found here:
# under OAuth 2.0 Client IDs. https://console.cloud.google.com/apis/credentials
export OTHER_CLIENT_ID='__other_client_id__'
export OTHER_CLIENT_SECRET='__other_client_secret__'
