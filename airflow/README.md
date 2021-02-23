Based on (https://github.com/airflow-helm/charts/tree/main/charts/airflow)
Based on [Setting up Airflow on a local Kubernetes cluster using Helm](https://medium.com/uncanny-recursions/setting-up-airflow-on-a-local-kubernetes-cluster-using-helm-57eb0b73dc02)

# Airflow Helm Chart

> ⚠️ this chart is the continuation of [stable/airflow](https://github.com/helm/charts/tree/master/stable/airflow), see [issue #6](https://github.com/airflow-helm/charts/issues/6) for upgrade guide

[Airflow](https://airflow.apache.org/) is a platform to programmatically author, schedule, and monitor workflows.

## Usage

### 1 - Add the Repo

```sh
helm repo add airflow-stable https://airflow-helm.github.io/charts
helm repo update
kubectl create namespace af
```

### 2 - Install the Chart

```sh
# Helm 3

kubectl apply -f af-ns.yaml

helm install airflow airflow-stable/airflow --version 7.16.0 --namespace af --values config.yaml

NAME: airflow
LAST DEPLOYED: Wed Jan 27 21:18:32 2021
NAMESPACE: af
STATUS: deployed
REVISION: 1
TEST SUITE: None
NOTES:
Congratulations. You have just deployed Apache Airflow!

1. Get the Airflow Service URL by running these commands:
   export NODE_PORT=$(kubectl get --namespace af -o jsonpath="{.spec.ports[0].nodePort}" services airflow-web)
   export NODE_IP=$(kubectl get nodes --namespace af -o jsonpath="{.items[0].status.addresses[0].address}")
   echo http://$NODE_IP:$NODE_PORT/

2. Open Airflow in your web browser
```

### 3 - Run commands in Webserver Pod

```sh
kubectl exec \
  -it \
  --namespace [NAMESPACE] \
  --container airflow-web \
  Deployment/[RELEASE_NAME]-web \
  /bin/bash

# then run commands like 
airflow create_user ...
```

---
Delete airflow
helm delete airflow -n af




