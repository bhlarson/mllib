Based on https://zero-to-jupyterhub.readthedocs.io/en/latest/jupyterhub/installation.html

```sh
openssl rand -hex 32
nano config.yaml
```

proxy:
  secretToken: "<RANDOM_HEX>"

```sh
microk8s enable helm3
helm repo add jupyterhub https://jupyterhub.github.io/helm-chart/
kubectl create namespace jhub
helm install jhub jupyterhub/jupyterhub --values config.yaml --namespace jhub
kc get services -n jhub
```

blarson@ipc:~/zzz$ kubectl get service --namespace jhub
NAME           TYPE           CLUSTER-IP       EXTERNAL-IP    PORT(S)        AGE
hub            ClusterIP      10.152.183.194   <none>         8081/TCP       33m
proxy-api      ClusterIP      10.152.183.181   <none>         8001/TCP       33m
proxy-public   LoadBalancer   10.152.183.37    10.64.140.45   80:32034/TCP   33m

From browser in the Kubernetes cluster:
http://10.64.140.45/

From browser accessible to cluster:
http://ipc.larson.myds.me:32034/

Customize:
RELEASE=jhub
NAMESPACE=jhub

helm upgrade --cleanup-on-fail jhub jupyterhub/jupyterhub --namespace jhub --values config.yaml



