On microk8s, need to set containerRuntimeExecutor to pns

```console
kubectl create namespace argo
wget https://raw.githubusercontent.com/argoproj/argo-workflows/stable/manifests/quick-start-minimal.yaml

kubectl apply -n argo -f quick-start-minimal.yaml
kubectl get all -n argo
```
Install the argo cli application
```console
curl -sLO https://github.com/argoproj/argo-workflows/releases/download/v3.1.13/argo-linux-amd64.gz
gunzip argo-linux-amd64.gz
chmod +x argo-linux-amd64
sudo mv ./argo-linux-amd64 /usr/local/bin/argo
argo version
```

Port forward 
```console
kubectl -n argo port-forward deployment/argo-server 2746:2746 --address='0.0.0.0'
```


To get an authentication token, use the console interface and then paste the full output into the token login "Bearer <token value>"
```console
 argo auth token
```

