[Quick Start](https://argoproj.github.io/argo-workflows/quick-start/)

```console
kubectl create ns argo
kubectl apply -n argo -f https://raw.githubusercontent.com/argoproj/argo-workflows/stable/manifests/quick-start-postgres.yaml
kubectl -n argo port-forward deployment/argo-server 2746:2746
```

[Install Argo CLI](https://github.com/argoproj/argo-workflows/releases)
```console
# Download the binary
curl -sLO https://github.com/argoproj/argo/releases/download/v2.12.11/argo-linux-amd64.gz

# Unzip
gunzip argo-linux-amd64.gz

# Make binary executable
chmod +x argo-linux-amd64

# Move binary to path
sudo mv ./argo-linux-amd64 /usr/local/bin/argo

# Test installation
argo version
```

Finally, submit an example workflow:
```console
argo submit -n argo --watch https://raw.githubusercontent.com/argoproj/argo-workflows/master/examples/hello-world.yaml
argo list -n argo
argo get -n argo @latest
argo logs -n argo @latest
```