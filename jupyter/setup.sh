# Suggested values: advanced users of Kubernetes and Helm should feel
# free to use different values.
helm install jhub --namespace jhub --version=0.10.6 --values config.yaml jupyterhub/jupyterhub  --dry-run