From mllib:
```console
helm create helm/mllib
```
Edit k8s/mllib/values.yaml
From mllib:
## Helm commands
```console
helm install <name> <helm chart directory> #install helm chart from <helm chart directory> as <name>
helm install mllib helm/mllib --namespace ml --create-namespace
helm install --debug --dry-run <name> <helm chart directory> #render and print but do not install helm chart from <helm chart directory> as <name> 
helm install mllib helm/mllib --namespace ml --create-namespace --debug --dry-run
helm list
helm get manifset <name> # output generated yaml of <name>
helm uninstall <name>  # uninstall helm chart <name>
helm uninstall mllib -n mllib
```