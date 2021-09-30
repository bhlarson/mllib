helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update
helm dependency update
helm upgrade cvat --install ./helm-chart -f ./helm-chart/values.yaml --namespace cv --create-namespace