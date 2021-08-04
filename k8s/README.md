```console
kubectl create secret tls tls-ssl-cl -n cl --key ~/keys/privkey.pem --cert ~/keys/cert.pem
py k8s/setup.py
```
