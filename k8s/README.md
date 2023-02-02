[microk8s snap instll](https://microk8s.io/docs/setting-snap-channel)

```console
sudo snap remove microk8s --purge
sudo snap info microk8s
sudo snap install microk8s --channel=1.25/stable --classic
sudo microk8s.enable dns gpu helm3 ingress registry storage metallb:10.64.140.43-10.64.140.143
sudo usermod -a -G microk8s $USER
sudo snap install kubectl --classic
cd $HOME
mkdir .kube
cd .kube
microk8s config > config
```

```console
kubectl create secret tls tls-ssl-cl -n cl --key ~/keys/privkey.pem --cert ~/keys/cert.pem
py k8s/setup.py
```
