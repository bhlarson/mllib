# mllib

>  --trainingset 2020310Lit --initialmodel 20200302Lit02 --debug True

Enable SSH : https://stackoverflow.com/questions/59325078/cannot-connect-to-coral-dev-board-after-updating-to-4-0-mdt-shell-does-not-work#
1) use ssh-keygen to create private and pub key files.
2) append (or copy) the pubkey file to target /home/mendel/.ssh/authorized_keys
3) copy the private key file to ~/.config/mdt/keys/mdt.key
4) add to local .ssh/config to something like this:

Host tpu
         IdentityFile ~/.config/mdt/keys/mdt.key
         IdentitiesOnly=yes

Docker build training/test image
> docker build --pull --rm -f "dockerfile" -t ml:latest context
> docker run --gpus '"device=0"' -it --rm -v "$(pwd):/app" -v "/store/Datasets:/store/Datasets" -p 6006:6006/tcp -p 3000:3000 mllib:latest

Jupyter notebook development:
docker pull jupyter/tensorflow-notebook

<ol type="1">
    <li>System Setup</li>
        <ol type="a">
            <li>Ubuntu</li>
                <ol type="i">
                    <li>os</li>
                    <li>ssh</li>
                    <li>file system</li>  
                </ol>
            <li>MicroK8s or Kubernetes</li>
                <ol type="i">
                    <li>docker</li>
                    <li>snap</li>
                    <li>microk8s</li>  
                </ol>
            <li><a href=https://zero-to-jupyterhub.readthedocs.io/en/latest/setup-jupyterhub/index.html>Jupyter Hub</a> </li>
            <li> <a href=https://min.io>MINIO</a> data storage
            <li><a href=https://github.com/opencv/cvat>CVAT</a></li>  
        </ol>
    <li>Collect images for training, test and validation</li>
    <li>Annotation images using</li>
    <li>Convert annotations to TFRecord training set</li>
        <ol type="a">
            <li></li>
        </ol>
    <li>Select segmentation model</li>
    <li>Select inference hardware</li>
    <li>Select inference server</li>
    <li>Train model</li>
    <li>Verify trained model</li>
    <li>Optimize model for inference hardware</li>
    <li>Deploy to inference hardware</li>
    <li>Validate inference results</li>
    <li></li>
    <li></li>
        <ol type="a">
            <li></li>
            <li></li>
            <li></li>
        </ol>
    <li></li>
    <li></li>
</ol>

# Notes:
- 1. Development docker image
   > docker run --device=/dev/video0:/dev/video0 --gpus '"device=0"' -it --rm -v "$(pwd):/app" -v "/store:/store" -p 8889:8888/tcp -p 8009:8008/tcp -p 5001:5000/tcp -p 3001:3000 ml:latest

# To Do:
* segmentation.py line 187 - Plot results.  pred_mask.shape (8, 224, 224, 19) - need max pooling argmax to convert to segmentation results (224,224) array

# jtop diagnostics: https://github.com/rbonghi/jetson_stats