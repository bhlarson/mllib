# Machine Learning Jupyter Notebooks
1. Build jupyter docker container
Docker build training/test image 
```console
$ djb
```
2. Run container
```console
$ dj
```
3. <ctl+click> on console output url to launch the browser link and token

# Notes:
- 1. Development docker image
   > docker run --device=/dev/video0:/dev/video0 --gpus '"device=0"' -it --rm -v "$(pwd):/app" -v "/store:/store" -p 8889:8888/tcp -p 8009:8008/tcp -p 5001:5000/tcp -p 3001:3000 ml:latest

# To Do:
* fcn.py line 13 - Valid model
