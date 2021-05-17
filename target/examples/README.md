Notebook descriptions:

Run with:
nvidia-docker run --net=host  -v $(pwd):/workspace -it nvcr.io/nvidia/tensorflow:20.06-tf2-py3 jupyter lab  --notebook-dir=/workspace --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.base_url=${NB_PREFIX} 

Notebooks:
0. Official TF-TRT: Demonstrates a way of using TF-TRT directly without the wrapper - this is what is happening with the wrapper behind the scenes
1. Wrapper Basic: usage of TF-TRT through the ModelOptimizer helper
2. Wrapper Class: Uses ModelOptimizer to optimize every type of model in tf.keras.applications
3. Wrapper Segment: Uses ModelOptimizer to optimize the Keras U-Net tutorial script
4. Wrapper Detect: Uses ModelOptimizer to optimize the Keras RetinaNet tutorial script
