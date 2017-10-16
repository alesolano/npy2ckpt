# npy2ckpt

This is a repo to convert deep learning models from .npy format to .ckpt format.


### Set environment

Using Anaconda:

```
conda create -n npy2ckpt python=3.6
source activate npy2ckpt
pip install tensorflow
conda install -c menpo opencv3
```

### Example of use (GoogleNet)

Download GoogleNet model code (.py) and trained variables (.npy) from: http://www.deeplearningmodel.net/
(You can also find there the imagenet-classes.txt file)

Move the files to the `models` folder.

Change first line of the model code (.py):

```
#from kaffe.tensorflow import Network
from network import Network
```

Run the converter code, pointing to the .npy file:

`python npy2ckpt_GoogleNet.py models/googlenet.npy`

**Take a look and how I set the parameters** (`width`, `height`, `channels`, `input_node_name`) in npy2ckpt_GoogleNet.py.

Test the results:

`python test_GoogleNet.py`


### Example of use (OpenPose)

Following this [blog post](https://arvrjourney.com/human-pose-estimation-using-openpose-with-tensorflow-part-1-7dd4ca5c8027), use the covert.py function in the [Caffe2Tensorflow repo](https://github.com/ethereon/caffe-tensorflow) and move the output files to the `models` folder.

Change first line of the model code (.py):

```
#from kaffe.tensorflow import Network
from network import Network
```

Run the converter code, pointing to the .npy file:

`python npy2ckpt_OpenPoseNet.py models/openposenet.npy`

**Take a look and how I set the parameters** (`width`, `height`, `channels`, `input_node_name`) in npy2ckpt_OpenPoseNet.py.



### Why

Amazing job here to convert models from Caffe to TensorFlow: https://github.com/ethereon/caffe-tensorflow
. The thing is that the output is in .npy format, and I'm not very comfortable dealing with that.

Most of the code is borrowed from that repo. I changed some things to update it to Python 3 and TensorFlow 1.
