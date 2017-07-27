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

### Example of use

Download GoogleNet model code (.py) and trained variables (.npy) from: http://www.deeplearningmodel.net/
(You can also find there the imagenet-classes.txt file)

Move the files to the `models` folder.

Change first line of the model code (.py):

```
#from kaffe.tensorflow import Network
from network import Network
```

Run the converter code, pointing to the .npy file:

`python npy2ckpt models/googlenet.npy`

Test the results:

`python restorer_ckpt.py`


### Why

Amazing job here to convert models from Caffe to TensorFlow: https://github.com/ethereon/caffe-tensorflow/tree/master/examples/imagenet
. The thing is that the output is in .npy format, and I'm not very comfortable dealing with that.

Most of the code is borrowed from that repo. I changed some things to update it to Python 3 and TensorFlow 1.
