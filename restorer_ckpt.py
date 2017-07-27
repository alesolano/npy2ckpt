import tensorflow as tf
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ignore debugging logs

### GRAPH ###

tf.reset_default_graph()

saver = tf.train.import_meta_graph('/tmp/googlenet_model.ckpt.meta')

# Check the names of the tensors in your graph
#for tensor in tf.get_default_graph().as_graph_def().node:
#    print(tensor.name)

inputs = tf.get_default_graph().get_tensor_by_name('inputs:0')
prob = tf.get_default_graph().get_tensor_by_name('prob:0')
print('The input placeholder is expecting an array of shape {} and type {}'.format(inputs.shape, inputs.dtype))


### CLASSES ###
with open('./models/imagenet-classes.txt') as f:
    classes = f.read().splitlines()


### IMAGE PREPROCESSING ###
img = cv2.imread('./tests/daisy_test.jpg')
prep_img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_CUBIC)
prep_img = prep_img.reshape([1, 224, 224, 3])
print("The input image has been resized from {} to {}".format(img.shape, prep_img.shape))

assert list(prep_img.shape[1:]) == inputs.get_shape().as_list()[1:], \
    'Dimensions of the input image and the placeholder should match'
print('Dimensions match!')


### SESSION ###
with tf.Session() as sess:
    saver.restore(sess, '/tmp/googlenet_model.ckpt')
    
    prob_values = sess.run(prob, feed_dict={
            inputs: prep_img
        })
    

### RESULTS ###

pred_idx = prob_values[0].argmax()
pred_class = classes[pred_idx]
pred_certain = round(100*prob_values[0][pred_idx], 2) # two decimals

print("\nI'm {}% sure that this is a {}.".format(pred_certain, pred_class))


