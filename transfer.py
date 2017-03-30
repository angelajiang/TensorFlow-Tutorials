
# Functions and classes for loading and using the Inception model.
import inception
from inception import transfer_values_cache
import cifar10
from cifar10 import num_classes
import os
import setup

def calculate_transfer_values():
    file_path_cache_train = os.path.join(cifar10.data_path, 'inception_cifar10_train.pkl')
    file_path_cache_test = os.path.join(cifar10.data_path, 'inception_cifar10_test.pkl')

    print("Processing Inception transfer-values for training-images ...")

    # Scale images because Inception needs pixels to be between 0 and 255,
    # while the CIFAR-10 functions return pixels between 0.0 and 1.0
    images_scaled = images_train * 255.0

    # If transfer-values have already been calculated then reload them,
    # otherwise calculate them and save them to a cache-file.
    transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train,
                                                  images=images_scaled,
                                                  model=model)
    print("Processing Inception transfer-values for test-images ...")

    # Scale images because Inception needs pixels to be between 0 and 255,
    # while the CIFAR-10 functions return pixels between 0.0 and 1.0
    images_scaled = images_test * 255.0

    # If transfer-values have already been calculated then reload them,
    # otherwise calculate them and save them to a cache-file.
    transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test,
                                                 images=images_scaled,
                                                 model=model)

setup.setup()   # Download ciphar and inception

# Get image data
images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()
class_names = cifar10.load_class_names()

# Get inception model
model = inception.Inception()

calculate_transfer_values()

transfer_len = model.transfer_len
# Placeholder for inputting transfer values from Inception to new model
x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# Build new neural network
x_pretty = pt.wrap(x)
with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        fully_connected(size=1024, name='layer_fc1').\
        softmax_classifier(num_classes=num_classes, labels=y_true)

# Variable for tracking number of optimization iterations performed
global_step = tf.Variable(initial_value=0,
                          name='global_step', trainable=False)

opt_vars = [v for v in tf.trainable_variables() if v.name.startswith("layer17/")]

print("================== Opt Vars ==================")
print(opt_vars)

# http://stackoverflow.com/questions/38749120/fine-tuning-a-deep-neural-network-in-tensorflow
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step)

# Classification accuracy
y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("================== Accuracy ==================")
print(accuracy)

# Run TensorFlow
session = tf.Session()
session.run(tf.global_variables_initializer())


