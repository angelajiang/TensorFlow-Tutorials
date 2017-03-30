
# Functions and classes for loading and using the Inception model.
import inception
from inception import transfer_values_cache
import cifar10
from cifar10 import num_classes
import os
import setup

setup.setup()   # Download ciphar and inception

# Get image data
images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()
class_names = cifar10.load_class_names()

# Get inception model
model = inception.Inception()

# Calculate transfer values

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

calculate_transfer_values()



