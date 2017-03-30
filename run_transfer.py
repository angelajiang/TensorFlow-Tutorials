import transfer
import cifar10
transfer.setup()
images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()
transfer.plot_images(images=images_train[0:10], cls_true=cls_train[0:10])

