import numpy as np
import matplotlib.image as mpimg
from keras.utils import np_utils

total_classes = 4

def data_split(data):
    # Converting values into 4 ranges for less labels
    for i in range(0, len(data)):
        if data[i][5] >= 25:
            data[i][5] = 3
        elif data[i][5] < 25 & data[i][5] >= 20:
            data[i][5] = 2
        elif data[i][5] < 20 & data[i][5] >= 13:
            data[i][5] = 1
        else:
            data[i][5] = 0

    # Data split into training, testing, validation
    images_data = [[] for _ in range(len(data))]
    labels_data = []

    for k in range(len(data)):
        images_data[k].append(data[k][0][:])
        labels_data.append(data[k][5])

    #20% testing split
    test_images = np.array(images_data[:int(len(data) * 0.2)])
    test_labels = np.array(labels_data[:int(len(data) * 0.2)])
    randy1 = np.random.permutation(len(test_images))
    test_images = test_images[randy1]
    test_labels = test_labels[randy1]

    #20% validation split
    valid_images = np.array(images_data[int(len(data) * 0.2):int(len(data) * 0.4)])
    valid_labels = np.array(labels_data[int(len(data) * 0.2):2 * int(len(data) * 0.4)])
    randy2 = np.random.permutation(len(valid_images))
    valid_images = valid_images[randy2]
    valid_labels = valid_labels[randy2]

    #60% training split
    train_images = np.array(images_data[int(len(data) * 0.4):])
    train_labels = np.array(labels_data[int(len(data) * 0.4):])
    randy3 = np.random.permutation(len(train_images))
    train_images = train_images[randy3]
    train_labels = train_labels[randy3]

    n = len(train_labels)
    n1 = len(test_labels)
    n2 = len(valid_labels)
    train_labels = train_labels.reshape(n, 1)
    test_labels = test_labels.reshape(n1, 1)
    valid_labels = valid_labels.reshape(n2, 1)


    train_labels = np_utils.to_categorical(train_labels - 1, total_classes)
    test_labels = np_utils.to_categorical(test_labels - 1, total_classes)
    valid_labels = np_utils.to_categorical(valid_labels - 1, total_classes)

    #Unpack the 3 specific slices
    train_images = [np.reshape(np.array(train_images[:,:,17,:,:]), (n, 256,256,1)), np.reshape(np.array(train_images[:,:,18,:,:]), (n,256,256,1)), np.reshape(np.array(train_images[:,:,19,:,:]),(n,256,256,1) )]
    test_images = [np.reshape(np.array(test_images[:,:,17,:,:]),(n1,256,256,1)), np.reshape(np.array(test_images[:,:,18,:,:]), (n1,256,256,1)), np.reshape(np.array(test_images[:,:,19,:,:]),(n1,256,256,1))]
    valid_images = [np.reshape(np.array(valid_images[:,:,17,:,:]), (n2,256,256,1)), np.reshape(np.array(valid_images[:,:,18,:,:]), (n2,256,256,1)), np.reshape(np.array(valid_images[:,:,19,:,:]),(n2,256,256,1))]


    #mpimg.imsave('before.png', train_images[15, 0, 1, :, :])

    #train_images = train_images.reshape(n, 3, 256, 256, 1)
    #test_images = test_images.reshape(n1, 3, 256, 256, 1)
    #valid_images = valid_images.reshape(n2, 3, 256, 256, 1)

    #mpimg.imsave('after.png', train_images[15, 1, :, :, 0])

    return test_images, test_labels, valid_images, valid_labels, train_images, train_labels