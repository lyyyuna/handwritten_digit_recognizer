# f = gzip.open(filename, 'rb')
# training_data, validation_data, test_data = cPickle.load(f)
# f.close()
from __future__ import print_function
import data_loader

tmp_data = data_loader.load_test()
training_data = tmp_data

print (training_data[0][0])
print (training_data[1][1])


#### Libraries

# Standard library
import cPickle
import gzip
import os.path
import random

# Third-party libraries
import numpy as np

print("Expanding the MNIST training set")

if False:
    print("The expanded training set already exists.  Exiting.")
else:
    # f = gzip.open("../data/mnist.pkl.gz", 'rb')
    # training_data, validation_data, test_data = cPickle.load(f)
    # f.close()
    expanded_training_pairs = []
    j = 0 # counter
    for x, y in zip(training_data[0], training_data[1]):
        expanded_training_pairs.append((x, y))
        image = np.reshape(x, (-1, 28))
        j += 1
        if j % 1000 == 0: print("Expanding image number", j)
        # iterate over data telling us the details of how to
        # do the displacement
        for d, axis, index_position, index in [
                (1,  0, "first", 0),
                (-1, 0, "first", 27),
                (1,  1, "last",  0),
                (-1, 1, "last",  27)]:
            new_img = np.roll(image, d, axis)
            if index_position == "first":
                new_img[index, :] = np.zeros(28)
            else:
                new_img[:, index] = np.zeros(28)
            expanded_training_pairs.append((np.reshape(new_img, 784), y))
    random.shuffle(expanded_training_pairs)
    expanded_training_data = [list(d) for d in zip(*expanded_training_pairs)]
    print("Saving expanded data. This may take a few minutes.")
    f = gzip.open("../data/expanded02.pkl.gz", "w")
    cPickle.dump(expanded_training_data, f)
    f.close()
