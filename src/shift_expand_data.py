import numpy as np
from scipy import ndimage
from scipy import misc
import cv2
import numpy
import csv
import cPickle
import gzip
import random

dt = numpy.dtype('B')
training_data = []

f = gzip.open("../data1/rotate_expand.pkl.gz", 'rb')
training_data = cPickle.load(f)
f.close()

print 'begin to shit'

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
f = gzip.open("../data1/shift_rotate_expand1.pkl.gz", "wb")
cPickle.dump(expanded_training_data, f)
f.close()
# x = numpy.fromfile('../train/0a31cebb', dt)
# image = np.reshape(x, (-1, 28))
# ro_image = ndimage.rotate(image, 20)
#
# ros_image = misc.imresize(ro_image, (28,28))
# mask = np.where(ros_image < 50)
# for i in range(0, len(mask[0])):
#     ros_image[mask[0][i], mask[1][i]] = 0

# ss = ''
# for i in image:
#     for j in i:
#         if j > 0:
#             ss += ' '
#         else:
#             ss += '#'
#     ss += '\n'
# print ss
# print
#
# ss = ''
# for i in ros_image:
#     for j in i:
#         if j > 0:
#             ss += ' '
#         else:
#             ss += '#'
#     ss += '\n'
# print ss
# print
