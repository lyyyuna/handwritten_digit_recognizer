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

reader = csv.reader(open('../train.csv'))
f = gzip.open("../data1/rotate_expand.pkl.gz", "w")

print 'begin to rotate'
index = 0
for filename, digit in reader:
    path = '../train/' + filename
    x = numpy.fromfile(path, dt)
    image = np.reshape(x, (-1, 28))
    ro_image = ndimage.rotate(image, 10)


    ros_image = misc.imresize(ro_image, (28,28))
    mask = np.where(ros_image < 80)
    for i in range(0, len(mask[0])):
        ros_image[mask[0][i], mask[1][i]] = 0


    ro_image = ndimage.rotate(image, -10)
    rosright_image = misc.imresize(ro_image, (28,28))
    mask = np.where(rosright_image < 80)
    for i in range(0, len(mask[0])):
        rosright_image[mask[0][i], mask[1][i]] = 0

    digit = int(digit)
    #x = x
    training_data.append((numpy.reshape(x, 784), digit))
    #ros_image = ros_image
    training_data.append((numpy.reshape(ros_image, 784), digit))
    #rosright_image = rosright_image
    training_data.append((numpy.reshape(rosright_image, 784), digit))


    index += 1
    if (index % 10 == 0):
        print 'number', index

random.shuffle(training_data)
training_data = [list(d) for d in zip(*training_data)]
print ('begin to dump ....')
cPickle.dump(training_data, f)
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
