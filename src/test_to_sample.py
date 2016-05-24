# import data_loader
# test_data = data_loader.load_sample()


# import mnist_loader
# training_data, validation_data, _ = mnist_loader.load_data_wrapper()

# import network
# net = network.Network([784, 30, 10])
# net.learn(training_data, 1, 10, 3.0, test_data)

# import network2
# net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
# net.large_weight_initializer()
# net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)

# import network2

# net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost)
# net.large_weight_initializer()
# print 'begin to learn'
# net.SGD(training_data, 60, 10, 0.5, lmbda=5.0,evaluation_data=validation_data,monitor_evaluation_accuracy=True)
# print 'finish learn'

import network3
from network3 import ReLU
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data = network3.load_data_shared("../data1/elastic_rotate_expand.pkl")
mini_batch_size = 10
# expanded_training_data, _, _ = network3.load_data_shared(
#          "../data/mnist_expanded.pkl.gz")
# net = Network([
#         FullyConnectedLayer(n_in=784, n_out=100),
#         SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
# net.SGD(training_data, 60, mini_batch_size, 0.1,
#             validation_data, test_data)

net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=ReLU),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=ReLU),
        FullyConnectedLayer(
            n_in=40*4*4, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
        FullyConnectedLayer(
            n_in=1000, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
        SoftmaxLayer(n_in=1000, n_out=10, p_dropout=0.5)],
        mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.03,
            validation_data, test_data)

########################################
import numpy
import csv
# unsigned char
dt = numpy.dtype('B')


reader = csv.reader(open('../sample.csv'))
writer = csv.writer(open('../output.csv', 'w'))

lines = [l for l in reader]
new_lines = []
for filename, n0,n1,n2,n3,n4,n5,n6,n7,n8,n9 in lines:
    new_lines.append([filename])


num_test_batches = network3.size(test_data)/mini_batch_size

file_index = 0
for minibatch_index in xrange(num_test_batches):
    mini_result = net.test_mb_predictions(minibatch_index)
    for result in mini_result:
        new_lines[file_index].append(float('%0.4f' % float(result[0])))
        new_lines[file_index].append(float('%0.4f' % float(result[1])))
        new_lines[file_index].append(float('%0.4f' % float(result[2])))
        new_lines[file_index].append(float('%0.4f' % float(result[3])))
        new_lines[file_index].append(float('%0.4f' % float(result[4])))
        new_lines[file_index].append(float('%0.4f' % float(result[5])))
        new_lines[file_index].append(float('%0.4f' % float(result[6])))
        new_lines[file_index].append(float('%0.4f' % float(result[7])))
        new_lines[file_index].append(float('%0.4f' % float(result[8])))
        new_lines[file_index].append(float('%0.4f' % float(result[9])))
        file_index += 1


print net.test_mb_predictions(0)

# lines = [l for l in reader]
# new_lines = []
# for filename, n0,n1,n2,n3,n4,n5,n6,n7,n8,n9 in lines:
#     path = '../test/' + filename
#     x = numpy.fromfile(path, dt)
#     x = numpy.reshape(x, (784, 1))
#     result = net.feedforward(x)
#
#     line = [filename,
#             float(result[0][0]) ,
#             float(result[1][0]) ,
#             float(result[2][0]) ,
#             float(result[3][0]) ,
#             float(result[4][0]) ,
#             float(result[5][0]) ,
#             float(result[6][0]) ,
#             float(result[7][0]) ,
#             float(result[8][0]) ,
#             float(result[9][0])
#             ]
#     new_lines.append(line)


writer.writerows(new_lines)
