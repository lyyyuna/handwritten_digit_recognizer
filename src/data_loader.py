import csv
import numpy
import gzip
import cPickle

def vectorized_result(i):
    o = numpy.zeros([10, 1])
    o[i] = 1.0
    return o

def load():
    # unsigned char
    dt = numpy.dtype('B')
    training_data = []

    reader = csv.reader(open('../train.csv'))
    for filename, digit in reader:
        path = '../train/' + filename
        x = numpy.fromfile(path, dt)
        digit = int(digit)
        training_data.append((numpy.reshape(x, (784, 1)), digit))


    # print len(training_data[:8000] + training_data[8000:])
    # train_data = training_data[:8000]
    # valitate_data = training_data[8000:]
    #
    # trainss_data = []
    # for (x,y) in train_data:
    #     y = vectorized_result(y)
    #     trainss_data.append((x,y))
    #
    # return trainss_data, valitate_data
    return [], training_data

# f = open('../train/00cc337c', 'rb')
# ss = f.read()
# tt = ''
# for i in range(1,len(ss)):
#     ch = ss[i]
#     if ord(ch) > 10:
#         tt += ' '
#     else:
#         tt += '#'
#     if i % 28 ==0:
#         print tt
#         tt = ''
# print



def load_sample():
    # unsigned char
    dt = numpy.dtype('B')
    test_data = []

    reader = csv.reader(open('../sample.csv'))
    index = 0
    for filename, n0,n1,n2,n3,n4,n5,n6,n7,n8,n9 in reader:
        path = '../test/' + filename
        x = numpy.fromfile(path, dt)
        x = x / 255.0
        digit = 1 # no use
        test_data.append(x)
        index += 1
        if index%1000 == 0:
            print index

    return [test_data, [1,1]]

def load_test():
    # unsigned char
    dt = numpy.dtype('B')
    test_data = []
    test_result = []

    reader = csv.reader(open('../train.csv'))
    index = 0
    for filename, digit in reader:
        path = '../train/' + filename
        x = numpy.fromfile(path, dt)
        x = x / 255.0
        test_result.append(int(digit))
        test_data.append(x)
        index += 1
        if index%1000 == 0:
            print index

    return [test_data, test_result]


def load_train(filename):
    f = gzip.open(filename, 'rb')
    tmp_data = cPickle.load(f)
    f.close()
    training_data = []
    training_result = []
    index = 0
    print 'loading'
    for x, y in zip(tmp_data[0], tmp_data[1]):
        x = x/255.0
        training_data.append(x)
        training_result.append(y)
        index += 1

        if index%1000 == 0:
            print index

    return [training_data, training_result]

def load_train1(filename):
    f = open(filename, 'rb')
    tmp_data = cPickle.load(f)
    f.close()
    training_data = []
    training_result = []
    index = 0
    print 'loading'
    for x, y in zip(tmp_data[0], tmp_data[1]):
        x = x/255.0
        training_data.append(x)
        training_result.append(y)
        index += 1

        if index%1000 == 0:
            print index

    return [training_data, training_result]


if __name__ == '__main__':
    load_train("../data/shift_rotate_expand.pkl.gz")
