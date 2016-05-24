import numpy

class Network():
    def __init__(self, sizes):
        self.layers = len(sizes)
        self.sizes = sizes
        self.biases = [numpy.random.randn(y, 1)
                            for y in sizes[1:]]
        self.weights = [numpy.random.randn(y, x)
                            for x,y in zip(sizes[:-1], sizes[1:])]

    def validate(self, data):
        results = [(numpy.argmax(self.feedforward(x)), y)
                        for (x, y) in data]

        return sum(int(x == y) for (x, y) in results)

    def sigmoid(self, z):
        return 1.0/(1.0 + numpy.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1-self.sigmoid(z))

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(numpy.dot(w, a) + b)
        return a

    def learn(self, train_data, runs, batch_size, eta, valitate_data):
        n = len(train_data)
        for i in xrange(runs):
            numpy.random.shuffle(train_data)
            # split to batchs
            batchs = [
                train_data[j:j+batch_size]
                for j in xrange(0, n, batch_size)
            ]
            for data in batchs:
                self.step(data, eta)

            n_validate = len(valitate_data)
            print 'run %s: %s' % (i, self.validate(valitate_data)*1.0/n_validate)

    def step(self, data, eta):
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]

        for x, y in data:
            # BP
            delta_nabla_b, delta_nabla_w = self.BP(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta/len(data))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(data))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def BP(self, x, y):
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = numpy.dot(w, activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (activations[-1] - y) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = numpy.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = numpy.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = numpy.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
