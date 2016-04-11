"""
2015-09-21 - Created by Ehsan Hosseini-Asl

Deeply-Supervised Siamese Net

"""

__author__ = 'ehsanh'

class ConvolutionLayer(object):
    ACT_TANH = 't'
    ACT_SIGMOID = 's'
    ACT_ReLu = 'r'
    ACT_SoftPlus = 'p'

    def __init__(self, rng, input, filter_shape, poolsize=(2,2), stride=None, if_pool=False, act=None, share_with=None,
                 tied=None, border_mode='valid'):
        self.input = input

        if share_with:
            self.W = share_with.W
            self.b = share_with.b

            self.W_delta = share_with.W_delta
            self.b_delta = share_with.b_delta

        elif tied:
            self.W = tied.W.dimshuffle(1,0,2,3)
            self.b = tied.b

            self.W_delta = tied.W_delta.dimshuffle(1,0,2,3)
            self.b_delta = tied.b_delta

        else:
            fan_in = np.prod(filter_shape[1:])
            poolsize_size = np.prod(poolsize) if poolsize else 1
            fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / poolsize_size)
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(
                np.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)

            self.W_delta = theano.shared(
                np.zeros(filter_shape, dtype=theano.config.floatX),
                borrow=True
            )

            self.b_delta = theano.shared(value=b_values, borrow=True)

        conv_out = nnet.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            border_mode=border_mode)

        #if poolsize:
        if if_pool:
            pooled_out = downsample.max_pool_2d(
                input=conv_out,
                ds=poolsize,
                st=stride,
                ignore_border=True)
            tmp = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        else:
            tmp = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        if act == ConvolutionLayer.ACT_TANH:
            self.output = T.tanh(tmp)
        elif act == ConvolutionLayer.ACT_SIGMOID:
            self.output = nnet.sigmoid(tmp)
        elif act == ConvolutionLayer.ACT_ReLu:
            self.output = tmp * (tmp>0)
        elif act == ConvolutionLayer.ACT_SoftPlus:
            self.output = T.log2(1+T.exp(tmp))
        else:
            self.output = tmp

        # store parameters of this layer
        self.params = [self.W, self.b]

        self.deltas = [self.W_delta, self.b_delta]

    def get_state(self):
        return self.W.get_value(), self.b.get_value()

    def set_state(self, state):
        self.W.set_value(state[0], borrow=True)
        self.b.set_value(state[1], borrow=True)

class HiddenLayer(object):

    def __init__(self, rng, input, n_in, n_out, share_with=None, activation=None):

        self.input = input

        if share_with:
            self.W = share_with.W
            self.b = share_with.b

            self.W_delta = share_with.W_delta
            self.b_delta = share_with.b_delta
        else:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == nnet.sigmoid:
                W_values *= 4

            self.W = theano.shared(value=W_values, name='W', borrow=True)

            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name='b', borrow=True)

            self.W_delta = theano.shared(
                    np.zeros((n_in, n_out), dtype=theano.config.floatX),
                    borrow=True
                )

            self.b_delta = theano.shared(value=b_values, borrow=True)

        self.params = [self.W, self.b]

        self.deltas = [self.W_delta, self.b_delta]

        lin_output = T.dot(self.input, self.W) + self.b

        if activation == 'tanh':
            self.output = T.tanh(lin_output)
        elif activation == 'sigmoid':
            self.output = nnet.sigmoid(lin_output)
        elif activation == 'relu':
            self.output = T.maximum(lin_output, 0)
        else:
            self.output = lin_output

    def get_state(self):
        return self.W.get_value(), self.b.get_value()

    def set_state(self, state):
        self.W.set_value(state[0], borrow=True)
        self.b.set_value(state[1], borrow=True)

class softmaxLayer(object):
    def __init__(self, input, n_in, n_out):

        self.W = theano.shared(
            value=np.zeros(
                (n_in,n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )

        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.W_delta = theano.shared(
                np.zeros((n_in,n_out), dtype=theano.config.floatX),
                borrow=True
            )

        self.b_delta = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX),
            name='b',
            borrow=True)

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

        self.deltas = [self.W_delta, self.b_delta]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def get_state(self):
        return self.W.get_value(), self.b.get_value()

    def set_state(self, state):
        self.W.set_value(state[0], borrow=True)
        self.b.set_value(state[1], borrow=True)

class siamese(object):
    def __init__(self, image_shape):
        rng = np.random.RandomState(None)
        shreds = T.ftensor4('shreds')
        target = T.ivector('targets')
        if shreds.ndim == 3:
            shreds = shreds.dimshuffle(0, 'x', 1, 2)
        num_panes = [20, 50]
        ip_output_dim = [500, 100, 2]
        pool_size=(2, 2)

        conv1 = ConvolutionLayer(rng,
                                 input=shreds[:,0,:,:].dimshuffle(0,'x',1,2),
                                 filter_shape=(num_panes[0], 1, 5, 5),
                                 poolsize=pool_size,
                                 act=ConvolutionLayer.ACT_SIGMOID,
                                 if_pool=True)
        featuremap_shape_conv1 = (image_shape - np.array([4,4]))/2
        conv2 = ConvolutionLayer(rng,
                                 input=conv1.output,
                                 filter_shape=(num_panes[1], num_panes[0], 5, 5),
                                 poolsize=pool_size,
                                 act=ConvolutionLayer.ACT_SIGMOID,
                                 if_pool=True)

        featuremap_shape_conv2 = (featuremap_shape_conv1 - np.array([4,4]))/2
        ip1_input = conv2.output.flatten(2)
        ip1_input_dim = np.prod(featuremap_shape_conv2)*num_panes[1]
        ip1 = HiddenLayer(rng,
                          ip1_input,
                          ip1_input_dim,
                          ip_output_dim[0],
                          activation='relu')

        ip2 = HiddenLayer(rng,
                          ip1.output,
                          ip_output_dim[0],
                          ip_output_dim[1])

        feat = HiddenLayer(rng,
                           ip2.output,
                           ip_output_dim[1],
                           ip_output_dim[2])

        #parallel network
        conv1_p = ConvolutionLayer(rng,
                                   input=shreds[:,1,:,:].dimshuffle(0,'x',1,2),
                                   filter_shape=(num_panes[0], 1, 5, 5),
                                   poolsize=pool_size,
                                   share_with=conv1,
                                   act=ConvolutionLayer.ACT_SIGMOID,
                                   if_pool=True)

        conv2_p = ConvolutionLayer(rng,
                                   input=conv1_p.output,
                                   filter_shape=(num_panes[1], num_panes[0], 5, 5),
                                   poolsize=pool_size,
                                   share_with=conv2,
                                   act=ConvolutionLayer.ACT_SIGMOID,
                                   if_pool=True)

        ip1_p_input = conv2_p.output.flatten(2)

        ip1_p = HiddenLayer(rng,
                            ip1_p_input,
                            ip1_input_dim,
                            ip_output_dim[0],
                            share_with=ip1,
                            activation='relu')

        ip2_p = HiddenLayer(rng,
                            ip1_p.output,
                            ip_output_dim[0],
                            ip_output_dim[1],
                            share_with=ip2)

        feat_p = HiddenLayer(rng,
                             ip2_p.output,
                             ip_output_dim[1],
                             ip_output_dim[2],
                             share_with=feat)

        Dw1 = ((conv1.output-conv1_p.output)**2).mean(axis=(1, 2, 3))
        Dw2 = ((conv2.output-conv2_p.output)**2).mean(axis=(1, 2, 3))
        Dw3 = ((ip1.output-ip1_p.output)**2).mean(axis=1)
        Dw4 = ((ip2.output-ip2_p.output)**2).mean(axis=1)
        Dw5 = ((feat.output-feat_p.output)**2).mean(axis=1)
        alpha = 0.1

        L2 = (
            ((conv1.W ** 2)).sum()
            + (conv2.W ** 2).sum()
            + (ip1.W ** 2).sum()
            + (ip2.W ** 2).sum()
            + (feat.W ** 2).sum()
        )

        L2_reg = 0.00

        m=1.25

        cost1 = T.mean((1-target)*Dw1 + target*T.maximum(0, (m-Dw1)))
        cost2 = T.mean((1-target)*Dw2 + target*T.maximum(0, (m-Dw2)))
        cost3 = T.mean((1-target)*Dw3 + target*T.maximum(0, (m-Dw3)))
        cost4 = T.mean((1-target)*Dw4 + target*T.maximum(0, (m-Dw4)))
        cost5 = T.mean((1-target)*Dw5 + target*T.maximum(0, (m-Dw5)))
        cost = T.mean((alpha**4)*cost1+ (alpha**3)*cost2+ (alpha**2)*cost3+ alpha*cost4 + cost5)+ L2_reg * L2

        self.layers = [conv1, conv2, ip1, ip2, feat]
        self.params = sum([l.params for l in self.layers[-3:]], [])
        self.deltas = sum([l.deltas for l in self.layers[-3:]], [])

        self.grads = T.grad(cost, self.params)
        updates = adadelta_updates(self.params, self.grads, rho=0.95, eps=1e-6)

        self.train = theano.function(
            [shreds, target],
            cost,
            updates=updates,
            name='train_model') # mode='DebugMode'


        self.forward = theano.function(
            [shreds],
            (ip1.output, ip2.output, feat.output),
            name='forward')

        self.test = theano.function(
            [shreds],
            Dw,
            name ='test'
        )

    def save(self, filename):
        f = open(filename, 'w')
        for l in self.layers:
            cPickle.dump(l.get_state(), f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    def load_caffe(self, filename):
        f = open(filename, 'r')
        params = cPickle.load(f)
        for param, l in zip(params[:2], self.layers[:2]):  # only load conv layers params
            param[1] = param[1].reshape(param[1].shape[0],)
            l.set_state(param)
        f.close()
        print 'caffe model loaded from', filename

    def load_partial(self, filename):
        f = open(filename, 'r')
        for l in self.layers[:2]:
            l.set_state(cPickle.load(f))
        f.close()
        print 'model first 2 layers are loaded from', filename

    def load(self, filename):
        f = open(filename, 'r')
        for l in self.layers:
            l.set_state(cPickle.load(f))
        f.close()
        print 'model loaded from', filename
