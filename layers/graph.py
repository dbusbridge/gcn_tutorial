import tensorflow as tf


def _dot(x, y, sparse=False):
    if sparse:
        return tf.sparse_tensor_dense_matmul(x, y)
    return tf.matmul(x, y)


class GraphConvLayer:
    def __init__(self, input_dim, output_dim,
                 name, act=tf.nn.relu, bias=False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.bias = bias

        with tf.variable_scope(name):
            with tf.name_scope('weights'):
                self.w = tf.get_variable(
                    name='w',
                    shape=(self.input_dim, self.output_dim),
                    initializer=tf.contrib.layers.xavier_initializer())

            if self.bias:
                with tf.name_scope('biases'):
                    self.b = tf.get_variable(
                        name='b',
                        initializer=tf.constant(0.1, shape=(self.output_dim,)))

    def call(self, adj_norm, x, sparse=False):
        hw = _dot(x=x, y=self.w, sparse=sparse)
        ahw = _dot(x=adj_norm, y=hw, sparse=True)

        if not self.bias:
            return self.act(ahw)

        return self.act(tf.add(ahw, self.bias))

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)
