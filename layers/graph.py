import tensorflow as tf


def matmul(x, y, sparse=False):
    """Wrapper for sparse matrix multiplication."""
    if sparse:
        return tf.sparse_tensor_dense_matmul(x, y)
    return tf.matmul(x, y)


class GraphConvLayer:
    def __init__(
            self,
            input_dim,
            output_dim,
            activation=None,
            use_bias=False,
            name="graph_conv"):
        """Initialise a Graph Convolution layer.

        Args:
            input_dim (int): The input dimensionality.
            output_dim (int): The output dimensionality, i.e. the number of
                units.
            activation (callable): The activation function to use. Defaults to
                no activation function.
            use_bias (bool): Whether to use bias or not. Defaults to `False`.
            name (str): The name of the layer. Defaults to `graph_conv`.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.use_bias = use_bias
        self.name = name

        with tf.variable_scope(self.name):
            self.w = tf.get_variable(
                name='w',
                shape=(self.input_dim, self.output_dim),
                initializer=tf.initializers.glorot_uniform())

            if self.use_bias:
                self.b = tf.get_variable(
                    name='b',
                    initializer=tf.constant(0.1, shape=(self.output_dim,)))

    def call(self, adj_norm, x, sparse=False):
        x = matmul(x=x, y=self.w, sparse=sparse)  # XW
        x = matmul(x=adj_norm, y=x, sparse=True)  # AXW

        if self.use_bias:
            x = tf.add(x, self.use_bias)              # AXW + B

        if self.activation is not None:
            x = self.activation(x)                # activation(AXW + B)

        return x

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)
