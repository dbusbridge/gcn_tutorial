import scipy.sparse

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tensorflow as tf

import layers.graph as lg
import utils.sparse as us

g = nx.read_graphml('R/karate.graphml')

# nx.draw(
#     g,
#     cmap=plt.get_cmap('jet'),
#     node_color=np.log(list(nx.get_node_attributes(g, 'membership').values())))

adj = nx.adj_matrix(g)

# Get important parameters of adjacency matrix
n_nodes = adj.shape[0]

# Some preprocessing
adj_tilde = adj + np.identity(n=adj.shape[0])
d_tilde_diag = np.squeeze(np.sum(np.array(adj_tilde), axis=1))
d_tilde_inv_sqrt_diag = np.power(d_tilde_diag, -1/2)
d_tilde_inv_sqrt = np.diag(d_tilde_inv_sqrt_diag)
adj_norm = np.dot(np.dot(d_tilde_inv_sqrt, adj_tilde), d_tilde_inv_sqrt)
adj_norm_tuple = us.sparse_to_tuple(scipy.sparse.coo_matrix(adj_norm))

# Features are just the identity matrix
feat_x = np.identity(n=adj.shape[0])
feat_x_tuple = us.sparse_to_tuple(scipy.sparse.coo_matrix(feat_x))

# TensorFlow placeholders
ph = {
    'adj_norm': tf.sparse_placeholder(tf.float32, name="adj_mat"),
    'x': tf.sparse_placeholder(tf.float32, name="x")}

l_sizes = [4, 4, 2]

o_fc1 = lg.GraphConvLayer(input_dim=feat_x.shape[-1],
                          output_dim=l_sizes[0],
                          name='fc1',
                          act=tf.nn.tanh)(adj_norm=ph['adj_norm'],
                                          x=ph['x'], sparse=True)

o_fc2 = lg.GraphConvLayer(input_dim=l_sizes[0],
                          output_dim=l_sizes[1],
                          name='fc2',
                          act=tf.nn.tanh)(adj_norm=ph['adj_norm'], x=o_fc1)

o_fc3 = lg.GraphConvLayer(input_dim=l_sizes[1],
                          output_dim=l_sizes[2],
                          name='fc3',
                          act=tf.nn.tanh)(adj_norm=ph['adj_norm'], x=o_fc2)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

feed_dict = {ph['adj_norm']: adj_norm_tuple,
             ph['x']: feat_x_tuple}

outputs = sess.run(o_fc3, feed_dict=feed_dict)
x_min, x_max = outputs[:, 0].min(), outputs[:, 0].max()
y_min, y_max = outputs[:, 1].min(), outputs[:, 1].max()

node_pos_gcn = {n: tuple(outputs[j]) for j, n in enumerate(nx.nodes(g))}
node_pos_ran = {n: (np.random.uniform(low=x_min, high=x_max),
                    np.random.uniform(low=y_min, high=y_max))
                for j, n in enumerate(nx.nodes(g))}

all_node_pos = (node_pos_gcn, node_pos_ran)
plot_titles = ('3-layer randomly initialised graph CNN', 'random')

# Two subplots, unpack the axes array immediately
f, axes = plt.subplots(nrows=2, ncols=1, sharey=True, sharex=True)

for i, ax in enumerate(axes.flat):
    pos = all_node_pos[i]
    ax.set_title(plot_titles[i])

    nx.draw(
        g,
        cmap=plt.get_cmap('jet'),
        node_color=np.log(
            list(nx.get_node_attributes(g, 'membership').values())),
        pos=pos, ax=ax)
