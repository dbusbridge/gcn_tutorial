# A tutorial on Graph Convolutional Neural Networks

## Data

The data we use is [Zachary's karate club](https://en.wikipedia.org/wiki/Zachary%27s_karate_club), a standard toy social network. It is a data set consisting of: 

+ 34 nodes, each corresponding to members of a karate club

+ 78 pairwise links that correspond to social interactions of the members outside of the club.

A conflict arose in the club which caused the club to split into several factions. Using modularity based clustering, these factions can be recovered from the graph structure alone (except for one node).

## Original implementation of Graph Convolutional Neural Networks

For the original TensorFlow implementation (in a Keras style) see https://github.com/tkipf/gcn.

## References

### Blog posts

+ Great introductory post https://tkipf.github.io/graph-convolutional-networks/

### Papers

+ Defferrard, M., Bresson, X., & Vandergheynst, P. (2016). Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering. Nips, (Nips), 1–14. http://arxiv.org/abs/1606.09375

+ Kipf, T. N., & Welling, M. (2016). Semi-Supervised Classification with Graph Convolutional Networks, 1–14. http://arxiv.org/abs/1609.02907

+ Kipf, T. N., & Welling, M. (2016). Variational Graph Auto-Encoders. Nipsw, (2), 1–3. http://arxiv.org/abs/1611.07308

+ Schlichtkrull, M., Kipf, T. N., Bloem, P., Berg, R. van den, Titov, I., & Welling, M. (2017). Modeling Relational Data with Graph Convolutional Networks, 1–12. http://arxiv.org/abs/1703.06103
