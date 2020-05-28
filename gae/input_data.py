import scipy.sparse as sp
import numpy as np


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {
        c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)
    }
    labels_onehot = np.array(
        list(map(classes_dict.get, labels)), dtype=np.int32
    )
    return labels_onehot


def load_data(dataset="covid"):
    """Load citation network dataset (cora only for now)"""

    print("Loading {} dataset...".format(dataset))

    idx_features_labels = np.genfromtxt(
        f"data/{dataset}.content", dtype=np.dtype(str), delimiter=","
    )

    features = sp.csr_matrix(idx_features_labels[:, 2], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, 1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)

    idx_map = {j: i for i, j in enumerate(idx)}

    edges_unordered = np.genfromtxt(f"data/{dataset}.cites", dtype=np.int32)

    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32
    ).reshape(edges_unordered.shape)

    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(labels.shape[0], labels.shape[0]),
        dtype=np.float32,
    )

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    print(
        "Dataset has {} nodes, {} edges, {} features.".format(
            adj.shape[0], edges.shape[0], features.shape[1]
        )
    )

    return adj, features.todense()
