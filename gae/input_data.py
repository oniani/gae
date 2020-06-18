import scipy.sparse as sp
import numpy as np
import networkx as nx


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {
        c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)
    }
    labels_onehot = np.array(
        list(map(classes_dict.get, labels)), dtype=np.int32
    )
    return labels_onehot


def read_emb(filename: str):
    """Reads the embeddings."""

    x = []
    y = []

    # Open embedding file
    with open(filename) as file:
        next(file)
        for line in file:
            splits = line.strip().split()
            label = splits[0]
            vec = [float(v) for v in splits[1:]]

            x.append(vec)
            y.append(label)

    return dict(zip(y, x))


def load_data(dataset="covid"):
    """Load citation network dataset (cora only for now)"""

    print("Loading {} dataset...".format(dataset))

    idx_features_labels = np.genfromtxt(
        "data/{}.content".format(dataset), dtype=np.dtype(str), delimiter=","
    )

    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    graph = read_emb("data/{}-nodupe.emd".format(dataset))

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    print("Done loading the data!")

    return adj, features.todense()
