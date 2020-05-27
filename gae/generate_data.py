#!/usr/bin/env python3
# encoding: UTF-8

"""
Filename: generate_data.py
Author:   David Oniani
E-mail:   oniani.david@mayo.edu

Description:
    Generate allx, graph, text.index, tx, and x files.
"""

import pickle

import numpy as np
import scipy.sparse as sp

from collections import defaultdict


DATASET = "cora"
PATH = "cora"
SAVE_ROOT = "generated_cora"


def encode_onehot(labels):
    """Perform a one-hot encoding based on the labels."""

    classes = set(labels)

    classes_dict = {
        c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)
    }

    labels_onehot = np.array(
        list(map(classes_dict.get, labels)), dtype=np.int32
    )

    return labels_onehot


def main() -> None:
    """The main function. Data generation is done here."""

    # idx, features, and labels
    idx_features_labels = np.genfromtxt(
        f"{PATH}/{DATASET}.content", dtype=np.dtype(str)
    )

    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # Build the graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(f"{PATH}/{DATASET}.cites", dtype=np.int32)

    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32
    ).reshape(edges_unordered.shape)

    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(labels.shape[0], labels.shape[0]),
        dtype=np.float32,
    )

    # Build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    # Save the files
    pickle.dump(features[idx_train], open(f"{SAVE_ROOT}/ind.cora.x", "wb"))
    pickle.dump(
        sp.vstack((features[: idx_test[0]], features[idx_test[-1] + 1 :])),
        open(f"{SAVE_ROOT}/ind.cora.allx", "wb"),
    )
    pickle.dump(features[idx_test], open(f"{SAVE_ROOT}/ind.cora.tx", "wb"))

    pickle.dump(labels[idx_train], open(f"{SAVE_ROOT}/ind.cora.y", "wb"))
    pickle.dump(labels[idx_test], open(f"{SAVE_ROOT}/ind.cora.ty", "wb"))
    pickle.dump(
        np.vstack((labels[: idx_test[0]], labels[idx_test[-1] + 1 :])),
        open(f"{SAVE_ROOT}/ind.cora.ally", "wb"),
    )

    with open(f"{SAVE_ROOT}/ind.cora.test.index", "w") as file:
        for item in list(idx_test):
            file.write("%s\n" % item)

    # Save the graph
    array_adj = np.argwhere(adj.toarray())
    ori_graph = defaultdict(list)
    for edge in array_adj:
        ori_graph[edge[0]].append(edge[1])
    pickle.dump(ori_graph, open(f"{SAVE_ROOT}/ind.cora.graph", "wb"))


if __name__ == "__main__":
    main()
