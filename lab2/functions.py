from __future__ import print_function
from numpy import random


def rand_is_corrupted(p):
    return random.rand() < p


def print_path(path, id, nodes):
    path_size = len(path)
    i = 0
    print(nodes[id], end=' -> ')
    while i < path_size - 1:
        print(nodes[path[i]], end=' -> ')
        i = i + 1
    print(nodes[path[path_size - 1]])


def ring_topology(num, p):
    nodes = []
    for i in range(num):
        node_disabled = rand_is_corrupted(p)
        if node_disabled:
            print("node №", i, "is disabled")
        if not node_disabled:
            print("node №", i, "is working")
            nodes.append(i)
    num_nodes = list(range(len(nodes)))
    edges = []
    for i in range(len(nodes) - 1):
        if nodes[i + 1] - nodes[i] == 1:
            edges.append((i + 1, i))
            edges.append((i, i + 1))
    if nodes[0] == 0 and nodes[len(nodes) - 1] == num - 1:
        edges.append((len(nodes) - 1, 0))
        edges.append((0, len(nodes) - 1))
    return nodes, edges, num_nodes


def star_topology(num, p):
    nodes = []
    for i in range(num):
        node_disabled = rand_is_corrupted(p)
        if node_disabled:
            print("node №", i, "is disabled")
        if not node_disabled:
            print("node №", i, "is working")
            nodes.append(i)
    num_nodes = list(range(len(nodes)))
    edges = []
    if nodes[0] != 0:
        return nodes, edges, num_nodes
    for i in range(len(nodes) - 1):
        edges.append((0, i + 1))
        edges.append((i + 1, 0))
    return nodes, edges, num_nodes


def bus_topology(num, p):
    nodes = []
    for i in range(num):
        node_disabled = rand_is_corrupted(p)
        if node_disabled:
            print("node №", i, "is disabled")
        if not node_disabled:
            print("node №", i, "is working")
            nodes.append(i)
    num_nodes = list(range(len(nodes)))
    edges = []
    for i in range(len(nodes) - 1):
        if nodes[i + 1] - nodes[i] == 1:
            edges.append((i + 1, i))
            edges.append((i, i + 1))
    return nodes, edges, num_nodes
