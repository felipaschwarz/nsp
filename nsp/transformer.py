import networkx as nx
import numpy as np
import scipy.sparse

class Transformer:
    def compute_laplacian(graph):
        undir_graph = graph.to_undirected()
        degrees = [val for (node, val) in sorted(undir_graph.degree(), key=lambda pair: pair[0])]
        L = - nx.adjacency_matrix(undir_graph) # L = -W
        L.setdiag(degrees) # L = D-W
        return L

    def compute_inv_fourier_matrix(graph, type='standard'):
        if  type == 'standard':
            nodes = list(graph.nodes)
            n = graph.number_of_nodes()
            inv_F = np.identity(n)
            for node in nodes:
                for incoming_node, _ in graph.in_edges(node):
                    inv_F[nodes.index(node)] = np.logical_or(inv_F[nodes.index(node)], inv_F[nodes.index(incoming_node)])
            inv_F = scipy.sparse.tril(inv_F, format='csr')
        elif type == 'laplacian':
            # compute non-normalized graph laplacian matrix (Paper: The Emerging Field of Signal Processing on Graphs Paper, p.3/B L := D-W)
            L = Transformer.compute_laplacian(graph)
            n = graph.number_of_nodes()
            eigenvalues, eigenvectors = np.linalg.eigh(L.asfptype().toarray())
            inv_F = scipy.linalg.lu_factor(eigenvectors)
        else:
            raise NotImplementedError
        return inv_F

    def compute_moebius_sum(F, inv_F, x1, x2):
        sum = 0
        for y in range(x2, x1): # x2, x2+1, ... , x1-2, x1-1 ==> y < x1
            if inv_F[v1][v2] and inv_F[v2][v1]:
                sum = sum + F[y][x2]
        return sum

    def compute_fourier_matrix(graph, inv_F=None):
        inv_F = inv_F.toarray()
        # compute fourier matrix via moebius function ((5) in Causal Signal Processing Paper)
        nodes = list(graph.nodes)
        F = np.identity(graph.number_of_nodes())
        for x1 in range(graph.number_of_nodes()):
            print(x1)
            for x2 in range(x1):
                if inv_F[x1][x2] != 0:
                    F[x1][x2] = - Transformer.compute_moebius_sum(F, inv_F, x1, x2)
        return F

    def compute_fourier_activations(activations, inv_F, type, lu_piv=None):
        signals = activations.to_vector()

        if type == 'standard':
            fourier_signals = scipy.sparse.linalg.spsolve_triangular(inv_F, signals)
        elif type == 'laplacian':
            if lu_piv is None:
                warnings.warn('lu_piv = None - pass lu-decomposition if used on more than one signal')
                fourier_signals = scipy.linalg.solve(inv_F, signals)
            else:
                fourier_signals = scipy.linalg.lu_solve(lu_piv, signals)
        else:
            raise NotImplementedError

        fourier_activations = activations.to_activations(fourier_signals)
        return fourier_activations
