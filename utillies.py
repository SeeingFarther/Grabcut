import numpy as np

def compute_sample_expectation(img):
    total_sum = 0
    num_of_edges = 0
    expectation = 0

    # We use element-wise multiply because, as we checked and got benchmark results,it
    # is faster than doing Einstein notation (dot product Einstein notation - np.einsum("ijk, ijk -> ijk", A, B))
    # Compute sum of dot product between neighbors in the down diagonal right
    dist_down_diagonal_right = np.array(img[:-1, : -1] - img[1:, 1:])  # Np index games(shifting) for fast dist compute
    dot_down_diagonal_right = np.array(np.sum(dist_down_diagonal_right * dist_down_diagonal_right, axis=2),  dtype=np.float64)
    total_sum = np.sum(dot_down_diagonal_right)  # Add sum to total sum
    num_of_edges = dot_down_diagonal_right.size  # Add number of edges to total num of edges

    # Compute sum of dot product between neighbors in the right
    dist_right_col = np.array(img[:, : -1] - img[:, 1:])  # Np index games(shifting) for fast dist compute
    dot_right_col = np.array(np.sum(dist_right_col * dist_right_col, axis=2),  dtype=np.float64)
    total_sum += np.sum(dot_right_col)  # Add sum to total sum
    num_of_edges += dot_right_col.size  # Add number of edges to total num of edges

    # Compute sum of dot product between neighbors in the bottom
    dist_bottom_row = np.array(img[: -1, :] - img[1:, :])  # Np index games(shifting) for fast dist compute
    dot_bottom_row = np.array(np.sum(dist_bottom_row * dist_bottom_row, axis=2),  dtype=np.float64)
    total_sum += np.sum(dot_bottom_row)  # Add sum to total sum
    num_of_edges += dot_bottom_row.size  # Add number of edges to total num of edges

    # Compute sum of dot product between neighbors in the up diagonal right side
    dist_up_diagonal_right = np.array(img[1:, :-1] - img[:-1, 1:]) # Np index games(shifting) for fast dist compute
    dot_up_diagonal_right = np.array(np.sum(dist_up_diagonal_right * dist_up_diagonal_right, axis=2),  dtype=np.float64)
    total_sum += np.sum(dot_up_diagonal_right)  # Add sum to total sum
    num_of_edges += dot_up_diagonal_right.size  # Add number of edges to total num of edges

    # Compute expectation
    expectation = total_sum / num_of_edges
    return expectation


def calculate_n_edges(graph, img, expectation):
    h = img.shape[0]
    w = img.shape[1]
    n_edges = []
    n_cap = []
    gamma = 50
    n_edge_sum = np.zeros((h, w), dtype=np.float64).reshape(h, w)
    vertices_map = graph.get_vertices_map()
    beta = (1 / (2 * expectation))

    # Build N-edges
    # We use element-wise multiply because we checked and got benchmark results that it
    # is faster than doing Einstein notation (dot product Einstein notation - np.einsum("ijk, ijk -> ijk", A, B))
    # Compute sum of dot product between neighbors in the down diagonal right
    dist_down_diagonal_right = img[:-1, : -1] - img[1:, 1:]  # Np index games(shifting) for fast dist compute
    dot_down_diagonal_right = np.array(np.sum(dist_down_diagonal_right * dist_down_diagonal_right, axis=2), dtype=np.float64)
    # Calculate the N-edges capacities and add them to the list
    cap = np.exp(-dot_down_diagonal_right * beta) * (gamma / np.sqrt(2))
    n_cap = cap.flatten().tolist()
    left_vertex_edges = (vertices_map[:-1, : -1]).flatten().tolist()
    right_vertex_edges = (vertices_map[1:, 1:]).flatten().tolist()
    # Add each edge sum to total N-edges sum of the relevant pixel which will be used to calculate K from the paper
    n_edge_sum[:-1, : -1] += cap
    n_edge_sum[1:, 1:] += cap

    # Compute sum of dot product between neighbors in the right
    dist_right_col = img[:, :-1] - img[:, 1:]  # Np index games(shifting) for fast dist compute
    dot_right_col = np.array(np.sum(dist_right_col * dist_right_col, axis=2),  dtype=np.float64)
    # Calculate the N-edges capacities and add them to the list
    cap = np.exp(-dot_right_col * beta) * gamma
    n_cap += cap.flatten().tolist()
    left_vertex_edges += (vertices_map[:, :-1]).flatten().tolist()
    right_vertex_edges += (vertices_map[:, 1:]).flatten().tolist()
    # Add each edge sum to total N-edges sum of the relevant pixel which will be used to calculate K from the paper
    n_edge_sum[:, :-1] += cap
    n_edge_sum[:, 1:] += cap

    # Compute sum of dot product between neighbors in the bottom
    dist_bottom_row = img[:-1, :] - img[1:, :]  # Np index games(shifting) for fast dist compute
    dot_bottom_row = np.array(np.sum(dist_bottom_row * dist_bottom_row, axis=2),  dtype=np.float64)
    # Calculate the N-edges capacities and add them to the list
    cap = np.exp(-dot_bottom_row * beta) * gamma
    n_cap += cap.flatten().tolist()
    left_vertex_edges += (vertices_map[:-1, :]).flatten().tolist()
    right_vertex_edges += (vertices_map[1:, :]).flatten().tolist()
    # Add each edge sum to total N-edges sum of the relevant pixel which will be used to calculate K from the paper
    n_edge_sum[:-1, :] += cap
    n_edge_sum[1:, :] += cap

    # Compute sum of dot product between neighbors in the up diagonal right side
    dist_up_diagonal_right = img[1:, :-1] - img[:-1, 1:]  # Np index games(shifting) for fast dist compute
    dot_up_diagonal_right = np.array(np.sum(dist_up_diagonal_right * dist_up_diagonal_right, axis=2),  dtype=np.float64)
    # Calculate the N-edges capacities and add them to the list
    cap = np.exp(-dot_up_diagonal_right * beta) * (gamma / np.sqrt(2))
    n_cap += cap.flatten().tolist()
    left_vertex_edges += (vertices_map[1:, :-1]).flatten().tolist()
    right_vertex_edges += (vertices_map[:-1, 1:]).flatten().tolist()
    # Add each edge sum to total N-edges sum of the relevant pixel which will be used to calculate K from the paper
    n_edge_sum[1:, :-1] += cap
    n_edge_sum[:-1, 1:] += cap

    # Create edges
    for i in range(len(left_vertex_edges)):
        n_edges.append((left_vertex_edges[i], right_vertex_edges[i]))

    # Set K
    graph.set_K(np.max(n_edge_sum))

    # Add the N-edges and their capacities to the graph
    graph.set_n_edges_cap(n_cap)
    graph.set_n_edges(n_edges)
    return


def build_soft_t_edges(s, t, soft_indices, d_bg, d_fg, t_edges, capacities):
    index = 0
    for u in soft_indices:
        t_edges.append((s, u))
        t_edges.append((t, u))
        capacities.append(d_bg[index])
        capacities.append(d_fg[index])
        index += 1

    return t_edges, capacities


def build_hard_t_edges(graph, bg_hard_indices, fg_hard_indices):
    capacities = []
    t_edges = []
    s = graph.get_source()
    t = graph.get_target()
    K = graph.get_K()

    # Build hard foreground edges
    for u in fg_hard_indices:
        t_edges.append((s, u))
        t_edges.append((t, u))
        capacities.append(K)
        capacities.append(0)

    # Build hard background T-edges
    for u in bg_hard_indices:
        t_edges.append((s, u))
        t_edges.append((t, u))
        capacities.append(0)
        capacities.append(K)

    # Add hard foreground and hard background T-edges to the graph
    graph.set_hard_t_edges_cap(capacities)
    graph.set_hard_t_edges(t_edges)

    return