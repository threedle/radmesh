# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Tuple, Optional, Union, List
from random import randint

import torch


# ------------------------ Laplacian Matrices ------------------------ #
# This file contains implementations of differentiable laplacian matrices.
# These include
# 1) Standard Laplacian matrix
# 2) Cotangent Laplacian matrix
# 3) Norm Laplacian matrix
# -------------------------------------------------------------------- #


def laplacian(verts: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """
    Computes the laplacian matrix.
    The definition of the laplacian is
    L[i, j] =    -1       , if i == j
    L[i, j] = 1 / deg(i)  , if (i, j) is an edge
    L[i, j] =    0        , otherwise
    where deg(i) is the degree of the i-th vertex in the graph.

    Args:
        verts: tensor of shape (V, 3) containing the vertices of the graph
        edges: tensor of shape (E, 2) containing the vertex indices of each edge
    Returns:
        L: Sparse FloatTensor of shape (V, V)
    """
    V = verts.shape[0]

    e0, e1 = edges.unbind(1)

    idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)

    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device)
    # pyre-fixme[16]: Module `sparse` has no attribute `FloatTensor`.
    A = torch.sparse_coo_tensor(idx, ones, (V, V))

    # the sum of i-th row of A gives the degree of the i-th vertex
    deg = torch.sparse.sum(A, dim=1).to_dense()

    # We construct the Laplacian matrix by adding the non diagonal values
    # i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge
    deg0 = deg[e0]
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
    deg1 = deg[e1]
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)
    val = torch.cat([deg0, deg1])
    # pyre-fixme[16]: Module `sparse` has no attribute `FloatTensor`.
    L = torch.sparse_coo_tensor(idx, val, (V, V))

    # Then we add the diagonal values L[i, i] = -1.
    idx = torch.arange(V, device=verts.device)
    idx = torch.stack([idx, idx], dim=0)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device)
    # pyre-fixme[16]: Module `sparse` has no attribute `FloatTensor`.
    L -= torch.sparse_coo_tensor(idx, ones, (V, V))

    return L


def cot_laplacian(
    verts: torch.Tensor, faces: torch.Tensor, eps: float = 1e-12
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the Laplacian matrix with cotangent weights and the inverse of the
    face areas.

    Args:
        verts: tensor of shape (V, 3) containing the vertices of the graph
        faces: tensor of shape (F, 3) containing the vertex indices of each face
    Returns:
        2-element tuple containing
        - **L**: Sparse FloatTensor of shape (V,V) for the Laplacian matrix.
           Here, L[i, j] = cot a_ij + cot b_ij iff (i, j) is an edge in meshes.
           See the description above for more clarity.
        - **inv_areas**: FloatTensor of shape (V,) containing the inverse of sum of
           face areas containing each vertex
    """
    V, F = verts.shape[0], faces.shape[0]

    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Side lengths of each triangle, of shape (sum(F_n),)
    # A is the side opposite v1, B is opposite v2, and C is opposite v3
    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    # Area of each triangle (with Heron's formula); shape is (sum(F_n),)
    s = 0.5 * (A + B + C)
    # note that the area can be negative (close to 0) causing nans after sqrt()
    # we clip it to a small positive value
    # pyre-fixme[16]: `float` has no attribute `clamp_`.
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=eps).sqrt()

    # Compute cotangents of angles, of shape (sum(F_n), 3)
    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=1)
    cot /= 4.0

    # Construct a sparse matrix by basically doing:
    # L[v1, v2] = cota
    # L[v2, v0] = cotb
    # L[v0, v1] = cotc
    ii = faces[:, [1, 2, 0]]
    jj = faces[:, [2, 0, 1]]
    idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
    # pyre-fixme[16]: Module `sparse` has no attribute `FloatTensor`.
    L = torch.sparse_coo_tensor(idx, cot.view(-1), (V, V))

    # Make it symmetric; this means we are also setting
    # L[v2, v1] = cota
    # L[v0, v2] = cotb
    # L[v1, v0] = cotc
    L += L.t()

    # For each vertex, compute the sum of areas for triangles containing it.
    idx = faces.view(-1)
    inv_areas = torch.zeros(V, dtype=torch.float32, device=verts.device)
    val = torch.stack([area] * 3, dim=1).view(-1)
    inv_areas.scatter_add_(0, idx, val)
    idx = inv_areas > 0
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    inv_areas[idx] = 1.0 / inv_areas[idx]
    inv_areas = inv_areas.view(-1, 1)

    return L, inv_areas


def norm_laplacian(
    verts: torch.Tensor, edges: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """
    Norm laplacian computes a variant of the laplacian matrix which weights each
    affinity with the normalized distance of the neighboring nodes.
    More concretely,
    L[i, j] = 1. / wij where wij = ||vi - vj|| if (vi, vj) are neighboring nodes

    Args:
        verts: tensor of shape (V, 3) containing the vertices of the graph
        edges: tensor of shape (E, 2) containing the vertex indices of each edge
    Returns:
        L: Sparse FloatTensor of shape (V, V)
    """
    edge_verts = verts[edges]  # (E, 2, 3)
    v0, v1 = edge_verts[:, 0], edge_verts[:, 1]

    # Side lengths of each edge, of shape (E,)
    w01 = 1.0 / ((v0 - v1).norm(dim=1) + eps)

    # Construct a sparse matrix by basically doing:
    # L[v0, v1] = w01
    # L[v1, v0] = w01
    e01 = edges.t()  # (2, E)

    V = verts.shape[0]
    # pyre-fixme[16]: Module `sparse` has no attribute `FloatTensor`.
    L = torch.sparse_coo_tensor(e01, w01, (V, V))
    L = L + L.t()

    return L


def masked_gather(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Helper function for torch.gather to collect the points at
    the given indices in idx where some of the indices might be -1 to
    indicate padding. These indices are first replaced with 0.
    Then the points are gathered after which the padded values
    are set to 0.0.

    Args:
        points: (N, P, D) float32 tensor of points
        idx: (N, K) or (N, P, K) long tensor of indices into points, where
            some indices are -1 to indicate padding

    Returns:
        selected_points: (N, K, D) float32 tensor of points
            at the given indices
    """

    if len(idx) != len(points):
        raise ValueError("points and idx must have the same batch dimension")

    N, P, D = points.shape

    if idx.ndim == 3:
        # Case: KNN, Ball Query where idx is of shape (N, P', K)
        # where P' is not necessarily the same as P as the
        # points may be gathered from a different pointcloud.
        K = idx.shape[2]
        # Match dimensions for points and indices
        idx_expanded = idx[..., None].expand(-1, -1, -1, D)
        points = points[:, :, None, :].expand(-1, -1, K, -1)
    elif idx.ndim == 2:
        # Farthest point sampling where idx is of shape (N, K)
        idx_expanded = idx[..., None].expand(-1, -1, D)
    else:
        raise ValueError("idx format is not supported %s" % repr(idx.shape))

    idx_expanded_mask = idx_expanded.eq(-1)
    idx_expanded = idx_expanded.clone()
    # Replace -1 values with 0 for gather
    idx_expanded[idx_expanded_mask] = 0
    # Gather points
    selected_points = points.gather(dim=1, index=idx_expanded)
    # Replace padded values
    selected_points[idx_expanded_mask] = 0.0
    return selected_points


# from sample_farthest_points
def sample_farthest_points(
    points: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    K: Union[int, List, torch.Tensor] = 50,
    random_start_point: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Iterative farthest point sampling algorithm [1] to subsample a set of
    K points from a given pointcloud. At each iteration, a point is selected
    which has the largest nearest neighbor distance to any of the
    already selected points.

    Farthest point sampling provides more uniform coverage of the input
    point cloud compared to uniform random sampling.

    [1] Charles R. Qi et al, "PointNet++: Deep Hierarchical Feature Learning
        on Point Sets in a Metric Space", NeurIPS 2017.

    Args:
        points: (N, P, D) array containing the batch of pointclouds
        lengths: (N,) number of points in each pointcloud (to support heterogeneous
            batches of pointclouds)
        K: samples required in each sampled point cloud (this is typically << P). If
            K is an int then the same number of samples are selected for each
            pointcloud in the batch. If K is a tensor is should be length (N,)
            giving the number of samples to select for each element in the batch
        random_start_point: bool, if True, a random point is selected as the starting
            point for iterative sampling.

    Returns:
        selected_points: (N, K, D), array of selected values from points. If the input
            K is a tensor, then the shape will be (N, max(K), D), and padded with
            0.0 for batch elements where k_i < max(K).
        selected_indices: (N, K) array of selected indices. If the input
            K is a tensor, then the shape will be (N, max(K), D), and padded with
            -1 for batch elements where k_i < max(K).
    """
    N, P, D = points.shape
    device = points.device

    # Validate inputs
    if lengths is None:
        lengths = torch.full((N,), P, dtype=torch.int64, device=device)
    else:
        if lengths.shape != (N,):
            raise ValueError("points and lengths must have same batch dimension.")
        if lengths.max() > P:
            raise ValueError("Invalid lengths.")

    # TODO: support providing K as a ratio of the total number of points instead of as an int
    if isinstance(K, int):
        K = torch.full((N,), K, dtype=torch.int64, device=device)
    elif isinstance(K, list):
        K = torch.tensor(K, dtype=torch.int64, device=device)

    if K.shape[0] != N:
        raise ValueError("K and points must have the same batch dimension")

    # Find max value of K
    max_K = torch.max(K)

    # List of selected indices from each batch element
    all_sampled_indices = []

    for n in range(N):
        # Initialize an array for the sampled indices, shape: (max_K,)
        sample_idx_batch = torch.full(
            # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
            #  typing.Tuple[int, ...]]` but got `Tuple[Tensor]`.
            (int(max_K.item()),),
            fill_value=-1,
            dtype=torch.int64,
            device=device,
        )
        length_n = int(lengths[n].item())

        # Initialize closest distances to inf, shape: (P,)
        # This will be updated at each iteration to track the closest distance of the
        # remaining points to any of the selected points
        closest_dists = points.new_full(
            # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
            #  typing.Tuple[int, ...]]` but got `Tuple[Tensor]`.
            (length_n,),
            float("inf"),
            dtype=torch.float32,
        )

        # Select a random point index and save it as the starting point
        # pyre-fixme[6]: For 2nd argument expected `int` but got `Tensor`.
        selected_idx = randint(0, length_n - 1) if random_start_point else 0
        sample_idx_batch[0] = selected_idx

        # If the pointcloud has fewer than K points then only iterate over the min
        # pyre-fixme[6]: For 1st param expected `SupportsRichComparisonT` but got
        #  `Tensor`.
        # pyre-fixme[6]: For 2nd param expected `SupportsRichComparisonT` but got
        #  `Tensor`.
        k_n = min(length_n, int(K[n].item()))

        # Iteratively select points for a maximum of k_n
        for i in range(1, k_n):
            # Find the distance between the last selected point
            # and all the other points. If a point has already been selected
            # it's distance will be 0.0 so it will not be selected again as the max.
            dist = points[n, selected_idx, :] - points[n, : lengths[n], :]
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            dist_to_last_selected = (dist**2).sum(-1)  # (P - i)

            # If closer than currently saved distance to one of the selected
            # points, then updated closest_dists
            closest_dists = torch.min(dist_to_last_selected, closest_dists)  # (P - i)

            # The aim is to pick the point that has the largest
            # nearest neighbour distance to any of the already selected points
            selected_idx = torch.argmax(closest_dists)
            sample_idx_batch[i] = selected_idx

        # Add the list of points for this batch to the final list
        all_sampled_indices.append(sample_idx_batch)

    all_sampled_indices = torch.stack(all_sampled_indices, dim=0)

    # Gather the points
    all_sampled_points = masked_gather(points, all_sampled_indices)

    # Return (N, max_K, D) subsampled points and indices
    return all_sampled_points, all_sampled_indices
