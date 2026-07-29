from typing import Tuple, Literal, Optional, List, Callable, Sequence, Union, cast, Set
from typing import Type
from functools import cached_property
import os
import random
from dataclasses import dataclass
from . import misc_helpers

import torch

TORCH_PLS_BE_DETERMINISTIC = misc_helpers.get_bool_env_variable(
    "TORCH_PLS_BE_DETERMINISTIC"
)
if TORCH_PLS_BE_DETERMINISTIC:
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

DRMSH_PROCRUSTES_DEGEN_SVDVALS_BECAREFUL = misc_helpers.get_bool_env_variable(
    "DRMSH_PROCRUSTES_DEGEN_SVDVALS_BECAREFUL", True
)

import igl
import cholespy
import numpy as np
import torch.nn as nn

from .pytorch3d.structures import Meshes
from .pytorch3d import transforms as pt3d_transforms
from .pytorch3d import ops as pt3d_ops

from thlog import Thlogger, LOG_INFO, LOG_DEBUG, LOG_TRACE, VIZ_INFO, VIZ_TRACE
from thronf import Thronfig, InvalidConfigError

thlog = Thlogger(LOG_INFO, VIZ_INFO, "deformations")

NameOfSpecialVertexToPin = Literal[
    "vertex_0", "min_z", "max_z", "min_y", "max_y", "min_x", "max_x"
]


########################################### misc utils
def normalize_to_side2_cube_inplace(meshes: Meshes):
    bounding_boxes = meshes.get_bounding_boxes()  # (n_meshes, 3, 2)
    mesh_to_verts_packed_first_idx = meshes.mesh_to_verts_packed_first_idx()

    bounding_boxes_packed = bounding_boxes[meshes.verts_packed_to_mesh_idx()]
    # ^ (sum of all vertex counts, 3, 2)
    min_coords_packed = bounding_boxes_packed[:, :, 0]
    max_coords_packed = bounding_boxes_packed[:, :, 1]
    extent_packed, _ = (max_coords_packed - min_coords_packed).max(dim=-1)

    scale_per_mesh = 2 / extent_packed[mesh_to_verts_packed_first_idx]  # (n_meshes, )
    center_packed = (min_coords_packed + max_coords_packed) / 2

    # normalize in-place
    meshes.offset_verts_(-center_packed)
    meshes.scale_verts_(scale_per_mesh)


def normalize_to_side2_cube_np_singlemesh(verts: np.typing.ArrayLike) -> np.ndarray:
    # actually just AABB
    mincoord = np.min(verts, axis=0)
    maxcoord = np.max(verts, axis=0)
    extent = (maxcoord - mincoord).max()
    scale = 2 / extent
    center = (mincoord + maxcoord) / 2
    return (verts - center) * scale


def make_sparse_diag(diag: torch.Tensor) -> torch.Tensor:
    """
    given `diag`, 1D dense tensor of shape (n,), builds a square sparse_coo
    matrix of shape (n,n) that has it as the main diagonal
    """
    idx = torch.arange((n := diag.size(0)), device=diag.device)
    return torch.sparse_coo_tensor(torch.stack((idx, idx), dim=0), diag, (n, n))


def calc_gradient_operator(
    verts: torch.Tensor,
    faces: torch.Tensor,
    face_normals_if_available: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    calculates the grad operator for a mesh using a per-vertex hat basis,
    resulting in a per-triangle linear operator.

    Returns:
    - grad, a (sparse) tensor of shape (#F * n_coords, #V) that can be
    matmul'd with any per-vertex quantity given as an (#V,*) tensor to give
    the per-face gradients (this result would have shape (#F * n_coords, *),
    which can be viewed as (#F, n_coords, *) if that's more convenient)
    - computed original-length face normals (#F, 3)
    - face doubleareas (which is also the lengths of those face normals) (#F,)
    """
    device = verts.device
    face_verts_coords = verts[faces]  # (n_faces, 3, n_coords)
    n_coords = face_verts_coords.shape[-1]
    v1 = face_verts_coords[:, 0]
    v2 = face_verts_coords[:, 1]
    v3 = face_verts_coords[:, 2]
    # edge vectors are named after the vertex they are opposite
    # verts and edges go CCW:
    #         v2
    #      e1 /|
    #     v3 / |
    #        \ | e3
    #      e2 \|
    #         v1
    # normal (u) points out from the screen
    e1 = v3 - v2
    e2 = v1 - v3
    e3 = v2 - v1

    if n_coords == 2:
        # stick onto e1, e2, e3 a z=0 coordinate and remove that later
        zeros = torch.zeros_like(e1[:, -1:])
        e1 = torch.cat((e1, zeros), dim=-1)
        e2 = torch.cat((e2, zeros), dim=-1)
        e3 = torch.cat((e3, zeros), dim=-1)

    face_normals = (
        face_normals_if_available
        if face_normals_if_available is not None
        else torch.linalg.cross(e1, e2)
    )
    face_doubleareas = torch.linalg.norm(
        face_normals, dim=-1, keepdim=True
    )  # also face normal magnitude
    u = face_normals / face_doubleareas  # face unit normals

    # edges rotated 90deg around normal (so that they point into the triangle)
    # (still on the triangle's plane), with length = original length / face doublearea
    e3perp = torch.linalg.cross(u, e3)
    e3perp /= e3perp.norm(dim=-1, keepdim=True)  # normalize,
    e3perp *= e3.norm(dim=-1, keepdim=True) / face_doubleareas

    e2perp = torch.linalg.cross(u, e2)
    e2perp /= e2perp.norm(dim=-1, keepdim=True)
    e2perp *= e2.norm(dim=-1, keepdim=True) / face_doubleareas

    e1perp = -(e3perp + e2perp)

    if n_coords == 2:
        # take out the fake zero coord (added for cross product purposes) if we
        # originally had 2d verts only
        e1perp = e1perp[:, :2]
        e2perp = e2perp[:, :2]
        e3perp = e3perp[:, :2]

    # build values and indices to fill the sparse matrix
    n_faces = faces.shape[0]
    n_verts = verts.shape[0]
    N_VERTS_PER_FACE = 3
    # indices is 2 stacked tensors, made by stacking `faces` and `vert_indices`:
    #   (faces = [0,0,0,0,0,0,0,0,0,3,3,3,3,3,3,3,3,3,6,6,6,6,6,6,6,6,6,...3*(n_faces-1),3*(n_faces-1),3*(n_faces-1)],
    #          + [0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2...,0,1,2],
    #    vert_indices = [f1_v1i,f1_v1i,f1_v1i, f1_v2i,f1_v2i,f1_v2i, f1_v3i,f1_v3i,f1_v3i, ..., f_nfaces_v0i, f_nfaces_v1i, f_nface_v2i])
    # all these indices__ tensors are 1D with the same length n_faces * N_COORDS * N_VERTS_PER_FACE
    #
    # values are
    #  [f1_e1perp_x, f1_e1perp_y, f1_e1perp_z, f1_e2perp_x, f1_e2perp_y, f1_e2perp_z, f1_e3perp_x, f1_e3perp_y, f1_e3perp_z, ...]
    indices__faces = torch.repeat_interleave(
        torch.arange(n_faces, device=device), n_coords * N_VERTS_PER_FACE
    )
    indices__axes = torch.arange(n_coords, device=device).repeat(n_faces * N_VERTS_PER_FACE)
    indices__vert_indices = torch.repeat_interleave(
        torch.stack((faces[:, 0], faces[:, 1], faces[:, 2]), dim=0).T.flatten(),
        n_coords,
    )
    indices = torch.stack(
        (n_coords * indices__faces + indices__axes, indices__vert_indices), dim=0
    )
    values = torch.stack((e1perp, e2perp, e3perp), dim=0).transpose(0, 1).flatten()
    grad = torch.sparse_coo_tensor(
        indices, values, size=(n_faces * n_coords, n_verts), device=device
    )
    return grad, face_normals, face_doubleareas.squeeze(-1)


def make_padded_to_packed_indexer(
    num_elements_per_mesh: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A pytorch3d `Meshes` object provides quantities organized in packed tensors
    and padded tensors, i.e. `verts_packed` of shape `(n_verts_across_all_meshes,
    3)` and `verts_padded` of shape `(batch_size, max_n_verts_in_a_mesh, 3)`.

    Given the corresponding `meshes.num_*_per_mesh()` tensor, this will return
    indexing tensors `batch_idx`, `idx_in_batch` such that
    >>> elems_padded[batch_idx, idx_in_batch] == elems_packed

    The same indexing tensors can be used to modify elements in the padded tensor, i.e.
    >>> elems_padded[batch_idx, idx_in_batch] = new_elems_packed
    """
    batch_idx = torch.arange(
        num_elements_per_mesh.size(0),
        device=(device := num_elements_per_mesh.device),
    ).repeat_interleave(num_elements_per_mesh, dim=0)
    idx_in_batch = torch.cat(
        tuple(torch.arange(int(n), device=device) for n in num_elements_per_mesh),
        dim=0,
    )
    return batch_idx, idx_in_batch


def gather_per_face_quantity_into_per_vertex_quantity_inplace(
    face_areas: torch.Tensor,
    sum_incident_face_area_per_vertex: torch.Tensor,
    faces: torch.Tensor,
    quantity: torch.Tensor,
    out: torch.Tensor,
):
    """
    gather face quantities onto adjacent verts; around a vert, the weight of
    each face's value is its area in proportion with the sum of adjacent faces'
    areas.
    face_areas is assumed to have shape (n_faces,) and is unsqueezed accordingly
    to broadcast to quantity which should have shape (n_faces, *)
    """
    dim_expander = tuple(1 for _ in quantity.shape[1:])
    face_areas_times_quantity = quantity * face_areas.view(
        face_areas.size(0), *dim_expander
    )
    out.zero_()
    out.index_put_((faces[:, 0],), face_areas_times_quantity, accumulate=True)
    out.index_put_((faces[:, 1],), face_areas_times_quantity, accumulate=True)
    out.index_put_((faces[:, 2],), face_areas_times_quantity, accumulate=True)
    out.div_(
        sum_incident_face_area_per_vertex.view(
            sum_incident_face_area_per_vertex.size(0), *dim_expander
        )
    )


def per_vertex_packed_to_list(
    meshes: Meshes, per_vertex_quantity: torch.Tensor
) -> List[torch.Tensor]:
    # the pytorch3d.structures.packed_to_list function just calls torch.split so
    return per_vertex_quantity.split(list(meshes.num_verts_per_mesh()), dim=0)


def per_face_packed_to_list(
    meshes: Meshes, per_face_quantity: torch.Tensor
) -> List[torch.Tensor]:
    return per_face_quantity.split(list(meshes.num_faces_per_mesh()), dim=0)


def calc_barycentric_mass_matrix(
    verts: torch.Tensor, faces: torch.Tensor, return_diagonal_only=False
) -> torch.Tensor:
    """
    The barycentric mass matrix is a (sparse) diagonal tensor of shape (n_verts,
    n_verts) where the entry for vertex i is 1/3 the total area of the faces
    surrounding vertex i.
    If return_diagonal_only (default=False) is True, then return masses as a 1D
    tensor of shape (n_verts,) rather than a 2D sparse matrix.
    """

    faceverts0 = faces[:, 0]
    faceverts1 = faces[:, 1]
    faceverts2 = faces[:, 2]
    n_verts = verts.shape[0]

    face_areas = 0.5 * torch.linalg.norm(
        torch.cross(
            verts[faceverts2] - verts[faceverts0], verts[faceverts1] - verts[faceverts0]
        ),
        dim=-1,
    )

    mass_per_vertex = torch.zeros(n_verts, device=verts.device)
    mass_per_vertex.index_put_((faceverts0,), face_areas, accumulate=True)
    mass_per_vertex.index_put_((faceverts1,), face_areas, accumulate=True)
    mass_per_vertex.index_put_((faceverts2,), face_areas, accumulate=True)
    mass_per_vertex *= 1 / 3

    if return_diagonal_only:
        return mass_per_vertex
    else:
        mass_matrix = torch.sparse_coo_tensor(
            torch.tile(torch.arange(n_verts, device=verts.device), (2, 1)),
            mass_per_vertex,
            size=(n_verts, n_verts),
        ).to(verts.device)
        return mass_matrix


def calc_sum_incident_face_area_per_vertex(
    verts: torch.Tensor, faces: torch.Tensor
) -> torch.Tensor:
    return 3 * calc_barycentric_mass_matrix(
        verts, faces, return_diagonal_only=True
    ).unsqueeze(-1)


########################################### end misc utils

batched_svd = torch.linalg.svd


ARAPEnergyTypeName = Literal["spokes_mine", "spokes_and_rims_mine"]


DeformOptimQuantityName = Literal[
    "verts_offsets",
    "faces_normals",
    "verts_normals",
    "verts_normals_and_scale",
    "faces_3x2rotations",
    "verts_3x2rotations",
    "faces_jacobians",
    "verts_jacobians",
]
DeformSolveMethodName = Optional[Literal["arap", "njfpoisson"]]

PostprocessAfterSolveName = Literal[
    "recenter_rescale", "recenter_only", "recenter_components"
]


@dataclass(slots=True)
class MeshesPackedIndexer:
    padded_aranges: torch.Tensor
    """ (n_meshes_in_batch, max(num_per_mesh)) """
    num_per_mesh: torch.Tensor
    """ (n_meshes_in_batch,) """
    mesh_to_packed_first_idx: torch.Tensor
    """ (n_meshes_in_batch,1) """

    @classmethod
    def from_num_per_mesh(cls, num_per_mesh: torch.Tensor):
        """
        num_per_mesh must be 1D tensor of positive ints, indicating the number
        of packed elements per mesh in the dataset
        """
        assert not num_per_mesh.is_floating_point()
        assert num_per_mesh.ndim == 1
        packed_sz = int(num_per_mesh.sum().item())
        mesh_to_packed_first_idx = torch.cumsum(num_per_mesh, dim=0) - num_per_mesh
        # there is a point to the -999999999999; it's so that it hopefully never turns
        # positive when we add mesh_to_verts_packed_first_idx to it for __call__
        return cls(
            padded_aranges=torch.stack(
                tuple(
                    nn.functional.pad(
                        torch.arange(_n := int(n.item())),
                        (0, packed_sz - _n),
                        value=-999999999999,
                    )
                    for n in num_per_mesh
                ),
                dim=0,
            ).to(num_per_mesh.device),
            num_per_mesh=num_per_mesh,
            mesh_to_packed_first_idx=mesh_to_packed_first_idx.unsqueeze(-1),
        )

    @classmethod
    def from_meshes(
        cls, meshes: Meshes, quantity_defined_on: Literal["verts", "faces", "edges"]
    ):
        if quantity_defined_on == "verts":
            num_per_mesh = meshes.num_verts_per_mesh()
        elif quantity_defined_on == "faces":
            num_per_mesh = meshes.num_faces_per_mesh()
        elif quantity_defined_on == "edges":
            num_per_mesh = meshes.num_edges_per_mesh()
        else:
            raise ValueError(
                "unknown quantity_defined_on: use 'verts' or 'faces' or 'edges'"
            )
        return cls.from_num_per_mesh(num_per_mesh)

    def __call__(self, mesh_indices: Union[int, slice, list, tuple, torch.Tensor]):
        """
        returns a tensor you can use to index dim0 of the associated packed qty.
        >>> verts_packed_idxr = MeshesPackedIndexer.from_meshes(meshes, quantity_defined_on="verts")
        >>> verts_packed[verts_packed_idxr(index)]  # will be the verts_packed of meshes[index]
        """
        picked = (self.padded_aranges + self.mesh_to_packed_first_idx)[mesh_indices]
        return picked[picked >= 0]

    def __getitem__(self, mesh_indices: Union[int, slice, list, tuple, torch.Tensor]):
        """
        returns a new indexer that works for the packed quantity associated with the
        mesh subbatch at mesh_indices
        """
        new_num_per_mesh = self.num_per_mesh[mesh_indices]
        new_mesh_to_packed_first_idx = (
            torch.cumsum(new_num_per_mesh, dim=0) - new_num_per_mesh
        )
        return __class__(
            padded_aranges=self.padded_aranges[mesh_indices],
            num_per_mesh=self.num_per_mesh[mesh_indices],
            mesh_to_packed_first_idx=new_mesh_to_packed_first_idx,
        )

    def n_meshes_in_batch(self) -> int:
        return self.num_per_mesh.size(0)


def calc_rot_matrices_axisangle(
    rot_source_vectors: torch.Tensor, rot_target_vectors: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """
    rot_source_vectors and rot_target_vectors of shape (n, 3), *assumed to be unit normal*!!
    returns rotation matrices (n, 3, 3)
    """
    # a better way to find axis-angle rotation matrices from source to target. (we're not
    # doing procrustes like ARAP here... so this is FARAP in the words from the normal
    # analogies paper; face-only, no 'overlapping cells' i.e. 'spokes and rims' in use)

    # this is adapted from https://iquilezles.org/articles/noacos/ this method avoids not
    # only trig (which i also avoided in my code), but also sqrt and clamp and normalize,
    # etc. Assumes rot_source_vectors and rot_target_vectors are already unit vectors (which
    # is indeed my case, of shape (n,3).
    z = rot_source_vectors
    d = rot_target_vectors

    # z (source): (n, 3), d (target): (n, 3)
    v = torch.linalg.cross(z, d)  # (n,); this is the rotation axes
    c = (z * d).sum(dim=-1)  # (n,); this is the cosine angle btwn source and target
    k = torch.reciprocal(1.0 + c + epsilon)[:, None, None]  # (n, 1, 1)
    vx = v[:, 0]  # (n,)
    vy = v[:, 1]  # (n,)
    vz = v[:, 2]  # (n,)
    rot_matrices = v.unsqueeze(1) * v.unsqueeze(2) * k + torch.stack(
        (
            torch.stack((c, -vz, vy), dim=-1),
            torch.stack((vz, c, -vx), dim=-1),
            torch.stack((-vy, vx, c), dim=-1),
        ),
        dim=1,
    )
    return rot_matrices


@dataclass(slots=True)
class MeshConnectedComponent:
    face_idxs: torch.Tensor
    vert_idxs: torch.Tensor

    @classmethod
    def get_connected_components(cls, faces: torch.Tensor):
        def _group(values: torch.Tensor):
            sorted_values, indices = torch.sort(values)
            nondupe = torch.cat(
                [
                    torch.tensor([True], dtype=torch.bool, device=values.device),
                    sorted_values[1:] != sorted_values[:-1],
                ]
            )
            nondupe_indices = torch.cumsum(nondupe, dim=0) - 1
            counts = torch.bincount(nondupe_indices)
            groups = torch.split(indices, counts.tolist())
            return groups

        _, labels_np = igl.facet_components(faces.cpu().detach().numpy())
        labels = torch.from_numpy(labels_np)
        component_face_idxses = _group(labels.to(faces))
        components = tuple(
            (
                vert_idxs_thiscomponent := torch.unique_consecutive(
                    torch.sort(faces[face_idxs_thiscomponent].view(-1)).values
                ),
                cls(face_idxs_thiscomponent, vert_idxs_thiscomponent),
            )[-1]
            for face_idxs_thiscomponent in component_face_idxses
        )
        return components


def pin_at_least_1_vertex_each_connected_component(
    verts_pinmask: Optional[torch.Tensor],
    connected_components: Sequence[MeshConnectedComponent],
    faces: torch.Tensor,
    verts: torch.Tensor,
    pin_special_vertex: NameOfSpecialVertexToPin,
) -> torch.Tensor:
    """returns a new pinmask to guarantee each component has at least 1 pinned vertex"""
    if verts_pinmask is None:
        verts_pinmask_new = torch.zeros(
            verts.size(0), dtype=torch.bool, device=faces.device
        )
    else:
        verts_pinmask_new = verts_pinmask.clone()

    for component in connected_components:
        vert_idxs_thiscomponent = component.vert_idxs
        if not torch.any(verts_pinmask_new[vert_idxs_thiscomponent]):
            # pin_idx indexes into vert_idxs_thiscomponent
            if pin_special_vertex == "vertex_0":
                pin_idx = 0
            elif pin_special_vertex == "min_x":
                pin_idx = torch.argmin(verts[vert_idxs_thiscomponent, 0])
            elif pin_special_vertex == "min_y":
                pin_idx = torch.argmin(verts[vert_idxs_thiscomponent, 1])
            elif pin_special_vertex == "min_z":
                pin_idx = torch.argmin(verts[vert_idxs_thiscomponent, 2])
            elif pin_special_vertex == "max_x":
                pin_idx = torch.argmax(verts[vert_idxs_thiscomponent, 0])
            elif pin_special_vertex == "max_y":
                pin_idx = torch.argmax(verts[vert_idxs_thiscomponent, 1])
            elif pin_special_vertex == "max_z":
                pin_idx = torch.argmax(verts[vert_idxs_thiscomponent, 2])
            else:
                raise InvalidConfigError("unknown pin_special_vertex")

            verts_pinmask_new[vert_idxs_thiscomponent[pin_idx]] = True
    return verts_pinmask_new


def cholespy_solve(
    solver: Union[cholespy.CholeskySolverF, cholespy.CholeskySolverD], rhs: torch.Tensor
) -> torch.Tensor:
    """
    rhs must be of shape (n_rows, k) where k > 0 and k <= 128 (limitation of cholespy)
    n_rows must be the same n_rows used to initialize the solver
    """
    dtype = torch.float if isinstance(solver, cholespy.CholeskySolverF) else torch.double
    assert rhs.dtype == dtype, (
        f"rhs dtype needs to match the solver type {solver.__class__.__qualname__} (F=float, D=double)"
    )
    if rhs.ndim == 2:
        # needs contiguous() otherwise cholespy's nanobind will throw a mysterious error
        # about unsupported input types
        rhs = rhs.contiguous()
        out = torch.zeros_like(rhs)
        solver.solve(rhs, out)
        return out
    else:
        raise ValueError("rhs has an unsupported number of dimensions")


class CholespySymmetricSolve_AutogradFn(torch.autograd.Function):
    """
    based on, and simplified from, Neural Jacobian Fields's SPLUSolveLayer
    """

    @staticmethod
    def forward(
        ctx,
        solver: Union[cholespy.CholeskySolverF, cholespy.CholeskySolverD],
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        ctx.solver = solver
        return cholespy_solve(solver, rhs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        forward() is intended for symmetric matrices (e.g. laplacians) so the
        backward is just this; otherwise backward() would require transposing
        the system matrix before doing the solve on the grad_output
        """
        grad = cholespy_solve(ctx.solver, grad_output)
        # because forward() needed 2 arguments, we must also return two things
        return None, grad


@dataclass
class PinnedVertsAndRemovedLColumns:
    """
    pinned verts info for a single mesh (not a batch of meshes), associated with
    the mesh's laplacian solver
    """

    verts_pinmask: torch.Tensor
    """ (n_verts,) bool tensor, True where vertex is pinned/column is removed """
    removed_L_columns: torch.Tensor
    """
    (n_verts, n_pinned_verts), matrix of columns removed from the original laplacian
    n_pinned_verts is n_removed_L_columns, each column corresponding to one pinned vertex
    """

    @cached_property
    def verts_freemask(self) -> torch.Tensor:
        return ~self.verts_pinmask

    def to(self, device: torch.device):
        return __class__(
            verts_pinmask=self.verts_pinmask.to(device),
            removed_L_columns=self.removed_L_columns.to(device),
        )

    def adjust_rhs_for_solving_L_with_removed_rowcols(
        self, rhs: torch.Tensor, verts: torch.Tensor
    ) -> torch.Tensor:
        """
        rhs is (n_verts, 3), no padding allowed. (i.e. must match the number of
        verts saved in the solver for this mesh)
        verts may have padding rows after the first n_verts rows.
        returns (n_free_verts, 3)
        """

        # grab verts[pin_verts_indices] (n_removed_columns, 3)
        # and removed_L_columns has shape (n_verts, n_removed_columns)
        # just matmul them to get (n_verts, 3) and then subtract from rhs
        pinmask = self.verts_pinmask
        freemask = self.verts_freemask
        n_verts = rhs.size(0)
        return (rhs - self.removed_L_columns.mm(verts[:n_verts][pinmask]))[freemask]

    def patch_solution_into_verts(
        self,
        soln: torch.Tensor,
        verts: torch.Tensor,
        handle_nans=False,
    ) -> torch.Tensor:
        """
        soln is (n_free_verts, 3).
        verts may have padding rows after the first n_verts rows.
        returns (len(verts), 3), preserving the padding if any.
        """
        ret = verts.clone()
        ret[: self.verts_pinmask.size(0)][self.verts_freemask] = soln
        if handle_nans:
            ret_is_nan = ret.isnan()
            ret[ret_is_nan] = verts[ret_is_nan]
        return ret


class CholeskySolver_ForSingleMeshWithLooseParts:
    connected_components: Sequence[MeshConnectedComponent]
    component_Lpin_idxses_and_solvers: Sequence[
        Tuple[torch.Tensor, Union[cholespy.CholeskySolverF, cholespy.CholeskySolverD]]
    ]

    def __init__(
        self,
        Lpin: torch.Tensor,
        n_verts_before_pin: int,
        faces: torch.Tensor,
        connected_components: Sequence[MeshConnectedComponent],
        verts_pinmask: Optional[torch.Tensor],
        solver_type: Union[Type[cholespy.CholeskySolverF], Type[cholespy.CholeskySolverD]],
    ):
        if verts_pinmask is not None:
            orig_v_idx_to_Lpin_rowcol_idx = torch.full(
                (n_verts_before_pin,), -1, dtype=faces.dtype, device=faces.device
            )
            orig_v_idx_to_Lpin_rowcol_idx[~verts_pinmask] = torch.arange(
                Lpin.size(0),
                dtype=orig_v_idx_to_Lpin_rowcol_idx.dtype,
                device=orig_v_idx_to_Lpin_rowcol_idx.device,
            )
        else:
            orig_v_idx_to_Lpin_rowcol_idx = torch.arange(
                n_verts_before_pin, dtype=faces.dtype, device=faces.device
            )

        self.component_Lpin_idxses_and_solvers = []
        self.connected_components = connected_components
        thlog.debug(f"n components: {len(self.connected_components)}")

        for i, component in enumerate(self.connected_components):
            vert_idxs_thiscomponent = component.vert_idxs
            Lpin_idxs_thiscomponent = orig_v_idx_to_Lpin_rowcol_idx[vert_idxs_thiscomponent]
            Lpin_idxs_thiscomponent = Lpin_idxs_thiscomponent[Lpin_idxs_thiscomponent >= 0]
            Lpin_rowcol_removemask_thiscomponent = torch.ones(
                Lpin.size(0), dtype=torch.bool, device=faces.device
            )
            Lpin_rowcol_removemask_thiscomponent[Lpin_idxs_thiscomponent] = False
            Lpin_this_component = remove_rowcols_from_square_sparse_coo_matrix(
                Lpin, Lpin_rowcol_removemask_thiscomponent
            ).coalesce()
            Lpin_this_component__indices = Lpin_this_component.indices()
            solver = solver_type(
                Lpin_this_component.size(0),
                Lpin_this_component__indices[0],
                Lpin_this_component__indices[1],
                Lpin_this_component.values(),
                cholespy.MatrixType.COO,
            )
            thlog.trace(f"- finished component {i}")
            self.component_Lpin_idxses_and_solvers.append((Lpin_idxs_thiscomponent, solver))

    def solve_with_pvrLc_adjusted_rhs(self, adjusted_rhs: torch.Tensor) -> torch.Tensor:
        soln = torch.zeros_like(adjusted_rhs)
        for Lpin_idxs_thiscomponent, solver in self.component_Lpin_idxses_and_solvers:
            adjusted_rhs_thiscomponent = adjusted_rhs[Lpin_idxs_thiscomponent]
            soln_thiscomponent = cast(
                torch.Tensor,
                CholespySymmetricSolve_AutogradFn.apply(solver, adjusted_rhs_thiscomponent),
            )
            soln[Lpin_idxs_thiscomponent] = soln_thiscomponent
        return soln


def remove_rowcols_from_square_sparse_coo_matrix(
    x: torch.Tensor, rowcol_removemask: torch.Tensor
) -> torch.Tensor:
    assert x.ndim == 2 and (n := x.size(0)) == x.size(1)
    assert rowcol_removemask.ndim == 1
    rowcol_keepmask = ~rowcol_removemask
    x = x.coalesce()
    indices = x.indices()  # (2, nnz)
    data = x.values()
    # mask_of_items_to_keep = ~(torch.isin(indices, removemask).any(dim=0))
    mask_of_items_to_keep = rowcol_keepmask[indices].all(dim=0)
    new_n = int(rowcol_keepmask.count_nonzero().item())
    # repair new indices to fill in holes left behind...
    old2newidx = torch.arange(n, device=x.device)
    old2newidx[rowcol_keepmask] = torch.arange(new_n, device=x.device)
    new_data = data[mask_of_items_to_keep]
    new_indices = old2newidx[indices[:, mask_of_items_to_keep]]
    return torch.sparse_coo_tensor(
        indices=new_indices,
        values=new_data,
        size=(new_n, new_n),
        dtype=x.dtype,
        device=x.device,
    )


def calc_cot_laplacian_for_solver(
    verts: torch.Tensor, faces: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """
    calculate the cot laplacian suitable for solving poisson/arap equations for one mesh
    (verts and faces should not be packed from a batch of more than 1 mesh!)

    coming from pytorch3d's cot laplacian, we need to do a subtraction of the rowsum
    followed by negation in order to obtain the cot laplacian suitable for solving. This
    correct cot laplacian can also be obtained by `grad.T @ mass @ grad`, or more
    specifically, since "mass" here is a (F*3coords,F*3coords) diagonal matrix where the
    diag is filled with the corresp. face's double-areas,
    >>> grad, _, face_doubleareas = calc_gradient_operator(verts, faces)
    >>> lap = (face_doubleareas.repeat_interleave((n_coords := 3)).unsqueeze(-1) * grad).t().mm(grad)
    """
    L_this_mesh, _ = pt3d_ops.cot_laplacian(verts, faces, eps=eps)
    L_this_mesh = (L_this_mesh - make_sparse_diag(L_this_mesh.sum(dim=0).to_dense())).neg()
    return L_this_mesh


def calc_cot_laplacian_and_cholespy_solver_until_it_works(
    verts: torch.Tensor,
    faces: torch.Tensor,
    verts_pinmask: Optional[torch.Tensor],
    ensure_each_connected_component_has_at_least_1_pinned_vertex: Optional[
        NameOfSpecialVertexToPin
    ],
    max_n_attempts: int = 15,
) -> Tuple[
    torch.Tensor,
    CholeskySolver_ForSingleMeshWithLooseParts,
    Optional[PinnedVertsAndRemovedLColumns],
]:
    """
    (this operates on a single mesh, not a batch of meshes)
    """
    rng = np.random.default_rng(seed=398380)

    def __fuzz_rot_verts(_verts: torch.Tensor) -> torch.Tensor:
        rot_axisangle = torch.zeros_like(_verts)
        angles = rng.random(size=(3,), dtype=np.float32) * 360
        thlog.trace(f"fuzz rotation: {angles}")
        rot_axisangle[:] = torch.from_numpy(np.deg2rad(angles))
        rot_mats = pt3d_transforms.axis_angle_to_matrix(rot_axisangle)
        rotated_verts = rot_mats.bmm(_verts.unsqueeze(-1)).squeeze(-1)
        return rotated_verts

    connected_components = MeshConnectedComponent.get_connected_components(faces)

    if pin_special_vertex := ensure_each_connected_component_has_at_least_1_pinned_vertex:
        verts_pinmask = pin_at_least_1_vertex_each_connected_component(
            verts_pinmask, connected_components, faces, verts, pin_special_vertex
        )

    for attempt_i in range(max_n_attempts):
        if attempt_i == 0:
            # on the first attempt, use the original verts, don't fuzz yet
            verts_for_lap_compute = verts
        else:
            verts_for_lap_compute = __fuzz_rot_verts(verts)

        L = calc_cot_laplacian_for_solver(verts_for_lap_compute, faces).coalesce()

        # Lpin is what we use to init the solver, we'll chop off the rhs's index 0 upon solve
        if verts_pinmask is not None:
            verts_pinmask = verts_pinmask.to(L.device)
            Lpin = remove_rowcols_from_square_sparse_coo_matrix(L, verts_pinmask).coalesce()
        else:
            Lpin = L.coalesce()

        try:
            if verts_pinmask is not None:
                pinned_vert_idxs = verts_pinmask.nonzero().view(-1)
                pvrLc = PinnedVertsAndRemovedLColumns(
                    verts_pinmask=verts_pinmask,
                    removed_L_columns=L.index_select(
                        1, pinned_vert_idxs.to(L.device)
                    ).to_dense(),
                )
            else:
                pinned_vert_idxs = None
                pvrLc = None

            with misc_helpers.stdout_redirected():
                # solver is very noisy about not-posdef, which is what we're trying to catch!
                solver = CholeskySolver_ForSingleMeshWithLooseParts(
                    Lpin,
                    verts.size(0),
                    faces,
                    connected_components,
                    verts_pinmask,
                    cholespy.CholeskySolverF,
                )
            if attempt_i > 0:
                thlog.debug(f"[cholespy solver init] okay that worked")

            return L, solver, pvrLc
        except ValueError:
            # most likely failed with not-positive-definite error
            # continue the loop...
            thlog.debug(
                f"[cholespy solver init] failed attempt {attempt_i + 1}, retrying by rotating the mesh and recomputing laplace operator"
            )
            pass
    # if code gets here, we've exhausted attempts, give up
    raise ValueError(
        f"after {max_n_attempts}, couldn't successfully initialize cholespy's cholesky solver for this mesh.\n"
        "(If your mesh has multiple connected components, make sure each component has at least one pinned vertex!)"
    )


@dataclass(slots=True)
class ProcrustesPrecompute:
    padded_cell_edges_per_vertex_packed: torch.Tensor
    """
    (n_verts_packed, max_cell_neighborhood_n_edges, 2) int tensor; last dim contains edge vertex indices.
    negative ints are padding
    """
    covar_lefts_packed: torch.Tensor
    """
    (n_verts_packed, 3, max_cell_neighborhood_n_edges + 1)
    which is found by a batch matmul between
    (n_verts_packed, 3, max_cell_neighborhood_n_edges + 1) bmm (n_verts_packed, max_cell_neighborhood_n_edges + 1,max_cell_neighborhood_n_edges + 1)

    left-multiplies with a (max_cell_neighborhood_n_edges + 1, 3) matrix
    which is formed by grabbing the edge vectors corresponding to pcepv_packed, which
    would be (n_verts_packed, max_cell_neighborhood_n_edges,3) concatenated with
    the target normals (with dim1 unsqueezed so with shape (n_verts_packed, 1,
    3)) in dim1.

    then we batch_svd solve this (n_verts_packed, 3, 3) matrix to get the rotation
    """
    _verts_packed_idxr: MeshesPackedIndexer
    _num_verts_per_mesh: torch.Tensor
    _mesh_to_verts_packed_first_idx: torch.Tensor

    @classmethod
    def from_meshes(
        cls,
        local_step_procrustes_lambda: float,
        arap_energy_type: Optional[ARAPEnergyTypeName],
        laplacians_solvers: "SparseLaplaciansSolvers",
        patient_meshes: Meshes,
    ):
        """
        (need the laplacians solvers just for the laplacian weights)
        """
        thlog.info("Calculating procrustes solve precomputation")
        verts_packed = patient_meshes.verts_packed()

        n_verts_packed = len(verts_packed)
        pcepv_packed: Tuple[Set[Tuple[int, int]], ...] = tuple(
            set() for _ in range(n_verts_packed)
        )
        need_spokes_and_rims = arap_energy_type == "spokes_and_rims_mine"
        for v0i_, v1i_, v2i_ in patient_meshes.faces_packed():
            v0i = int(v0i_.item())
            v1i = int(v1i_.item())
            v2i = int(v2i_.item())

            # correct procrustes neighborhood with directed edges, and each face
            # only contributing the edges that go in its CCW orientation
            e01i = (v0i, v1i)
            e12i = (v1i, v2i)
            e20i = (v2i, v0i)

            pcev0_set = pcepv_packed[v0i]
            pcev1_set = pcepv_packed[v1i]
            pcev2_set = pcepv_packed[v2i]

            # add spokes (radiating from vertex)
            pcev0_set.add(e01i)
            pcev1_set.add(e12i)
            pcev2_set.add(e20i)
            if need_spokes_and_rims:
                # other face-edge pointing to vertex
                pcev0_set.add(e20i)
                pcev1_set.add(e01i)
                pcev2_set.add(e12i)
                # rims
                pcev0_set.add(e12i)
                pcev1_set.add(e20i)
                pcev2_set.add(e01i)

        cell_neighborhood_n_edges = tuple(map(len, pcepv_packed))
        max_cell_neighborhood_n_edges = max(cell_neighborhood_n_edges)
        f: Callable[[Tuple[Set[Tuple[int, int]], int]], Tuple[Tuple[int, int], ...]] = (
            lambda _tup: (
                _set := _tup[0],
                _setlen := _tup[1],
                (
                    tuple(_set)
                    + tuple(
                        (-1, -1) for _ in range(max_cell_neighborhood_n_edges - _setlen)
                    )
                )
                if _setlen < max_cell_neighborhood_n_edges
                else tuple(_set),
            )[-1]
        )
        z = zip(pcepv_packed, cell_neighborhood_n_edges)
        pcepv_packed_tuples = tuple(map(f, z))
        padded_cell_edges_per_vertex_packed = torch.tensor(
            pcepv_packed_tuples, device=patient_meshes.device
        )
        thlog.debug("[procrustes precompute] done padded_cell_edges_per_vertex")
        ######################################## done computing padded_cell_edges_per_vertex

        cell_laplacian_weights_list = []
        for L, verts_packed_first_idx, n_verts in zip(
            laplacians_solvers.Ls,
            patient_meshes.mesh_to_verts_packed_first_idx(),
            patient_meshes.num_verts_per_mesh(),
        ):
            pcepv_this_mesh = (
                padded_cell_edges_per_vertex_packed[
                    verts_packed_first_idx : verts_packed_first_idx + n_verts
                ]
                - verts_packed_first_idx
            )
            pcepv_v0i_this_mesh = pcepv_this_mesh[:, :, 0]
            pcepv_v1i_this_mesh = pcepv_this_mesh[:, :, 1]
            pcepv_shape = pcepv_v1i_this_mesh.shape
            # cell_laplacian_weights_this_mesh = index_sparse_coo_matrix_rowcol(
            #     L, pcepv_v0i_this_mesh.flatten(), pcepv_v1i_this_mesh.flatten()
            # ).view(pcepv_shape)
            # ^ this runs out of mem on my laptop!
            # let's chunk this operation
            pcepv_v0i_numel = pcepv_v0i_this_mesh.numel()
            cell_laplacian_weights_this_mesh = torch.zeros(
                (pcepv_v0i_numel,), device=L.device, dtype=L.dtype
            )
            # each chunk fills the laplacian weights array for SPLITSZ edges
            SPLITSZ = 8192
            for chunk_idxs, v0idxs_this_chunk, v1idxs_this_chunk in zip(
                torch.arange(pcepv_v0i_numel).split(SPLITSZ),
                pcepv_v0i_this_mesh.flatten().split(SPLITSZ),
                pcepv_v1i_this_mesh.flatten().split(SPLITSZ),
            ):
                # dense indexing is way faster so we fetch the rows sparsely and index their columns densely
                L_v0idxs_this_chunk = L.index_select(0, v0idxs_this_chunk).to_dense()
                cell_laplacian_weights_this_mesh[chunk_idxs] = L_v0idxs_this_chunk[
                    torch.arange(v1idxs_this_chunk.size(-1)), v1idxs_this_chunk
                ]
                ## the older way:
                # cell_laplacian_weights_this_mesh[chunk_idxs] = (
                #     index_sparse_coo_matrix_rowcol(L, v0idxs_this_chunk, v1idxs_this_chunk)
                # )

            cell_laplacian_weights_this_mesh = cell_laplacian_weights_this_mesh.view(
                pcepv_shape
            ).to(patient_meshes.device)

            # wherever pcepv is negative, that's padding
            cell_laplacian_weights_this_mesh[pcepv_v1i_this_mesh < 0] = 0
            # ^ (n_verts, max_cell_neighborhood_n_edges,) float
            cell_laplacian_weights_list.append(cell_laplacian_weights_this_mesh)
        cell_laplacian_weights_packed = torch.cat(cell_laplacian_weights_list, dim=0)
        # (n_verts_packed, max_cell_neighborhood_n_edges,)
        thlog.debug("[procrustes precompute] done cotangent weights")
        ######################################## done putting cotan weights into pcepv format

        voronoi_verts_massmatrix__scipy = igl.massmatrix(
            verts_packed.cpu().detach().numpy(),
            patient_meshes.faces_packed().cpu().detach().numpy(),
        )
        # get diag of this thing
        voronoi_verts_mass_packed = (
            torch.from_numpy(voronoi_verts_massmatrix__scipy.diagonal())
            .float()
            .to(patient_meshes.device)
        ) * local_step_procrustes_lambda
        # ^ (n_verts_packed)
        assert voronoi_verts_mass_packed.shape == (n_verts_packed,)

        if (voronoi_verts_mass_packed == 0).all():
            raise RuntimeError(
                "igl.massmatrix gave all zeros! are you using a pre-nanobind version of libigl on numpy>2.0? please upgrade libigl, or downgrade numpy to <2.0"
            )

        # form the middle neighborhood-size-by-neighborhood-size matrix
        diags_packed = torch.cat(
            (cell_laplacian_weights_packed, voronoi_verts_mass_packed.unsqueeze(-1)), dim=-1
        )
        # ^ (n_verts_packed, max_cell_neighborhood_n_edges + 1)
        diagmats_packed = torch.diag_embed(diags_packed)
        # ^ (n_verts_packed, max_cell_neighborhood_n_edges+1, max_cell_neighborhood_n_edges+1, )
        thlog.debug("[procrustes precompute] done diagonal matrix")
        if thlog.logguard(LOG_TRACE):
            torch.set_printoptions(precision=1)
            np.set_printoptions(precision=3)
            thlog.trace(f"""
            vertsmass
{voronoi_verts_mass_packed}
            diags:
{diags_packed.cpu().detach().numpy()}
            diagmats
            {diagmats_packed.cpu().detach().numpy()}
            L
            {laplacians_solvers.Ls[0].to_dense().cpu().detach().numpy()}
            pcepv
            {padded_cell_edges_per_vertex_packed}
            """)
        ############################################### done making middle diag matrix

        # NOTE this bit of code is also how you compute the covar_rights matrix
        # for the in-progress edge vecs and target normals (target vert normals taking the place of original_vert_normals)

        pcepv_v1i = padded_cell_edges_per_vertex_packed[:, :, 1]
        pcepv_v0i = padded_cell_edges_per_vertex_packed[:, :, 0]
        original_cell_edge_vecs_packed = verts_packed[pcepv_v1i] - verts_packed[pcepv_v0i]
        # ^ (n_verts_packed, max_cell_neighborhood_n_edges, 3)
        # zero out wherever there is padding
        original_cell_edge_vecs_packed[pcepv_v1i < 0] = 0
        original_vert_normals = patient_meshes.verts_normals_packed().unsqueeze(1)
        # ^ (n_verts_packed, 1, 3)
        covar_lefts_lefts_packed = torch.cat(
            (original_cell_edge_vecs_packed, original_vert_normals), dim=1
        )
        # ^ (n_verts_packed, max_cell_neighborhood_n_edges + 1, 3)
        covar_lefts_packed = covar_lefts_lefts_packed.transpose(-1, -2).bmm(diagmats_packed)
        # ^ (n_verts_packed, 3, max_cell_neighborhood_n_edges + 1)
        thlog.debug("[procrustes precompute] done covariance matrix")

        # make misc indexing bookkeeping
        _num_verts_per_mesh = patient_meshes.num_verts_per_mesh()
        _verts_packed_idxr = MeshesPackedIndexer.from_num_per_mesh(_num_verts_per_mesh)
        return cls(
            padded_cell_edges_per_vertex_packed=padded_cell_edges_per_vertex_packed.to(
                patient_meshes.device
            ),
            covar_lefts_packed=covar_lefts_packed,
            _verts_packed_idxr=_verts_packed_idxr,
            _num_verts_per_mesh=_num_verts_per_mesh,
            _mesh_to_verts_packed_first_idx=patient_meshes.mesh_to_verts_packed_first_idx(),
        )

    def __getitem__(self, mesh_indices: Union[int, List[int], torch.Tensor]):
        new_packed_idxr = self._verts_packed_idxr[mesh_indices]
        pcepv_packed_to_mesh_idx = torch.arange(
            new_packed_idxr.n_meshes_in_batch(),
            device=new_packed_idxr.num_per_mesh.device,
        ).repeat_interleave(new_packed_idxr.num_per_mesh)
        packed_idx = self._verts_packed_idxr(mesh_indices)
        new_pcepv_packed_noadjust = self.padded_cell_edges_per_vertex_packed[packed_idx]

        # apply offset adjustment to the indices inside new_faceadj_noadjust
        new_num_verts_per_mesh = self._num_verts_per_mesh[mesh_indices]
        new_mesh_to_verts_packed_first_idx = (
            torch.cumsum(new_num_verts_per_mesh, dim=0) - new_num_verts_per_mesh
        )
        old_mesh_to_verts_packed_first_idx = self._mesh_to_verts_packed_first_idx[
            mesh_indices
        ]
        new_pcepv_packed_adjusted = (
            new_pcepv_packed_noadjust
            - old_mesh_to_verts_packed_first_idx[pcepv_packed_to_mesh_idx, None, None]
            + new_mesh_to_verts_packed_first_idx[pcepv_packed_to_mesh_idx, None, None]
        )
        return __class__(
            padded_cell_edges_per_vertex_packed=new_pcepv_packed_adjusted,
            covar_lefts_packed=self.covar_lefts_packed[packed_idx],
            _verts_packed_idxr=new_packed_idxr,
            _num_verts_per_mesh=new_num_verts_per_mesh,
            _mesh_to_verts_packed_first_idx=new_mesh_to_verts_packed_first_idx,
        )


def calc_rot_matrices_with_procrustes(
    procrustes_precompute: ProcrustesPrecompute,
    curr_deformed_verts_packed: torch.Tensor,
    target_verts_normals_packed: torch.Tensor,
    verts_selmask_packed: torch.Tensor,
    curr_verts_normals_packed__backupforaxisangle: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    curr_deformed_verts_packed (n_verts_packed, 3)
    target_normals_packed (n_verts_packed, 3), the targeted normals

    curr_verts_normals_packed__backupforaxisangle also (n_verts_packed, 3) only
    used if DRMSH_PROCRUSTES_DEGEN_SVDVALS_BECAREFUL is enabled and problematic
    svd'd results are found that make the grad unstable

    returns desired rot matrices (n_verts_packed, 3, 3),
    AND the covar matrices packed, also of shape (n_verts_packed, 3, 3), that
    was fed to SVD (for SVD grad debugging purposes...)

    verts with verts_pinmask_packed value True will get identity matrix
    """
    pcepv_v1i = procrustes_precompute.padded_cell_edges_per_vertex_packed[:, :, 1]
    pcepv_v0i = procrustes_precompute.padded_cell_edges_per_vertex_packed[:, :, 0]
    pcepv_v1 = curr_deformed_verts_packed[pcepv_v1i]
    pcepv_v0 = curr_deformed_verts_packed[pcepv_v0i]
    current_cell_edge_vecs_packed = pcepv_v1 - pcepv_v0
    # ^ (n_verts_packed, max_cell_neighborhood_n_edges, 3)
    if thlog.guard(VIZ_TRACE, needs_polyscope=True):
        pcepv_v1_for_vertex0 = pcepv_v1[0]
        pcepv_v1_for_vertex0 = pcepv_v1_for_vertex0[pcepv_v1i[0] >= 0]
        pcepv_v0_for_vertex0 = pcepv_v0[0]
        pcepv_v0_for_vertex0 = pcepv_v0_for_vertex0[pcepv_v0i[0] >= 0]
        cunet_pts = torch.cat((pcepv_v0_for_vertex0, pcepv_v1_for_vertex0), dim=0)
        cunet_edges = torch.stack(
            (
                torch.arange(len(pcepv_v0_for_vertex0)),
                len(pcepv_v0_for_vertex0) + torch.arange(len(pcepv_v0_for_vertex0)),
            ),
            dim=-1,
        )

        thlog.psr.register_curve_network(
            "v0 cell neigh",
            cunet_pts.cpu().detach().numpy(),
            cunet_edges.cpu().detach().numpy(),
        )
    current_cell_edge_vecs_packed[pcepv_v1i < 0] = 0
    # ^ (n_verts_packed, max_cell_neighborhood_n_edges, 3)
    target_verts_normals_packed = target_verts_normals_packed.unsqueeze(1)
    # ^ (n_verts_packed, 1, 3)
    covar_rights_packed = torch.cat(
        (current_cell_edge_vecs_packed, target_verts_normals_packed), dim=1
    )
    covar = procrustes_precompute.covar_lefts_packed.bmm(covar_rights_packed)
    # ^ (n_verts_packed, 3, 3)
    # so far that was  computed assuming vectors are column vecs, so
    # covar = covar.transpose(-1, -2)

    # if allow fallback to axis-angle for inputs that would be problematic for SVD grad
    if curr_verts_normals_packed__backupforaxisangle is not None:
        with torch.no_grad():
            thlog.trace("checking for problematic svd inputs")
            # find matrices that have nearly equal svals (svd grad will be nan)
            # from experience, this shows up between s0 and s1
            svals_packed = torch.linalg.svdvals(covar)
            s0s1diff = svals_packed[:, 0] - svals_packed[:, 1]
            zeros = torch.zeros_like(s0s1diff)
            thlog.trace(f"sval pair diff mins {s0s1diff.min()}")
            # pick out vertices where it'd be problematic to run SVD on its covar mat
            # and run axis-angle instead
            svdbad_idxr = torch.isclose(s0s1diff, zeros, atol=1e-9)
            svdgood_idxr = (~svdbad_idxr).logical_and(verts_selmask_packed)
            # svdbad_idxr should indicate verts to run axisangle; if vert is not selected,
            # then we don't even do that part (we just set their mats to identity)
            svdbad_idxr = svdbad_idxr.logical_and(verts_selmask_packed)
    else:
        svdbad_idxr = None
        svdgood_idxr = verts_selmask_packed

    rots_packed = torch.zeros_like(covar)

    # batch svd this thing (run svd on covar at vertices where it's fine to do so)
    uu, ss, vvt = batched_svd(covar[svdgood_idxr])
    vvt = torch.stack((-vvt[:, 0], -vvt[:, 1], vvt[:, 2]), dim=1)
    rots_packed[svdgood_idxr] = vvt.transpose(-1, -2).bmm(uu.transpose(-1, -2))

    # svd det correction code from https://github.com/OllieBoyne/pytorch-arap/blob/master/pytorch_arap/arap.py
    # for any det(Ri) <= 0
    entries_to_flip = torch.nonzero(
        rots_packed[svdgood_idxr].det() <= 0, as_tuple=False
    ).flatten()
    # ^idxs where det(R) <= 0
    if thlog.logguard(LOG_TRACE):
        thlog.trace(f"entries to flip {entries_to_flip}")
    if len(entries_to_flip) > 0:
        uumod = uu.clone()
        # minimum singular value is the last one
        uumod[entries_to_flip, :, -1] *= -1  # flip cols
        rots_packed[svdgood_idxr][entries_to_flip] = (
            vvt[entries_to_flip]
            .transpose(-1, -2)
            .bmm(uumod[entries_to_flip].transpose(-1, -2))
        )

    ## if this is configured, replace problematic svd'd entries with axis angle rotation
    if svdbad_idxr is not None and svdbad_idxr.any():
        thlog.debug(
            f"finding rot for degen covars at {svdbad_idxr.nonzero().flatten()} with axisangle"
        )
        assert curr_verts_normals_packed__backupforaxisangle is not None
        rots_packed[svdbad_idxr] = calc_rot_matrices_axisangle(
            curr_verts_normals_packed__backupforaxisangle[svdbad_idxr],
            target_verts_normals_packed.squeeze(1)[svdbad_idxr],
            epsilon=1e-6,
        ).transpose(-1, -2)
        # transpose just because we will transpose back at the return
        # (squeeze(1) because previously we unsqueezed(1) for the broadcast)

    ## then, anything not selected gets identity matrix. offdiag is already zero, fill diag with 1
    verts_pinmask_packed = ~verts_selmask_packed
    for i in (0, 1, 2):
        rots_packed[verts_pinmask_packed, i, i] = 1

    return rots_packed.transpose(-1, -2), covar


@dataclass(slots=True)
class SparseLaplaciansSolvers:
    """
    Indexable dataclass holding solvers for a batch of meshes.
    Each mesh may have multiple disjoint connected components.
    """

    Ls: torch.Tensor
    """
    cotangent laplacian,
    sparse_coo of shape (batch_size, max_n_verts, max_n_verts), where laplacians
    of meshes with fewer than max_n_verts verts (implicitly) get zero-padding
    """
    cholespy_solvers: Sequence[CholeskySolver_ForSingleMeshWithLooseParts]
    """
    This custom solver type transparently handles meshes with multiple connected components;
    other operations on L don't need to know about connected components at all
    """
    pin_verts_and_removed_L_columns: Sequence[Optional[PinnedVertsAndRemovedLColumns]]
    """
    for each mesh in batch: a struct holding mask of verts that are pinned, and the
    columns from the laplacian corresponding to them (which were removed from the laplacian)
    to perform the rhs adjustment upon solving
    NOTE that due to pinning settings, this mask may not actually correspond to
    the selection mask since we might be pinning not all the unselected vertices
    but only some of them (for example, picked by furthest-point sampling)
    (although actually, we basically always pin all verts that are nonselected)
    """
    njfpoisson_rhs_lefts: Optional[torch.Tensor] = None
    """
    this is not used for the radmesh method, but is needed for the njfpoisson solve method
    if you'd like to run a comparison.

    if present, sparse of shape `(batch_size, max_n_verts, max_n_faces*3coords)`
    storing the poisson system's rhs matrix to be left-multiplied with per-face
    transform matrices to form the complete right-hand side of a system.

    for each mesh in a batch, njfpoisson_rhs_lefts is computed via
    >>> grad, _, face_doubleareas = calc_gradient_operator(verts, faces)
        # where grad is (n_faces*3coords, n_verts)
        rhs_left = (face_doubleareas.repeat_interleave((n_coords:=3)).unsqueeze(-1) * grad)

    the system to solve is
    >>> L @ X = rhs_left @ per_face_transform_matrices

    (making sure that `per_face_transform_matrices` has been transposed to be
    `(batch, max_n_faces_per_mesh, 3coords, 3vertsperface)` and then viewed as
    `(batch, max_n_faces_per_mesh*3coords, 3vertsperface))` before doing this
    """

    @property
    def connected_components_per_mesh(self) -> Sequence[Sequence[MeshConnectedComponent]]:
        """
        each Sequence[MeshConnectedComponent] describes the connected components
        of a single mesh. a Solvers object pertains to a batch of such meshes,
        hence a sequence of sequence of MeshConnectedComponent.
        """
        return tuple(solver.connected_components for solver in self.cholespy_solvers)

    def __getitem__(self, mesh_indices: Union[int, List[int], torch.Tensor]):
        """
        return the batch item at `index` as its own SparseLaplaciansSolvers of batch size 1
        if `index` is a scalar (int or scalar tensor) or n if `index` is a 1D tensor of n
        index values.
        """
        if isinstance(mesh_indices, int):
            Ls = self.Ls[None, mesh_indices]
            njfpoisson_rhs_lefts = (
                self.njfpoisson_rhs_lefts[None, mesh_indices]
                if self.njfpoisson_rhs_lefts is not None
                else None
            )
            cholespy_solvers = (self.cholespy_solvers[mesh_indices],)
            pin_verts_and_removed_L_columns = (
                self.pin_verts_and_removed_L_columns[mesh_indices],
            )
        else:
            if isinstance(mesh_indices, list):
                mesh_indices = torch.tensor(mesh_indices)  # turn int list into int tensor
            if mesh_indices.ndim == 0:
                # 0D tensor containing just a scalar
                Ls = self.Ls[None, mesh_indices]
                njfpoisson_rhs_lefts = (
                    self.njfpoisson_rhs_lefts[None, mesh_indices]
                    if self.njfpoisson_rhs_lefts is not None
                    else None
                )
                cholespy_solvers = (self.cholespy_solvers[i := int(mesh_indices.item())],)
                pin_verts_and_removed_L_columns = (self.pin_verts_and_removed_L_columns[i],)
            else:
                # actual tensor being used as the index; sparse tensors don't support them
                # in the usual [] indexing syntax, we have to use index_select
                index_on_dev = mesh_indices.to(self.Ls.device)
                Ls = self.Ls.index_select(0, index_on_dev)
                njfpoisson_rhs_lefts = (
                    self.njfpoisson_rhs_lefts.index_select(0, index_on_dev)
                    if self.njfpoisson_rhs_lefts is not None
                    else None
                )
                cholespy_solvers = tuple(
                    self.cholespy_solvers[int(i.item())] for i in mesh_indices
                )
                pin_verts_and_removed_L_columns = tuple(
                    self.pin_verts_and_removed_L_columns[int(i.item())]
                    for i in mesh_indices
                )

        return __class__(
            Ls,
            pin_verts_and_removed_L_columns=pin_verts_and_removed_L_columns,
            cholespy_solvers=cholespy_solvers,
            njfpoisson_rhs_lefts=njfpoisson_rhs_lefts,
        )

    def to(self, device: torch.device) -> "SparseLaplaciansSolvers":
        return __class__(
            self.Ls.to(device),
            pin_verts_and_removed_L_columns=tuple(
                (pvrLc.to(device) if pvrLc is not None else pvrLc)
                for pvrLc in self.pin_verts_and_removed_L_columns
            ),
            cholespy_solvers=self.cholespy_solvers,
            njfpoisson_rhs_lefts=(
                self.njfpoisson_rhs_lefts.to(device)
                if self.njfpoisson_rhs_lefts is not None
                else None
            ),
        )

    @classmethod
    def from_meshes(
        cls,
        meshes: Meshes,
        verts_pinmask_each_mesh: Sequence[Optional[torch.Tensor]],
        ensure_each_connected_component_has_at_least_1_pinned_vertex: Optional[
            NameOfSpecialVertexToPin
        ],
        compute_njfpoisson_rhs_lefts: bool,
    ):
        """
        given a batch of meshes, compute the Laplace operator and a cholespy solver object
        with that operator as the system matrix. Optionally,
        if `compute_njfpoisson_rhs_lefts`: the NJF poisson system right-hand side
        premultiplier (see the docstrings of the fields of this dataclass for more info)

        `verts_pinmask_each_mesh` a sequence of tensors each (n_verts,) bool tensor
        where True indicates to pin that vertex in the solve for the corresponding mesh.
        For each pinned vertex, the laplacian L will have that row and column removed, and
        any solves using this solver must adjust the rhs to this reduced system (with the
        appropriate pinned vertices of each mesh in the batch subbed in) accordingly.
        The adjustment is done using the corresponding saved PinnedVertsAndRemovedLColumns.

        if `ensure_each_connected_component_has_at_least_1_pinned_vertex` is not None,
        it should specify the special vertex to pin for each connected component if
        it doesn't already have a pinned vertex. This is highly recommended to make
        the solver successfully factorize for most meshes.
        """
        max_n_verts_per_mesh = int(meshes.num_verts_per_mesh().max().item())
        max_n_faces_per_mesh = int(meshes.num_faces_per_mesh().max().item())
        square_shape = (max_n_verts_per_mesh, max_n_verts_per_mesh)
        Ls = []
        cholespy_solvers: list[CholeskySolver_ForSingleMeshWithLooseParts] = []
        njfpoisson_rhs_lefts_per_mesh = [] if compute_njfpoisson_rhs_lefts else None
        pin_verts_and_removed_L_columns_per_mesh: List[
            Optional[PinnedVertsAndRemovedLColumns]
        ] = []
        for verts, faces, verts_pinmask in zip(
            meshes.verts_list(), meshes.faces_list(), verts_pinmask_each_mesh
        ):
            L_this_mesh, cholespy_solver, pvrLc = (
                calc_cot_laplacian_and_cholespy_solver_until_it_works(
                    verts,
                    faces,
                    ensure_each_connected_component_has_at_least_1_pinned_vertex=ensure_each_connected_component_has_at_least_1_pinned_vertex,
                    verts_pinmask=verts_pinmask,
                )
            )
            cholespy_solvers.append(cholespy_solver)

            # keep the removed first column of the laplace operator for rhs adjust
            pin_verts_and_removed_L_columns_per_mesh.append(pvrLc)

            # matrix to be left-multiplied with face transforms to form the rhs
            # of a poisson system
            if njfpoisson_rhs_lefts_per_mesh is not None:
                grad, _, face_doubleareas = calc_gradient_operator(verts, faces)
                poisson_rhs_left_this_mesh = (
                    face_doubleareas.repeat_interleave((n_coords := 3)).unsqueeze(-1) * grad
                )
                # ^ this is sparse, with shape (n_faces_this_mesh * n_coords,
                # n_verts_this_mesh). First, we transpose it for quicker application at time
                # of use, since we'll need (V,F*3)...
                poisson_rhs_left_this_mesh = poisson_rhs_left_this_mesh.t()
                # and sparse-resize it to fit padding
                poisson_rhs_left_this_mesh.sparse_resize_(
                    (max_n_verts_per_mesh, max_n_faces_per_mesh * n_coords), 2, 0
                )
                njfpoisson_rhs_lefts_per_mesh.append(poisson_rhs_left_this_mesh)

            # done computing extra matrices for this mesh
            # resize the laplace operator this mesh to be the padded size
            L_this_mesh.sparse_resize_(square_shape, 2, 0)
            Ls.append(L_this_mesh)
        # done looping through meshes

        # stack the matrices into batched tensors
        padded_batched_laplacians = torch.stack(Ls, dim=0)

        njfpoisson_rhs_lefts = (
            torch.stack(njfpoisson_rhs_lefts_per_mesh, dim=0)
            if njfpoisson_rhs_lefts_per_mesh is not None
            else None
        )

        return cls(
            padded_batched_laplacians,
            pin_verts_and_removed_L_columns=pin_verts_and_removed_L_columns_per_mesh,
            cholespy_solvers=cholespy_solvers,
            njfpoisson_rhs_lefts=njfpoisson_rhs_lefts,
        )


def handle_postprocess_after_solve(
    meshes: Meshes,
    connected_components_per_mesh: Sequence[Sequence[MeshConnectedComponent]],
    soln_verts_packed___: torch.Tensor,
    postprocess: Optional[PostprocessAfterSolveName],
) -> torch.Tensor:
    if postprocess == "recenter_rescale":
        soln_verts_packed = recenter_to_centroid_and_rescale_new_verts_to_fit_old_bboxes(
            meshes, soln_verts_packed___
        )
    elif postprocess == "recenter_only":
        soln_verts_packed = recenter_to_centroid(meshes, soln_verts_packed___)
    elif postprocess == "recenter_components":
        soln_verts_packed = recenter_to_centroid_per_component(
            meshes, connected_components_per_mesh, soln_verts_packed___
        )
    else:
        soln_verts_packed = soln_verts_packed___
    return soln_verts_packed


def recenter_to_centroid(meshes: Meshes, new_verts_packed: torch.Tensor) -> torch.Tensor:
    new_verts_packed_recentered = torch.zeros_like(new_verts_packed)
    for verts_packed_first_idx, n_verts in zip(
        meshes.mesh_to_verts_packed_first_idx(), meshes.num_verts_per_mesh()
    ):
        indexer = slice(verts_packed_first_idx, verts_packed_first_idx + n_verts)
        verts_this_mesh = new_verts_packed[indexer]
        verts_this_mesh = verts_this_mesh - verts_this_mesh.mean(dim=0, keepdim=True)
        new_verts_packed_recentered[indexer] = verts_this_mesh
    return new_verts_packed_recentered


def recenter_to_centroid_per_component(
    meshes: Meshes,
    connected_components_per_mesh: Sequence[Sequence[MeshConnectedComponent]],
    new_verts_packed: torch.Tensor,
):
    new_verts_packed_recentered = torch.zeros_like(new_verts_packed)
    old_verts_packed = meshes.verts_packed()
    for verts_packed_first_idx, n_verts, components_this_mesh in zip(
        meshes.mesh_to_verts_packed_first_idx(),
        meshes.num_verts_per_mesh(),
        connected_components_per_mesh,
    ):
        this_mesh_indexer = slice(verts_packed_first_idx, verts_packed_first_idx + n_verts)
        new_verts_this_mesh = new_verts_packed[this_mesh_indexer]
        old_verts_this_mesh = old_verts_packed[this_mesh_indexer]
        for component in components_this_mesh:
            vert_idxs_thiscomponent = component.vert_idxs
            new_verts_thiscomponent = new_verts_this_mesh[vert_idxs_thiscomponent]
            old_verts_thiscomponent = old_verts_this_mesh[vert_idxs_thiscomponent]
            new_verts_packed_recentered[this_mesh_indexer][vert_idxs_thiscomponent] = (
                new_verts_thiscomponent
                - new_verts_thiscomponent.mean(dim=0, keepdim=True)
                + old_verts_thiscomponent.mean(dim=0, keepdim=True)
            )
    return new_verts_packed_recentered


def recenter_to_centroid_and_rescale_new_verts_to_fit_old_bboxes(
    meshes: Meshes, new_verts_packed: torch.Tensor
) -> torch.Tensor:
    old_bboxes = meshes.get_bounding_boxes()
    # (n_meshes, 3, 2) last dim is [min,max]
    old_bboxes_min = old_bboxes[:, :, 0]
    old_bboxes_max = old_bboxes[:, :, 1]
    old_sizes = (old_bboxes_max - old_bboxes_min).norm(dim=-1)  # (n_meshes,)

    new_verts_packed_scaled = torch.zeros_like(new_verts_packed)
    # old_verts_packed = meshes.verts_packed()
    for verts_packed_first_idx, n_verts, old_size in zip(
        meshes.mesh_to_verts_packed_first_idx(), meshes.num_verts_per_mesh(), old_sizes
    ):
        indexer = slice(verts_packed_first_idx, verts_packed_first_idx + n_verts)
        verts_this_mesh = new_verts_packed[indexer]

        # first, recenter to centroid
        verts_this_mesh = verts_this_mesh - verts_this_mesh.mean(dim=0, keepdim=True)
        # then scale factor is the ratio between the bounding box diagonal lengths
        bbox_diag_this_mesh = verts_this_mesh.max(dim=0)[0] - verts_this_mesh.min(dim=0)[0]
        new_size = bbox_diag_this_mesh.norm()
        new_verts_packed_scaled[indexer] = verts_this_mesh * (old_size / new_size)
    return new_verts_packed_scaled


def index_sparse_coo_matrix_rowcol(
    x: torch.Tensor, row_idxs: torch.Tensor, col_idxs: torch.Tensor
) -> torch.Tensor:
    """
    indexes a 2D sparse_coo matrix with row indices and column indices behaving like
    x[row_idxs, col_idxs] as if x were a dense 2D matrix (without needing to_dense())
    """
    assert x.ndim == 2
    idx_selected = x.index_select(0, row_idxs).index_select(1, col_idxs).coalesce()
    idx_selected_rows, idx_selected_cols = idx_selected.indices()
    return idx_selected.values()[idx_selected_rows == idx_selected_cols]


def calc_ARAP_global_solve(
    meshes: Meshes,
    laplacians_solvers: SparseLaplaciansSolvers,
    per_vertex_rot_matrices_packed: torch.Tensor,
    arap_energy_type: ARAPEnergyTypeName,
    postprocess: Optional[PostprocessAfterSolveName],
) -> torch.Tensor:
    """
    per_vertex_rot_matrices_packed: shape (n_verts_packed, 3, 3)
    returns deformed vertices (n_verts_packed, 3)
    """
    # the rest of this function here computes rhs directly from the 2004 paper formula
    # if arap_energy_type == spokes_mine, or the rhs from the spokes-and-rims energy
    # from Chao et al 2011 (also used in normal analogies; formula described in CGAL docs)
    # the rhs to find has shape (n_verts, 3)
    solutions = []
    for i, (L, verts_padded, faces, n_verts_this_mesh, verts_packed_first_idx) in enumerate(
        zip(
            laplacians_solvers.Ls,
            meshes.verts_padded(),
            meshes.faces_list(),
            meshes.num_verts_per_mesh(),
            meshes.mesh_to_verts_packed_first_idx(),
        )
    ):
        # it might be possible that verts_padded for this mesh has shorter dim0 length than
        # L because the meshes might have been indexed from a larger meshes batch, with
        # padding shrunken to fit just the largest mesh in the extracted batch. in this
        # case, we expand verts with padding to match the dim0 and dim1 size of the square L
        if (vp_sz0 := verts_padded.size(0)) < (L_sz0 := L.size(0)):
            verts_padded = nn.functional.pad(verts_padded, (0, 0, 0, L_sz0 - vp_sz0))

        # for each edge between a vertex i and vertex j, compute (w_ij / 2) * ((R_i
        # + R_j) @ (p_i - p_j)) (this is a 3d point)
        L = L.coalesce()

        if arap_energy_type == "spokes_mine":
            L_sp_indices = L.indices()
            dir_edges_vi = L_sp_indices[1]
            dir_edges_vj = L_sp_indices[0]
            dir_edges_weight = L.values()

            # Ri + Rj
            rot_vi_plus_rot_vj = (
                per_vertex_rot_matrices_packed[dir_edges_vi + verts_packed_first_idx]
                + per_vertex_rot_matrices_packed[dir_edges_vj + verts_packed_first_idx]
            )

            # pi - pj
            pi_minus_pj = verts_padded[dir_edges_vi] - verts_padded[dir_edges_vj]

            # (w_ij / 2) * ((R_i + R_j) @ (p_i - p_j))
            rhs_per_dir_edge = (dir_edges_weight / 2).unsqueeze(1) * rot_vi_plus_rot_vj.bmm(
                pi_minus_pj.unsqueeze(-1)
            ).squeeze(-1)

            # the rhs vector is the same shape as verts_padded; then each slot corresponding to
            # vertex index j in the rhs vector is the sum of the values of the directed edges
            # out of vertex j. Here we use j because it corresponded to L_sp_indices[0]; if we
            # use i, then we still get the right system soln but the y axis is flipped
            # rhs = torch.index_add(torch.zeros_like(verts), 0, dir_edges_vj, rhs_per_dir_edge)
            rhs = torch.index_put(
                torch.zeros_like(verts_padded),
                (dir_edges_vj,),
                rhs_per_dir_edge,
                accumulate=True,
            )
            # for this, index_put gives essentially the same result as index_add
            # there, but is not undefined behavior on duplicate indices, unlike index_add
        elif arap_energy_type == "spokes_and_rims_mine":
            faces_v0idx = faces[:, 0]
            faces_v1idx = faces[:, 1]
            faces_v2idx = faces[:, 2]
            v0 = verts_padded[faces_v0idx]
            v1 = verts_padded[faces_v1idx]
            v2 = verts_padded[faces_v2idx]
            r0 = per_vertex_rot_matrices_packed[faces_v0idx + verts_packed_first_idx]
            r1 = per_vertex_rot_matrices_packed[faces_v1idx + verts_packed_first_idx]
            r2 = per_vertex_rot_matrices_packed[faces_v2idx + verts_packed_first_idx]
            w01 = index_sparse_coo_matrix_rowcol(L, faces_v0idx, faces_v1idx)[:, None, None]
            w12 = index_sparse_coo_matrix_rowcol(L, faces_v1idx, faces_v2idx)[:, None, None]
            w20 = index_sparse_coo_matrix_rowcol(L, faces_v2idx, faces_v0idx)[:, None, None]

            e01 = (v1 - v0).unsqueeze(-1)
            e02 = (v2 - v0).unsqueeze(-1)
            e12 = (v2 - v1).unsqueeze(-1)
            e10 = -e01
            e20 = -e02
            e21 = -e12

            onethird = 1 / 3
            rs = r0 + r1 + r2
            w01rs = (w01 / 2) * rs
            w20rs = (w20 / 2) * rs
            w12rs = (w12 / 2) * rs
            v0_contrib = onethird * (w01rs.bmm(e01) + w20rs.bmm(e02)).squeeze(-1)
            v1_contrib = onethird * (w12rs.bmm(e12) + w01rs.bmm(e10)).squeeze(-1)
            v2_contrib = onethird * (w20rs.bmm(e20) + w12rs.bmm(e21)).squeeze(-1)
            v0v1v2idxs = torch.cat((faces_v0idx, faces_v1idx, faces_v2idx), dim=0)
            rhs_contribs = torch.cat((v0_contrib, v1_contrib, v2_contrib), dim=0)
            rhs = torch.index_put(
                torch.zeros_like(verts_padded), (v0v1v2idxs,), rhs_contribs, accumulate=True
            )
        else:
            raise AssertionError(
                f"shouldn't use calc_ARAP_global_solve with this arap_energy_type setting: {arap_energy_type} (if it's an igl arap energy type, maybe I forgot to init the solvers with igl_arap_rhs_lefts)"
            )

        # now do the solve
        solver = laplacians_solvers.cholespy_solvers[i]
        pvrLc = laplacians_solvers.pin_verts_and_removed_L_columns[i]
        adjusted_rhs = (
            pvrLc.adjust_rhs_for_solving_L_with_removed_rowcols(
                rhs[:n_verts_this_mesh], verts_padded
            )
            if pvrLc
            else rhs
        )

        soln = solver.solve_with_pvrLc_adjusted_rhs(adjusted_rhs)
        soln = (
            pvrLc.patch_solution_into_verts(soln, verts_padded[:n_verts_this_mesh])
            if pvrLc
            else soln
        )
        solutions.append(soln)

    soln_verts_packed = torch.cat(solutions, dim=0)  # (n_verts_packed,3)
    soln_verts_packed = handle_postprocess_after_solve(
        meshes,
        laplacians_solvers.connected_components_per_mesh,
        soln_verts_packed,
        postprocess,
    )

    return soln_verts_packed


def calc_njf_poisson_global_solve(
    meshes: Meshes,
    laplacians_solvers: SparseLaplaciansSolvers,
    faces_matrices_packed: torch.Tensor,
    postprocess: Optional[PostprocessAfterSolveName],
) -> torch.Tensor:
    """
    given per-face 3x3 transform matrices in `faces_matrices_packed` of shape
    `(len(meshes.faces_packed()), 3, 3)`, treating them as per-face jacobians of a piecewise
    linear mapping, do a poisson solve to find the best-fitting per-vertex map. (as seen in
    Neural Jacobian Fields)
    """
    njfpoisson_rhs_lefts = laplacians_solvers.njfpoisson_rhs_lefts
    assert njfpoisson_rhs_lefts is not None, (
        "must have njfpoisson_rhs_lefts precomputed in `laplacians` struct to run poisson solve. make sure to set compute_poisson_rhs_lefts=True when calling SparseLaplaciansSolvers.from_meshes"
    )
    n_verts_per_face = 3
    n_coords = 3
    # faces_matrices_packed has shape (n_faces_packed, 3vertsperface, 3coords)

    # transpose to be (n_faces, 3coords, 3vertsperface)
    faces_matrices_packed = faces_matrices_packed.transpose(-1, -2)

    # then pad it out to match (batch, max_n_faces, 3coords, 3vertsperface)
    num_faces_per_mesh = meshes.num_faces_per_mesh()
    batch_size = num_faces_per_mesh.size(0)
    max_n_faces_per_mesh = njfpoisson_rhs_lefts.size(2) // n_coords
    # ^ this is the max_n_faces in the shape of njfpoisson_rhs_lefts, which is at least as
    # large as max(num_faces_per_mesh). we want this shape to be compat with
    # njfpoisson_rhs_lefts so we'll use that number.

    faces_matrices_padded = torch.zeros(
        (batch_size, max_n_faces_per_mesh, n_coords, n_verts_per_face),
        dtype=faces_matrices_packed.dtype,
        device=faces_matrices_packed.device,
    )
    batch_idx, idx_in_batch = make_padded_to_packed_indexer(num_faces_per_mesh)
    faces_matrices_padded[batch_idx, idx_in_batch] = faces_matrices_packed

    # then view it as a batch of stacked matrices for bmm with njfpoisson_rhs_lefts
    faces_matrices_padded = faces_matrices_padded.view(
        batch_size, max_n_faces_per_mesh * n_coords, n_verts_per_face
    )

    # then do the bmm to get the right-hand side of the system
    rhs = njfpoisson_rhs_lefts.bmm(faces_matrices_padded)
    # njfpoisson_rhs_lefts is (batch_size, max_n_verts, max_n_faces * 3coords)
    # bmm with face_matrices_padded (batch_size, max_n_faces * 3coords, 3vertsperface)
    # to get rhs shape (batch_size, max_n_verts, 3vertsperface)

    # cholespy cannot init with a padded laplacian, we have to use the unpadded
    # laplacian and the unpadded rhs for this
    soln_verts_packed = torch.cat(
        tuple(
            (
                soln := solver.solve_with_pvrLc_adjusted_rhs(
                    pvrLc.adjust_rhs_for_solving_L_with_removed_rowcols(
                        rhs_this_mesh[: verts_this_mesh.size(0)], verts_this_mesh
                    )
                    if pvrLc
                    else rhs_this_mesh[: verts_this_mesh.size(0)]
                ),
                pvrLc.patch_solution_into_verts(soln, verts_this_mesh, handle_nans=True)
                if pvrLc
                else soln,
            )[-1]
            for solver, pvrLc, rhs_this_mesh, verts_this_mesh in (
                zip(
                    laplacians_solvers.cholespy_solvers,
                    laplacians_solvers.pin_verts_and_removed_L_columns,
                    rhs,
                    meshes.verts_list(),
                )
            )
        ),
        dim=0,
    )

    soln_verts_packed = handle_postprocess_after_solve(
        meshes,
        laplacians_solvers.connected_components_per_mesh,
        soln_verts_packed,
        postprocess,
    )
    return soln_verts_packed


def applymethod__vertex_rotations_into_ARAP_solve(
    patient_meshes: Meshes,
    laplacians_solvers: SparseLaplaciansSolvers,
    per_vertex_rot_matrices_packed: torch.Tensor,
    arap_energy_type: ARAPEnergyTypeName,
    rotations_are_3x2_repr: bool,
    postprocess: Optional[PostprocessAfterSolveName],
    return_offsets_to_solution: bool,
) -> torch.Tensor:
    """
    per_vertex_rot_matrices_packed is of shape `(len(patient_meshes.faces_packed()),3,3)`, a
    rotation matrix for each face if not rotations_are_3x2_repr ; otherwise, it's (...,3,2).
    if return_offsets_to_solution (default=True), returns the verts update
    that will take current verts to the solution, not the solution itself
    """
    soln_verts_packed = calc_ARAP_global_solve(
        patient_meshes,
        laplacians_solvers,
        per_vertex_rot_matrices_packed
        if not rotations_are_3x2_repr
        else convert_rot3x2_to_rot3x3(per_vertex_rot_matrices_packed),
        arap_energy_type,
        postprocess=postprocess,
    )
    if return_offsets_to_solution:
        return soln_verts_packed - patient_meshes.verts_packed()
    else:
        return soln_verts_packed


def convert_rot3x2_to_rot3x3(rot3x2: torch.Tensor) -> torch.Tensor:
    """
    from Hao Li's paper "On the Continuity of Rotation Representations in Neural Networks"
    """
    a1 = rot3x2[:, :, 0]
    a2 = rot3x2[:, :, 1]
    b1 = nn.functional.normalize(a1, dim=-1)
    b2 = nn.functional.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.linalg.cross(b1, b2)
    rot3x3 = torch.stack((b1, b2, b3), dim=-1)
    return rot3x3


def average_face_quaternions_onto_vertex_quaternions(
    face_areas: torch.Tensor,
    n_verts: int,
    faces: torch.Tensor,
    per_face_quaternions_packed: torch.Tensor,
) -> torch.Tensor:
    # let's do https://stackoverflow.com/a/72039849
    # which is itself from http://tbirdal.blogspot.com/2019/10/i-allocate-this-post-to-providing.html
    # which is itself from http://www.acsu.buffalo.edu/~johnc/ave_quat07.pdf Markley et al. 2007
    Q = per_face_quaternions_packed  # shape (n_faces, 4)
    oriented_Q = ((Q[:, 0:1] > 0).float() - 0.5) * 2 * Q

    # do a self-outer product. shape (n_faces, 4, 4)
    outprod_Q = torch.einsum("bi,bk->bik", (oriented_Q, oriented_Q))
    weighted_outprod_Q = outprod_Q * face_areas.view(face_areas.size(0), 1, 1)

    # gather the outer product
    out_A = torch.zeros((n_verts, 4, 4), dtype=Q.dtype, device=Q.device)
    out_A.index_put_((faceverts0 := faces[:, 0],), weighted_outprod_Q, accumulate=True)
    out_A.index_put_((faceverts1 := faces[:, 1],), weighted_outprod_Q, accumulate=True)
    out_A.index_put_((faceverts2 := faces[:, 2],), weighted_outprod_Q, accumulate=True)
    if thlog.logguard(LOG_TRACE):
        thlog.trace(f"""
        weighted_outprod_Q {weighted_outprod_Q}
        out_A {out_A}
        """)

    # gather the weights
    sumfacearea_per_vertex = torch.zeros(n_verts, device=Q.device)
    sumfacearea_per_vertex.index_put_((faceverts0,), face_areas, accumulate=True)
    sumfacearea_per_vertex.index_put_((faceverts1,), face_areas, accumulate=True)
    sumfacearea_per_vertex.index_put_((faceverts2,), face_areas, accumulate=True)

    # divide weighted sum on each vertex by sum of weights around that vertex
    out_A.div_(sumfacearea_per_vertex.view(sumfacearea_per_vertex.size(0), 1, 1))

    # see HACK below: detect bad out_A and overwrite before eigh. The intended eigvec for
    # such bad out_A are [1,0,0,0] (id quaternion), see below. We want to overwrite such
    # that the largest-eigval's eigenvector (for out_Q) comes out to be [1,0,0,0] and also
    # eigvalues are unique. One such matrix is just diag([3,2,1,0])
    out_A_has_degen_matrix = (
        (out_A[:, 1, 1] == 0)
        .logical_and_(out_A[:, 2, 2] == 0)
        .logical_and_(out_A[:, 0, 0] > 0)
    )
    out_A_has_degen_matrix_where = torch.where(out_A_has_degen_matrix)[0]
    if out_A_has_degen_matrix_where.numel() > 0:
        out_A[out_A_has_degen_matrix_where] = torch.diag(
            torch.arange(3, -1, -1, dtype=out_A.dtype, device=out_A.device)
        )

    # eigenvector corresponding to the largest eigenvalue
    eigh = torch.linalg.eigh(out_A)  # named tuple (eigenvalues, eigenvectors)
    out_Q = eigh.eigenvectors[:, :, -1]

    return out_Q


def applymethod__avg_face_rotations_into_ARAP_solve(
    patient_meshes: Meshes,
    laplacians_solvers: SparseLaplaciansSolvers,
    per_face_rot_matrices_packed: torch.Tensor,
    arap_energy_type: ARAPEnergyTypeName,
    rotations_are_3x2_repr: bool,
    postprocess: Optional[PostprocessAfterSolveName],
    return_offsets_to_solution: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    `per_face_rot_matrices_packed` is of shape
    `(len(patient_meshes.faces_packed()),3,3)`, a rotation matrix for each face
    if not rotations_are_3x2_repr ; otherwise, it's (..., 3, 2)

    applies ARAP solve, and returns
    - if return_offsets_to_solution (default=True), returns offsets for vertices (from
        patient_meshes's current locations towards locations obtained in the ARAP solve)
      otherwise, returns the solution positions directly
    - rot matrices per vertex (avg'd from each vert's adjacent faces' rot matrices)
    """
    patient_meshes_verts = patient_meshes.verts_packed()
    patient_meshes_faces = patient_meshes.faces_packed()
    patient_meshes_faces_areas = patient_meshes.faces_areas_packed()
    do_matrix_avg = rotations_are_3x2_repr
    last_dim_shape = 2 if rotations_are_3x2_repr else 3

    if do_matrix_avg:
        per_vertex_pred_packed = torch.zeros(
            (patient_meshes_verts.size(0), 3, last_dim_shape),
            dtype=per_face_rot_matrices_packed.dtype,
            device=per_face_rot_matrices_packed.device,
        )
        gather_per_face_quantity_into_per_vertex_quantity_inplace(
            patient_meshes_faces_areas,
            calc_sum_incident_face_area_per_vertex(
                patient_meshes_verts, patient_meshes_faces
            ),
            patient_meshes_faces,
            per_face_rot_matrices_packed,
            out=per_vertex_pred_packed,
        )
        if last_dim_shape == 2:
            per_vertex_rot_matrices_packed = convert_rot3x2_to_rot3x3(
                per_vertex_pred_packed
            )
        else:
            per_vertex_rot_matrices_packed = per_vertex_pred_packed
    else:
        # use the correct quaternion averaging for 3x3 (but this gives terrible gradients)
        per_face_rot_quats_packed = pt3d_transforms.matrix_to_quaternion(
            per_face_rot_matrices_packed
        )
        per_vertex_rot_quats_packed = average_face_quaternions_onto_vertex_quaternions(
            patient_meshes_faces_areas,
            patient_meshes_verts.size(0),
            patient_meshes_faces,
            per_face_rot_quats_packed,
        )
        per_vertex_rot_matrices_packed = pt3d_transforms.quaternion_to_matrix(
            per_vertex_rot_quats_packed
        )

    soln_verts_packed = calc_ARAP_global_solve(
        patient_meshes,
        laplacians_solvers,
        per_vertex_rot_matrices_packed,
        arap_energy_type,
        postprocess=postprocess,
    )

    if return_offsets_to_solution:
        return soln_verts_packed - patient_meshes_verts, per_vertex_rot_matrices_packed
    else:
        return soln_verts_packed, per_vertex_rot_matrices_packed


def applymethod__face_rotations_into_njf_poisson_solve(
    patient_meshes: Meshes,
    laplacians_solvers: SparseLaplaciansSolvers,
    per_face_rot_matrices_packed: torch.Tensor,
    rotations_are_3x2_repr: bool,
    postprocess: Optional[PostprocessAfterSolveName],
    return_offsets_to_solution: bool,
) -> torch.Tensor:
    """
    per_face_rot_matrices_packed is of shape `(len(patient_meshes.faces_packed()),3,3)`, a
    rotation matrix for each face if not rotations_are_3x2_repr ; otherwise, it's (...,3,2)
    if return_offsets_to_solution (default=True), returns the verts update
    that will take current verts to the solution, not the solution itself
    """
    soln_verts_packed = calc_njf_poisson_global_solve(
        patient_meshes,
        laplacians_solvers,
        per_face_rot_matrices_packed
        if not rotations_are_3x2_repr
        else convert_rot3x2_to_rot3x3(per_face_rot_matrices_packed),
        postprocess=postprocess,
    )
    if return_offsets_to_solution:
        return soln_verts_packed - patient_meshes.verts_packed()
    else:
        return soln_verts_packed


def seed_all(
    torch_seed: Optional[int], numpy_seed: Optional[int]
) -> Tuple[torch.Generator, int, int]:
    """
    unfortunately seeding doesn't really do much because bmm itself is nondeterministic
    (by default, unless we use deterministic algorithms, which are much slower)
    """
    if torch_seed is None:
        torch_seed = int(torch.initial_seed())
        thlog.info(f"torch initial seed is {torch_seed}")
    if numpy_seed is None:
        # we cannot get the actual numpy init seed, so we'll just generate a random number
        # and use that as a seed!!
        numpy_seed = np.random.randint(2**32)
        thlog.info(f"numpy initial seed is {numpy_seed}")

    rng = torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    np.random.seed(numpy_seed)
    random.seed(0)
    thlog.info(f"Set torch seed to {torch_seed} and numpy seed to {numpy_seed}")
    return rng, torch_seed, numpy_seed


ApplyScaleAfterProcrustesModeName = Literal["norm", "xyz", "coord456"]


def softcap(k: float, m: float, x: torch.Tensor) -> torch.Tensor:
    """
    soft min-like clamping function that is piecewise y = x and y = m *
    sigmoid(k/m) with a transition point where the derivative of the latter is 1
    """
    import math

    if k < 4:
        raise ValueError("softcap k must be at least 4")
    # if k < 4, the derivative of the scaled sigmoid never hits 1 so we won't be
    # able to find a smooth transition point between y=x and the sigmoid
    # def sigderivminus1(x: float) -> float:
    #     exx = np.exp(-k * x / max)
    #     return (k * exx) / ((1 + exx) * (1 + exx)) - 1
    # we can solve this analytically,
    exx = 0.5 * (k - math.sqrt(k - 4) * math.sqrt(k) - 2)
    u = math.log(exx) * m / (-k)

    sigu = m / (1 + exx)
    o = sigu - u

    return torch.where(x < sigu, x, m * torch.sigmoid((k / m) * (x - o)))


def leakyminmax(min: float, max: float, slope: float, x: torch.Tensor) -> torch.Tensor:
    x_maxclamped = max - torch.nn.functional.leaky_relu(max - x, negative_slope=slope)
    return min + torch.nn.functional.leaky_relu(x_maxclamped - min, negative_slope=slope)


@dataclass(slots=True)
class ApplyScaleAfterProcrustes_ClampAppliedScale_Settings(Thronfig):
    mode: Literal["hard", "softcap", "leaky"]
    target: Literal["coord", "det"]
    min: float
    max: float
    softcap_k: float = 4
    """ only applicable for softcap """
    leaky_slope: float = 0.01
    """ only applicable for leakymin """

    def apply_clamp(self, scale: torch.Tensor) -> torch.Tensor:
        if self.mode == "hard":
            clamper = lambda x: torch.clamp(x, min=self.min, max=self.max)
        elif self.mode == "softcap":
            # softcap handles the soft ceiling, clamp handles the hard floor
            clamper = lambda x: torch.clamp(
                softcap(self.softcap_k, self.max, x), min=self.min
            )
        elif self.mode == "leaky":
            clamper = lambda x: leakyminmax(self.min, self.max, self.leaky_slope, x)
        else:
            raise InvalidConfigError(f"unknown clamp mode {self.mode}")

        if self.target == "det":
            assert scale.shape[1:] == (3,), (
                "don't use 'det' mode if not coord456 just use 'coord' mode"
            )
            # must be [sx, sy, sz], and det is sx * sy * sz
            det = scale.prod(dim=-1, keepdim=True)
            clamped_det = clamper(det)
            ratio = clamped_det / det
            # mult each of [sx, sy, sz] by cbrt(ratio) so that det becomes the clamped det
            return scale * torch.pow(ratio, 1 / 3)
        elif self.target == "coord":
            return clamper(scale)
        else:
            raise InvalidConfigError("unknown target for clamping")


@dataclass(slots=True)
class ApplyScaleAfterProcrustes_Settings(Thronfig):
    mode: ApplyScaleAfterProcrustesModeName
    """
    if scalar, then grab the norm of the vector before normalizing and use it as
    a scale transform on top of the 3x3 rotation found by procrustes
    if xyz, then grab the ratio of the coordinates after and before normalizing
    as the scale transform on top of the 3x3 rotation found by procrustes
    """
    clamp: Optional[ApplyScaleAfterProcrustes_ClampAppliedScale_Settings]
    """
    clamp specifications
    """


@dataclass(slots=True)
class Procrustes_Settings(Thronfig):
    lamb: float
    normalize_target_normals: bool
    apply_scale_after_procrustes: Optional[ApplyScaleAfterProcrustes_Settings]


@dataclass(slots=True)
class VolumeBasedHeuristicsFnsForApplyScaleAfterProcrustes:
    use_orig_volume_instead_of_curr_volume: bool
    vns_scale_max_heuristic_calc_fn: Callable[[float, Optional[float]], float]


@dataclass(slots=True)
class ProcrustesPrecomputeAndWhetherToNormalize:
    pp: ProcrustesPrecompute
    normalize: bool
    """ whether to normalize the target normal """
    apply_scale_after_procrustes_and_vselvol_heuristics: Optional[
        Tuple[
            ApplyScaleAfterProcrustes_Settings,
            Optional[VolumeBasedHeuristicsFnsForApplyScaleAfterProcrustes],
        ]
    ]
    """ apply scaling 3x3 matrix after procrustes normal"""


#### convenience API that initializes and decides what to do based on config field values
# (the above calc functions are specialized, to be chosen depending on the config)
# mostly for use with the optimization pipeline and for applying saved deformation qty files
# saved from the optimization


@dataclass(slots=True)
class ProcrustesInitsForQuantityInit:
    procrustes_cfg: Procrustes_Settings
    solvers: SparseLaplaciansSolvers
    arap_energy_type: Optional[ARAPEnergyTypeName]
    vselvol_heuristics_fns: Optional[VolumeBasedHeuristicsFnsForApplyScaleAfterProcrustes]


@dataclass(slots=True)
class QuantityBeingOptimized:
    """
    - if this_is == "verts_offsets", tensor has shape (n_verts_packed, 3)
    - if this_is == "faces_normals", tensor has shape (n_faces_packed, 3)
    - if this_is == "verts_normals", tensor has shape (n_verts_packed, 3)
    - if this_is == "verts_normals_and_scale", tensor has shape (n_verts_packed, 6)
    - if this_is == "faces_jacobians", parameter has one tensor of shape (n_faces_packed, 3, 3)
    - if this_is == "faces_3x2rotations", parameter has one tensor of shape (n_faces_packed, 3, 2)
    - if this_is == "verts_3x2rotations", parameter has one tensor of shape (n_verts_packed, 3, 2)
    - if this_is == "verts_jacobians", parameter has one tensor of shape (n_verts_packed, 3, 3)

    the __getitem__ operation returns another QuantityBeingOptimized that has an extracted
    packed quantity as a new (n_verts_packed/n_faces_packed, *) tensor, containing elements
    corresponding to the indexed meshes in the original batched QuantityBeingOptimized

    (this is also the idea for all the dataclasses that have a MeshesPackedIndexer component
    in addition to this one. This one just happens to not use MeshesPackedIndexer but has
    the same behavior on indexing.)
    """

    tensor: torch.Tensor
    """ main parameter tensor to be optimized. present for all this_is possibilities """

    this_is: DeformOptimQuantityName

    num_verts_per_mesh: List[int]
    num_faces_per_mesh: List[int]
    """ originally from the meshes where this QuantityBeingOptimized came from.
    bookkeeping to help with __getitem__ """

    procrustes_struct_if_needed: Optional[ProcrustesPrecomputeAndWhetherToNormalize]
    """
    this is only non-None when optimize_deform_via is a method that requires procrustes
    """

    def __getitem__(self, index: Union[int, List[int], torch.Tensor]):
        list_idxr_fn: Callable[[Sequence], Sequence]
        if isinstance(index, int):
            list_idxr_fn = lambda xs: tuple(xs[index])
        elif isinstance(index, list):
            list_idxr_fn = lambda xs: tuple(xs[i] for i in index)
        else:
            list_idxr_fn = lambda xs: tuple(xs[int(i.item())] for i in index)

        if (
            (this_is := self.this_is) == "faces_jacobians"
            or this_is == "faces_3x2rotations"
            or this_is == "faces_normals"
        ):
            tensor = torch.cat(
                list_idxr_fn(torch.split(self.tensor, self.num_faces_per_mesh)),
                dim=0,
            )
        elif (
            this_is == "verts_offsets"
            or this_is == "verts_3x2rotations"
            or this_is == "verts_normals"
            or this_is == "verts_normals_and_scale"
            or this_is == "verts_jacobians"
        ):
            tensor = torch.cat(
                list_idxr_fn(torch.split(self.tensor, self.num_verts_per_mesh)),
                dim=0,
            )
        else:
            raise InvalidConfigError(f"unknown quantity to optimize {this_is}")

        return __class__(
            tensor=tensor,
            this_is=this_is,
            num_verts_per_mesh=list(list_idxr_fn(self.num_verts_per_mesh)),
            num_faces_per_mesh=list(list_idxr_fn(self.num_faces_per_mesh)),
            procrustes_struct_if_needed=(
                ProcrustesPrecomputeAndWhetherToNormalize(
                    self.procrustes_struct_if_needed.pp[index],
                    self.procrustes_struct_if_needed.normalize,
                    self.procrustes_struct_if_needed.apply_scale_after_procrustes_and_vselvol_heuristics,
                )
                if self.procrustes_struct_if_needed
                else None
            ),
        )

    @classmethod
    def init_according_to_cfg(
        cls,
        meshes: Meshes,
        optimize_deform_via: DeformOptimQuantityName,
        procrustes_inits_if_procrustes_needed: Optional[ProcrustesInitsForQuantityInit],
    ) -> "QuantityBeingOptimized":
        procrustes_struct_if_needed = None
        if optimize_deform_via == "verts_offsets":
            tensor = torch.zeros_like(meshes.verts_packed())
        elif optimize_deform_via == "faces_normals":
            # optimize the offsets to be added to normals
            tensor = torch.zeros_like(meshes.faces_normals_packed())
        elif (
            is_verts_normals_and_scale := (optimize_deform_via == "verts_normals_and_scale")
        ) or optimize_deform_via == "verts_normals":
            tensor = meshes.verts_normals_packed()
            if is_verts_normals_and_scale:
                tensor = torch.cat(
                    (
                        tensor,
                        torch.ones(
                            (tensor.size(0), 3), dtype=tensor.dtype, device=tensor.device
                        ),
                    ),
                    dim=-1,
                )
            if procrustes_inits_if_procrustes_needed:
                (
                    procrustes_cfg,
                    laplacians_solvers,
                    arap_energy_type,
                    vselvol_heuristics_fns,
                ) = (
                    procrustes_inits_if_procrustes_needed.procrustes_cfg,
                    procrustes_inits_if_procrustes_needed.solvers,
                    procrustes_inits_if_procrustes_needed.arap_energy_type,
                    procrustes_inits_if_procrustes_needed.vselvol_heuristics_fns,
                )
                if is_verts_normals_and_scale and (
                    (not procrustes_cfg.apply_scale_after_procrustes)
                    or procrustes_cfg.apply_scale_after_procrustes.mode != "coord456"
                ):
                    raise InvalidConfigError(
                        "is verts_normals_and_scale, requires procrustes_cfg.apply_scale_after_procrustes == 'coord456'"
                    )
                if (
                    procrustes_cfg.apply_scale_after_procrustes
                    and procrustes_cfg.apply_scale_after_procrustes.mode == "coord456"
                    and not is_verts_normals_and_scale
                ):
                    raise InvalidConfigError(
                        "procrustes_cfg.apply_scale_after_procrustes.mode == 'coord456' must correspond to optimize_deform_via 'verts_normals_and_scale'"
                    )

                procrustes_struct_if_needed = ProcrustesPrecomputeAndWhetherToNormalize(
                    pp=ProcrustesPrecompute.from_meshes(
                        local_step_procrustes_lambda=procrustes_cfg.lamb,
                        arap_energy_type=arap_energy_type,
                        laplacians_solvers=laplacians_solvers,
                        patient_meshes=meshes,
                    ),
                    normalize=procrustes_cfg.normalize_target_normals,
                    apply_scale_after_procrustes_and_vselvol_heuristics=(
                        procrustes_cfg.apply_scale_after_procrustes,
                        vselvol_heuristics_fns,
                    )
                    if procrustes_cfg.apply_scale_after_procrustes
                    else None,
                )
            else:
                procrustes_struct_if_needed = None

        elif optimize_deform_via == "faces_3x2rotations":
            verts = meshes.verts_packed()
            faces = meshes.faces_packed()
            tensor = torch.zeros(
                (faces.size(0), 3, 2),
                device=verts.device,
                dtype=verts.dtype,
            )
            # initialize to identity
            tensor[:, (0, 1), (0, 1)] = 1.0
        elif optimize_deform_via == "verts_3x2rotations":
            verts = meshes.verts_packed()
            faces = meshes.faces_packed()
            tensor = torch.zeros(
                (verts.size(0), 3, 2),
                device=verts.device,
                dtype=verts.dtype,
            )
            # initialize to identity
            tensor[:, (0, 1), (0, 1)] = 1.0
        elif optimize_deform_via == "verts_jacobians":
            verts = meshes.verts_packed()
            faces = meshes.faces_packed()
            tensor = torch.zeros(
                (verts.size(0), 3, 3),
                device=verts.device,
                dtype=verts.dtype,
            )
            # initialize to identity
            tensor[:, (0, 1, 2), (0, 1, 2)] = 1.0
        elif optimize_deform_via == "faces_jacobians":
            verts = meshes.verts_packed()
            faces = meshes.faces_packed()
            tensor = torch.zeros(
                (faces.size(0), n_coords := 3, n_verts_per_face := 3),
                device=verts.device,
                dtype=verts.dtype,
            )
            # initialize to identity
            tensor[:, (0, 1, 2), (0, 1, 2)] = 1.0
        else:
            raise InvalidConfigError(
                f"invalid/not yet implemented optimize_deform_via {optimize_deform_via}"
            )

        return cls(
            tensor=tensor.requires_grad_(),
            this_is=optimize_deform_via,
            num_verts_per_mesh=meshes.num_verts_per_mesh().tolist(),
            num_faces_per_mesh=meshes.num_faces_per_mesh().tolist(),
            procrustes_struct_if_needed=procrustes_struct_if_needed,
        )


@dataclass(slots=True)
class ElemNormals_IntermediateResults:
    vert_matrices: torch.Tensor
    procrustes_covar: Optional[torch.Tensor]
    rotscale_matrices: Optional[Tuple[torch.Tensor, torch.Tensor]]

    def get_sequence(
        self, num_verts_per_mesh: List[int]
    ) -> Sequence["ElemNormals_IntermediateResults"]:
        vert_matrices_list = self.vert_matrices.split(num_verts_per_mesh, dim=0)
        _nones = tuple(None for _ in num_verts_per_mesh)
        procrustes_covar_list = (
            self.procrustes_covar.split(num_verts_per_mesh, dim=0)
            if self.procrustes_covar is not None
            else _nones
        )
        if self.rotscale_matrices is not None:
            rot, scale = self.rotscale_matrices
            rot_list = rot.split(num_verts_per_mesh, dim=0)
            scale_list = scale.split(num_verts_per_mesh, dim=0)
            rotscale_list = zip(rot_list, scale_list)
        else:
            rotscale_list = _nones
        return tuple(
            ElemNormals_IntermediateResults(
                vert_matrices_this_mesh,
                procrustes_covar_this_mesh,
                rotscale_this_mesh,
            )
            for vert_matrices_this_mesh, procrustes_covar_this_mesh, rotscale_this_mesh in zip(
                vert_matrices_list, procrustes_covar_list, rotscale_list
            )
        )


DeformationIntermediateResults = Union[None, ElemNormals_IntermediateResults]
# add other structs here into this union for the other optimize_deform_via methods


@dataclass(slots=True)
class InputsToDeformationSolveMethods:
    intermediate_results: DeformationIntermediateResults = None
    rotations_are_3x2_repr: bool = False
    face_matrices_packed: Optional[torch.Tensor] = None
    vert_matrices_packed: Optional[torch.Tensor] = None
    shortcircuit_deformation_solution: Optional[Sequence[torch.Tensor]] = None
    """
    if this field is specified, then we skip the solve method and just take this as the
    deformation solution verts. certain solve methods (namely direct vertex offset) can use
    this to directly give the solution without going through a solve as is required by
    others; can also use this to inject a different pipeline that doesn't use the usual
    poisson solve pipeline here
    """


def calc_clamped_scale_vec_according_to_cfg(
    pp_etc_apply_scale: Tuple[
        ApplyScaleAfterProcrustes_Settings,
        Optional[VolumeBasedHeuristicsFnsForApplyScaleAfterProcrustes],
    ],
    quantity_being_optimized: QuantityBeingOptimized,
    current_and_original_selection_sum_volumes_for_vselvol_heuristics: Sequence[
        Tuple[Optional[float], Optional[float]]
    ],
) -> torch.Tensor:
    assert (
        quantity_being_optimized.this_is == "verts_normals_and_scale"
        or quantity_being_optimized.this_is == "verts_normals"
    ), "this is only applicable for verts_normals_and_scale and verts_normals deform qty"
    __get_scale_vec: Callable[[torch.Tensor], torch.Tensor]

    apply_scale_cfg, vselvol_heuristics_etc = pp_etc_apply_scale
    if apply_scale_cfg.mode == "coord456":
        # this must mean verts_normals_and_scale (we already checked at qty init)
        scale = quantity_being_optimized.tensor[:, 3:]
        __get_scale_vec = lambda _scale: _scale
    elif apply_scale_cfg.mode == "norm":
        # this cannot mean verts_normals_and_scale
        scale = quantity_being_optimized.tensor.norm(dim=-1, keepdim=True)
        __get_scale_vec = lambda _scale: _scale.repeat_interleave(3, dim=-1)
    elif apply_scale_cfg.mode == "xyz":
        raise NotImplementedError("xyz mode not implemented")
    else:
        raise InvalidConfigError("unknown scale mode")

    if clamp_cfg := apply_scale_cfg.clamp:
        if vselvol_heuristics_etc:

            def __apply_clamp_elemwise(_scalearr: torch.Tensor):
                scales_clamped_via_vselvol_heuristics = []
                old_scale_shape = _scalearr.shape
                for (curr_volume, orig_volume), scales_this_mesh in zip(
                    current_and_original_selection_sum_volumes_for_vselvol_heuristics,
                    _scalearr.split(quantity_being_optimized.num_verts_per_mesh, dim=0),
                ):
                    # modify the clamp_cfg.max in place, do the clamp apply, then revert that change
                    original_max = clamp_cfg.max
                    clamp_cfg.max = vselvol_heuristics_etc.vns_scale_max_heuristic_calc_fn(
                        clamp_cfg.max,
                        (
                            orig_volume
                            if vselvol_heuristics_etc.use_orig_volume_instead_of_curr_volume
                            else curr_volume
                        ),
                    )
                    scales_clamped_via_vselvol_heuristics.append(
                        clamp_cfg.apply_clamp(scales_this_mesh)
                    )
                    if thlog.logguard(LOG_TRACE):
                        thlog.trace(
                            f"clamp_cfg.max set by vselvol heuristic to {clamp_cfg.max}"
                        )
                    clamp_cfg.max = original_max
                _scalearr = torch.cat(scales_clamped_via_vselvol_heuristics, dim=0)
                assert _scalearr.shape == old_scale_shape
                return _scalearr
        else:

            def __apply_clamp_elemwise(_scalearr: torch.Tensor):
                return clamp_cfg.apply_clamp(_scalearr)
    else:
        __apply_clamp_elemwise = lambda _scalearr: _scalearr

    scale_vec = __apply_clamp_elemwise(__get_scale_vec(scale))

    if thlog.logguard(LOG_DEBUG):
        with torch.no_grad():
            thlog.debug(
                f"scale min, max, avg {scale_vec.min():.4f},{scale_vec.max():.4f},{scale_vec.mean():.4f}"
            )
    return scale_vec


def calc_inputs_to_solve_for_deformation_according_to_cfg(
    pt3d_batched_meshes: Meshes,
    quantity_being_optimized: QuantityBeingOptimized,
    current_vertex_selection_mask_packed: torch.Tensor,
    current_and_original_selection_sum_volumes_for_vselvol_heuristics: Sequence[
        Tuple[Optional[float], Optional[float]]
    ],
) -> InputsToDeformationSolveMethods:
    """
    This will run the local step, i.e. compute the inputs to the global solve.
    """
    if (quantity_is := quantity_being_optimized.this_is) == "faces_jacobians":
        ret = InputsToDeformationSolveMethods(
            face_matrices_packed=quantity_being_optimized.tensor
        )
    elif quantity_is == "faces_3x2rotations":
        ret = InputsToDeformationSolveMethods(
            face_matrices_packed=quantity_being_optimized.tensor,
            rotations_are_3x2_repr=True,
        )
    elif quantity_is == "verts_3x2rotations":
        ret = InputsToDeformationSolveMethods(
            vert_matrices_packed=quantity_being_optimized.tensor,
            rotations_are_3x2_repr=True,
        )
    elif quantity_is == "verts_jacobians":
        ret = InputsToDeformationSolveMethods(
            vert_matrices_packed=quantity_being_optimized.tensor,
            rotations_are_3x2_repr=False,
        )
    elif quantity_is == "faces_normals":
        # offset added to faces_normals, not the normals themselves
        # (we can't init as the normals themselves somehow)
        original_faces_normals = pt3d_batched_meshes.faces_normals_packed()
        updated_faces_normals = original_faces_normals + quantity_being_optimized.tensor
        face_matrices_packed = calc_rot_matrices_axisangle(
            original_faces_normals,
            torch.nn.functional.normalize(updated_faces_normals, dim=-1),
            epsilon=1e-6,
        )
        intermediate_results = ElemNormals_IntermediateResults(
            face_matrices_packed, None, None
        )
        ret = InputsToDeformationSolveMethods(
            face_matrices_packed=face_matrices_packed,
            intermediate_results=intermediate_results,
        )
    elif (
        is_verts_normals_and_scale := (quantity_is == "verts_normals_and_scale")
    ) or quantity_is == "verts_normals":
        normal_for_procrustes = (
            quantity_being_optimized.tensor[:, :3]
            if is_verts_normals_and_scale
            else quantity_being_optimized.tensor
        )
        if (pp_etc := quantity_being_optimized.procrustes_struct_if_needed) is not None:
            rot_matrices_packed, procrustes_covar = calc_rot_matrices_with_procrustes(
                procrustes_precompute=pp_etc.pp,
                curr_deformed_verts_packed=pt3d_batched_meshes.verts_packed(),
                target_verts_normals_packed=(
                    torch.nn.functional.normalize(normal_for_procrustes, dim=-1)
                    if pp_etc.normalize
                    else normal_for_procrustes
                ),
                verts_selmask_packed=current_vertex_selection_mask_packed,
                curr_verts_normals_packed__backupforaxisangle=(
                    pt3d_batched_meshes.verts_normals_packed()
                    if DRMSH_PROCRUSTES_DEGEN_SVDVALS_BECAREFUL
                    else None
                ),
            )
            if (
                pp_etc_apply_scale
                := pp_etc.apply_scale_after_procrustes_and_vselvol_heuristics
            ):
                scale_vec = calc_clamped_scale_vec_according_to_cfg(
                    pp_etc_apply_scale,
                    quantity_being_optimized,
                    current_and_original_selection_sum_volumes_for_vselvol_heuristics,
                )
                scale_matrices_packed = torch.diag_embed(scale_vec)
                _sclmatsz0, _sclmatsz1, _sclmatsz2 = scale_matrices_packed.shape
                assert (
                    _sclmatsz0 == rot_matrices_packed.size(0)
                    and _sclmatsz1 == 3
                    and _sclmatsz2 == 3
                )
                vert_matrices_packed = scale_matrices_packed.bmm(rot_matrices_packed)
                rotscale_matrices_packed = (rot_matrices_packed, scale_matrices_packed)
            else:
                vert_matrices_packed = rot_matrices_packed
                rotscale_matrices_packed = None
        else:
            # no procrustes and apply_scale_after_procrustes config
            rot_matrices_packed = calc_rot_matrices_axisangle(
                pt3d_batched_meshes.verts_normals_packed(),
                torch.nn.functional.normalize(normal_for_procrustes, dim=-1),
                epsilon=1e-6,
            )
            scale_matrices_packed = torch.diag_embed(
                quantity_being_optimized.tensor[:, 3:]
                if quantity_is == "verts_normals_and_scale"
                else torch.ones_like(quantity_being_optimized.tensor)
                # ^ ones_like((V, 3)), for "verts_normals"
            )
            vert_matrices_packed = scale_matrices_packed.bmm(rot_matrices_packed)
            procrustes_covar = None
            rotscale_matrices_packed = (rot_matrices_packed, scale_matrices_packed)

        intermediate_results = ElemNormals_IntermediateResults(
            vert_matrices_packed, procrustes_covar, rotscale_matrices_packed
        )
        ret = InputsToDeformationSolveMethods(
            vert_matrices_packed=vert_matrices_packed,
            intermediate_results=intermediate_results,
        )
    elif quantity_is == "verts_offsets":
        ret = InputsToDeformationSolveMethods(
            shortcircuit_deformation_solution=per_vertex_packed_to_list(
                pt3d_batched_meshes,
                pt3d_batched_meshes.verts_packed() + quantity_being_optimized.tensor,
            )
        )
    else:
        raise InvalidConfigError(
            f"unknown/not yet implemented quantity to optimize: {quantity_being_optimized.this_is}"
        )
    # (the selection mask has 1 for "enable deform" and 0 for "don't deform")
    # but we want to turn the "0"-marked vertices' matrices into identity
    invmask = current_vertex_selection_mask_packed.logical_not()
    if ret.rotations_are_3x2_repr:
        diag_rowscols = (0, 1)
        offdiag_rows = (0, 1, 2, 2)
        offdiag_cols = (1, 0, 0, 1)
    else:
        diag_rowscols = (0, 1, 2)
        offdiag_rows = (0, 0, 1, 1, 2, 2)
        offdiag_cols = (1, 2, 0, 2, 0, 1)

    if ret.vert_matrices_packed is not None:
        matrices_to_modify = ret.vert_matrices_packed
    elif ret.face_matrices_packed is not None:
        matrices_to_modify = ret.face_matrices_packed
        # invmask is a vertex mask, not a face mask!
        # cast invmask to a face mask: a face has the invmask value True
        # (=set to identity) if any of its vertices have an invmask value
        # True (i.e. a face is made identity if any vertex is made identity,
        # i.e. only faces with all vertices selected are allowed through!)
        invmask = invmask[pt3d_batched_meshes.faces_packed()].any(dim=1)
    elif ret.shortcircuit_deformation_solution is not None:
        # just return, nothing else to do for this
        return ret
    else:
        raise InvalidConfigError(
            "deform method made neither vert_matrices_packed nor face_matrices_packed nor shortcircuit in the InputsToDeformationSolveMethods"
        )

    for r, c in zip(diag_rowscols, diag_rowscols):
        # fill diag with 1
        matrices_to_modify[invmask, r, c] = 1.0
    for r, c in zip(offdiag_rows, offdiag_cols):
        # fill off-diag with 0
        matrices_to_modify[invmask, r, c] = 0.0
    return ret


def calc_deformed_verts_solution_according_to_cfg(
    pt3d_batched_meshes: Meshes,
    solve_method: DeformSolveMethodName,
    arap_energy_type: Optional[ARAPEnergyTypeName],
    postprocess: Optional[PostprocessAfterSolveName],
    my_solvers: SparseLaplaciansSolvers,
    inputs_to_deformation_solve_methods: InputsToDeformationSolveMethods,
) -> Tuple[Sequence[torch.Tensor], DeformationIntermediateResults]:
    """
    returns
    - a list of solution verts tensors, each corresponding to a struct in meshes_structs
    - intermediate results, present or None depending on the quantity_being_optimized, in
        case we wish to penalize or view some intermediate result involved in a deform method

    pt3d_batched_meshes must be a pytorch3d Meshes batch with the same number of
    meshes as len(meshes_structs), and each mesh in pt3d_batched_meshes must
    match the vertex (v_pos) and face (t_pos_idx) array of the corresp. mesh struct's nvdm_loaded_mesh
    """
    rotations_are_3x2_repr = inputs_to_deformation_solve_methods.rotations_are_3x2_repr
    if inputs_to_deformation_solve_methods.shortcircuit_deformation_solution is not None:
        soln_verts_list = (
            inputs_to_deformation_solve_methods.shortcircuit_deformation_solution
        )

    elif solve_method == "njfpoisson":
        assert inputs_to_deformation_solve_methods.face_matrices_packed is not None, (
            "this optimize_deform_via does not yield per-face transforms needed for NJF-style poisson solve"
        )
        soln_verts_packed = applymethod__face_rotations_into_njf_poisson_solve(
            pt3d_batched_meshes,
            my_solvers,
            inputs_to_deformation_solve_methods.face_matrices_packed,
            rotations_are_3x2_repr=rotations_are_3x2_repr,
            postprocess=postprocess,
            return_offsets_to_solution=False,
        )
        soln_verts_list = per_vertex_packed_to_list(pt3d_batched_meshes, soln_verts_packed)

    elif solve_method == "arap":
        assert arap_energy_type is not None, (
            "need non-None arap_energy_type for solve_method arap"
        )
        if inputs_to_deformation_solve_methods.face_matrices_packed is not None:
            soln_verts_packed, _ = applymethod__avg_face_rotations_into_ARAP_solve(
                pt3d_batched_meshes,
                my_solvers,
                inputs_to_deformation_solve_methods.face_matrices_packed,
                arap_energy_type,
                rotations_are_3x2_repr=rotations_are_3x2_repr,
                postprocess=postprocess,
                return_offsets_to_solution=False,
            )
        elif inputs_to_deformation_solve_methods.vert_matrices_packed is not None:
            soln_verts_packed = applymethod__vertex_rotations_into_ARAP_solve(
                pt3d_batched_meshes,
                my_solvers,
                inputs_to_deformation_solve_methods.vert_matrices_packed,
                arap_energy_type=arap_energy_type,
                rotations_are_3x2_repr=rotations_are_3x2_repr,
                postprocess=postprocess,
                return_offsets_to_solution=False,
            )
        else:
            raise AssertionError(
                "did I forget to set face_matrices_packed and vert_matrices_packed"
            )
        soln_verts_list = per_vertex_packed_to_list(pt3d_batched_meshes, soln_verts_packed)

    else:
        raise InvalidConfigError(f"unknown solve method {solve_method}")

    return soln_verts_list, inputs_to_deformation_solve_methods.intermediate_results
