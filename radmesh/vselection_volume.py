from typing import Tuple, Dict, Union, List, cast
import numpy as np
import igl

from thlog import Thlogger, LOG_INFO, VIZ_TRACE
from .misc_helpers import get_bool_env_variable

thlog = Thlogger(LOG_INFO, VIZ_TRACE, "vselvol")

VSELVOLDEBUG_SAVE_HOLECLOSED_COMPONENTS = get_bool_env_variable(
    "VSELVOLDEBUG_SAVE_HOLECLOSED_COMPONENTS"
)
"""
if present, this should be an .obj path to which to save the merged holeclosed components as a mesh
"""


def calc_mesh_volume_estimate(verts: np.ndarray, faces: np.ndarray) -> float:
    face_verts_coords = verts[faces]
    v1 = face_verts_coords[:, 0]
    v2 = face_verts_coords[:, 1]
    v3 = face_verts_coords[:, 2]
    return float(((v1 * (np.cross(v2, v3))).sum(axis=-1) / 6).sum())


class HolecloserGaveUp(BaseException):
    pass


def meshlab_holecloser(
    v_compact_this_group: np.ndarray, f_compact_this_group: np.ndarray, n_bdry_ekeys: int
):
    thlog.info("calling meshlab holecloser")
    import pymeshlab

    ms = pymeshlab.MeshSet()  # type: ignore
    mesh = pymeshlab.Mesh(  # type: ignore
        vertex_matrix=v_compact_this_group,
        face_matrix=f_compact_this_group,
    )
    ms.add_mesh(mesh, "hi")
    ms.meshing_close_holes(maxholesize=n_bdry_ekeys + 500)
    mesh = ms.current_mesh()

    v_after_closeholes = mesh.vertex_matrix()
    f_after_closeholes = mesh.face_matrix()
    return v_after_closeholes, f_after_closeholes


def calc_holeclosed_vselection_volume(
    v: np.ndarray,
    f: np.ndarray,
    vsel: np.ndarray,
) -> float:
    # vsel is a binary mask of shape (len(v),), True means selected, False is not
    # a face is selected if all its vertices are selected

    edgekey_to_faces: Dict[Tuple[int, int], Union[Tuple[int, int], int]] = {}

    # first pass: get edge-face adjacency
    selected_fis: List[int] = []
    selected_fis_e_keys: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = []

    for fi in range(len(f)):
        v0i, v1i, v2i = f[fi, 0], f[fi, 1], f[fi, 2]
        if not (vsel[v0i] and vsel[v1i] and vsel[v2i]):
            # face is considered not selected, skip it
            continue

        e01_key = (v0i, v1i)  # if v0i <= v1i else (v1i, v0i)
        e12_key = (v1i, v2i)  # if v1i <= v2i else (v2i, v1i)
        e20_key = (v2i, v0i)  # if v2i <= v0i else (v0i, v2i)
        e_keys_this_face = (e01_key, e12_key, e20_key)
        actual_used_e_keys_this_face: List[Tuple[int, int]] = []
        selected_fis.append(fi)

        for e_key in e_keys_this_face:
            e_key_swap = (e_key[1], e_key[0])
            if isinstance(existing_e_f_swap := edgekey_to_faces.get(e_key_swap), int):
                edgekey_to_faces[e_key_swap] = (existing_e_f_swap, fi)
                actual_used_e_keys_this_face.append(e_key_swap)
            elif existing_e_f_swap is None:
                actual_used_e_keys_this_face.append(e_key)
                if isinstance(existing_e_f := edgekey_to_faces.get(e_key), int):
                    edgekey_to_faces[e_key] = (fi, existing_e_f)
                elif existing_e_f is None:
                    edgekey_to_faces[e_key] = fi
                elif isinstance(existing_e_f, tuple):
                    raise ValueError("nonmanifold, edge borders more than two faces!")
                else:
                    raise ValueError("impossible")
            elif isinstance(existing_e_f_swap, tuple):
                raise ValueError("nonmanifold, edge borders more than two faces!")
            else:
                raise ValueError("impossible")

        assert len(actual_used_e_keys_this_face) == 3
        selected_fis_e_keys.append(
            cast(
                Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
                tuple(actual_used_e_keys_this_face),
            )
        )

    # second pass: union find to find connected components of faces and boundary edges
    # all faces start out as their own group
    group = np.arange(len(f))
    boundary_edgekey_to_group: Dict[Tuple[int, int], int] = {}
    for fi, e_keys_this_face in zip(selected_fis, selected_fis_e_keys):
        for e_key in e_keys_this_face:
            if isinstance(existing_e_f := edgekey_to_faces.get(e_key), tuple):
                # link the two faces together in the union-find, taking adj_f0i's group as representative
                adj_f0i, adj_f1i = existing_e_f
                group_of_f0i = group[adj_f0i]
                while group_of_f0i != (parent := group[group_of_f0i]):
                    group_of_f0i = parent
                group_of_f1i = group[adj_f1i]
                while group_of_f1i != (parent := group[group_of_f1i]):
                    group_of_f1i = parent

                # path compress
                group[adj_f0i] = group_of_f0i
                group[adj_f1i] = group_of_f0i
                group[group_of_f1i] = group_of_f0i  # link the two groups
            elif isinstance(existing_e_f, int):
                # boundary edge, remember its group
                boundary_edgekey_to_group[e_key] = group[existing_e_f]
            else:
                raise ValueError("impossible")

    # path-compress the unionfind
    group_to_face_idxs: Dict[int, List[int]] = {}
    for fi in selected_fis:
        group_of_fi = cast(int, group[fi])
        while group_of_fi != (parent := group[group_of_fi]):
            group_of_fi = parent
        group[fi] = group_of_fi

        if (
            existing_face_idxs_this_group := group_to_face_idxs.get(group_of_fi)
        ) is not None:
            existing_face_idxs_this_group.append(fi)
        else:
            group_to_face_idxs[group_of_fi] = [fi]

    # convert group idxs in boundary_edgekey_to_group to be the canonical/root group idxs
    for e_key, group_of_this in boundary_edgekey_to_group.items():
        boundary_edgekey_to_group[e_key] = group[group_of_this]

    # then invert the boundary_edgekey_to_group mapping
    group_to_boundary_edgekeys: Dict[int, List[Tuple[int, int]]] = {}
    for e_key, group_of_this in boundary_edgekey_to_group.items():
        if (existing_e_keys := group_to_boundary_edgekeys.get(group_of_this)) is not None:
            existing_e_keys.append(e_key)
        else:
            group_to_boundary_edgekeys[group_of_this] = [e_key]

    # assert set(group_to_boundary_edgekeys.keys()) == set(group_to_face_idxs.keys())

    volume = 0.0

    # code to save out the merged, holeclosed components by themselves
    if VSELVOLDEBUG_SAVE_HOLECLOSED_COMPONENTS:
        _VSELVOLDEBUG_concatted_holeclosed_components = ((), (), 0)
    else:
        _VSELVOLDEBUG_concatted_holeclosed_components = None

    def __VSELVOLDEBUG_update_concat_holeclosed_components(
        oldtup: None | tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], int],
        new_v_for_volume,
        new_f_for_volume,
    ):
        if oldtup is not None:
            oldvs, oldfs, oldvcount = oldtup
            newtup = (
                oldvs + (new_v_for_volume,),
                oldfs + (new_f_for_volume + oldvcount,),
                oldvcount + len(new_v_for_volume),
            )
            return newtup
        else:
            return None

    # for each connected component, close holes and compute volume
    for group_id, face_idxs_in_group in group_to_face_idxs.items():
        bdry_ekeys_in_group = group_to_boundary_edgekeys.get(group_id)

        f_this_group = f[face_idxs_in_group]
        vi_used_this_group = np.unique(f_this_group)
        n_v_this_group = len(vi_used_this_group)
        v_compact_this_group = v[vi_used_this_group]
        vi_old2compact = np.full((len(v),), -99999, dtype=int)
        vi_old2compact[vi_used_this_group] = np.arange(n_v_this_group)
        f_compact_this_group = vi_old2compact[f_this_group]

        if bdry_ekeys_in_group is not None:
            v_for_volume, f_for_volume = meshlab_holecloser(
                v_compact_this_group, f_compact_this_group, len(bdry_ekeys_in_group)
            )
            if thlog.guard(VIZ_TRACE, needs_polyscope=True):
                thlog.psr.register_surface_mesh(
                    f"group{group_id}",
                    v_for_volume,
                    f_for_volume,
                )
                # if VSELVOLDEBUG_SAVE_HOLECLOSED_COMPONENTS:
                #     igl.write_triangle_mesh(
                #         f"holeclosed--group{group_id}.obj", v_for_volume, f_for_volume
                #     )
                _VSELVOLDEBUG_concatted_holeclosed_components = (
                    __VSELVOLDEBUG_update_concat_holeclosed_components(
                        _VSELVOLDEBUG_concatted_holeclosed_components,
                        v_for_volume,
                        f_for_volume,
                    )
                )
            # compute volume of this now-closed connected component
            volume += calc_mesh_volume_estimate(v_for_volume, f_for_volume)
        else:
            # no boundary; this connected component is watertight
            if thlog.guard(VIZ_TRACE, needs_polyscope=True):
                v_for_volume = v
                f_for_volume = f[face_idxs_in_group]
                thlog.psr.register_surface_mesh(
                    f"group{group_id}", v_for_volume, f_for_volume
                )
                # if VSELVOLDEBUG_SAVE_HOLECLOSED_COMPONENTS:
                #     igl.write_triangle_mesh(
                #         f"holeclosed--group{group_id}.obj", v_for_volume, f_for_volume
                #     )
                _VSELVOLDEBUG_concatted_holeclosed_components = (
                    __VSELVOLDEBUG_update_concat_holeclosed_components(
                        _VSELVOLDEBUG_concatted_holeclosed_components,
                        v_for_volume,
                        f_for_volume,
                    )
                )

            volume += calc_mesh_volume_estimate(v, f[face_idxs_in_group])
    thlog.info(f"total volume of hole-closed selection: {volume:.6f}")
    if thlog.guard(VIZ_TRACE, needs_polyscope=True):
        thlog.psr.show()

    if _VSELVOLDEBUG_concatted_holeclosed_components is not None:
        assert isinstance(
            VSELVOLDEBUG_SAVE_HOLECLOSED_COMPONENTS, str
        ) and VSELVOLDEBUG_SAVE_HOLECLOSED_COMPONENTS.endswith(".obj")
        vs, fs, _ = _VSELVOLDEBUG_concatted_holeclosed_components
        igl.write_triangle_mesh(
            VSELVOLDEBUG_SAVE_HOLECLOSED_COMPONENTS,
            np.concatenate(vs, axis=0),
            np.concatenate(fs, axis=0),
        )

    return volume
