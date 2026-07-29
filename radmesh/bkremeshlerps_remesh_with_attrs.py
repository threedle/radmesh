import sys
import bkremeshlerps

import numpy as np
import torch


def do_isoremesh_with_vertex_selection_and_attributes(
    targetlen: float | torch.Tensor,
    n_iters: int,
    v: torch.Tensor,
    f: torch.Tensor,
    v_selection: torch.Tensor | None,
    v_attributes: torch.Tensor | None,
    do_smooth_step: bool,
    interp_using_barycoords: bool,
    override__adaptive_epsilon: float | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor | None, torch.Tensor | None
]:
    """
    returns
    - v_remeshed
    - f_remeshed
    - new2old_fi, a (len(f_remeshed),) array of indices into f that indicate the origin face
        of each resultant face in the remeshed mesh, or (-1) if a fresh face spawned via
        splitting or flipping.

        NOTE that faces that were mangled by collapses may still have a non-(-1)
        value for its entry in this array. To blot out all possible mangled faces whose
        'origin-face index' may be dubious, blot out and manually set to (-1) all selected
        faces in f_remeshed (i.e. faces with all three verts having 1 for their entry in
        v_selection_interpd).

    - v_selection_interpd (if a selection was given, else None)
    - v_attributes_interpd (if attributes were given, else None)
    """
    targetlen_floatornp = (
        (
            targetlen.cpu().detach().numpy()
            if isinstance(targetlen, torch.Tensor)
            else targetlen
        )
        if override__adaptive_epsilon is None
        else None
    )
    v_and_attributes_np = (
        (torch.cat((v, v_attributes), dim=-1) if v_attributes is not None else v)
        .cpu()
        .detach()
        .numpy()
    )
    f_np = f.cpu().detach().numpy()
    v_selection_np = (
        v_selection.cpu().detach().numpy().astype(bool)
        if v_selection is not None
        else np.ones((len(v_and_attributes_np),), dtype=bool)
    )
    if targetlen_floatornp is not None and override__adaptive_epsilon is None:
        (
            v_and_attributes_remeshed_np,
            f_remeshed_np,
            v_selection_interpd_np,
            fi_containing_v_proj,
            new2old_fi,
        ) = bkremeshlerps.remesh_botsch_with_interps(
            v_and_attributes_np,
            f_np,
            v_selection_np,
            targetlen=targetlen_floatornp,
            selection_threshold=0.5,
            iterations=n_iters,
            smooth=do_smooth_step,
            project=True,
            verbose=True,
        )
    elif targetlen_floatornp is None and override__adaptive_epsilon is not None:
        (
            v_and_attributes_remeshed_np,
            f_remeshed_np,
            v_selection_interpd_np,
            fi_containing_v_proj,
            new2old_fi,
        ) = bkremeshlerps.remesh_botsch_adaptive_with_interps(
            v_and_attributes_np,
            f_np,
            v_selection_np,
            epsilon=override__adaptive_epsilon,
            adaptive=True,
            selection_threshold=0.5,
            iterations=n_iters,
            smooth=do_smooth_step,
            project=True,
            verbose=True,
        )
        if np.any(np.isnan(v_and_attributes_remeshed_np[:, :3])):
            np.savez_compressed(
                "bkremeshlerpcalladaptive.npz",
                v_etc=v_and_attributes_np,
                f=f_np,
                vsel=v_selection_np,
                epsilon=np.array(override__adaptive_epsilon),
                iterations=np.array(n_iters),
            )
            raise ValueError("adaptive remesh gave nans! dumped call for debug")

    else:
        raise ValueError("either adaptive_epsilon or targetlen_floatornp")
    print(
        f"[bkremeshlerps_remesh] remeshed to {len(f_remeshed_np)} faces",
        file=sys.stderr,
        flush=True,
    )
    if interp_using_barycoords:
        barycoords, v_and_attributes_barylerped_np = (
            bkremeshlerps.barycentric_interp_on_Fi_containing_V_proj(
                v_and_attributes_np,
                f_np,
                v_and_attributes_remeshed_np[:, :3],
                fi_containing_v_proj,
            )
        )
        baryvalid = np.logical_and((barycoords >= 0), (barycoords <= 1)).all(axis=-1)
        if override__adaptive_epsilon is None:
            # replace the non-xyz quantities with the bary lerpd ones (xyz is already projected)
            v_and_attributes_remeshed_np[baryvalid, 3:] = v_and_attributes_barylerped_np[
                baryvalid, 3:
            ]
        else:
            # for adaptive_epsilon, (which is prone to bad?projection for some
            # reason), also do this for the xyz (so do it for all coords)
            baryvalid_or_nan = baryvalid | (
                np.any(np.isnan(v_and_attributes_barylerped_np[:, :3]), axis=-1)
            )
            if override__adaptive_epsilon is not None:
                v_and_attributes_remeshed_np[baryvalid_or_nan] = (
                    v_and_attributes_barylerped_np[baryvalid_or_nan]
                )

    v_and_attributes_remeshed = (
        torch.from_numpy(v_and_attributes_remeshed_np).to(v).contiguous()
    )
    v_remeshed = v_and_attributes_remeshed[:, :3]
    f_remeshed = torch.from_numpy(f_remeshed_np).to(f).contiguous()
    v_attributes_remeshed = (
        v_and_attributes_remeshed[:, 3:] if v_attributes is not None else None
    )
    v_selection_interpd = (
        torch.from_numpy(v_selection_interpd_np.astype(bool)).to(v_selection.device)
        if v_selection is not None
        else None
    )
    assert isinstance(new2old_fi, np.ndarray)
    return v_remeshed, f_remeshed, new2old_fi, v_selection_interpd, v_attributes_remeshed
