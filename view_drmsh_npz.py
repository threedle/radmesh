import numpy as np
import polyscope as ps
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("npzfname", type=str)
parser.add_argument(
    "--orig-albmesh",
    type=str,
    help="npz file containing keys 'vt' and 'ft', UVs for the original mesh, plus optionally 'tex' a texture map of shape (h,w,3) (handles both 0-1 float and 0-255 int)",
)
namespace = parser.parse_args()

with np.load(namespace.npzfname) as npz:
    original_verts = npz.get("original_verts")
    original_faces = npz.get("original_faces")
    original_geometry_xyz_for_current_triangles = npz.get(
        "original_geometry_xyz_for_current_triangles"
    )
    deformed_verts = npz["deformed_verts"]
    deformed_faces = npz["deformed_faces"]
    if cfg_str := str(npz.get("deform_by_csd_cfg")):
        print(cfg_str)
    if cfg_str := str(npz.get("dataset_cfg")):
        print(cfg_str)
    original_vsel = npz.get("original_vsel")
    deformed_vsel = npz.get("deformed_vsel")
    original_piggyback_vattrs = npz.get("original_piggyback_vattrs")
    current_piggyback_vattrs = npz.get("deformed_piggyback_vattrs")
    new2old_fi = npz.get("new2old_fi")

ps.init()
ps.set_ground_plane_mode("shadow_only")
if original_verts is not None and original_faces is not None:
    orig_psmesh = ps.register_surface_mesh(
        "orig", original_verts, original_faces, enabled=False
    )
else:
    orig_psmesh = None
final_psmesh = ps.register_surface_mesh(
    "final", deformed_verts, deformed_faces, smooth_shade=True, edge_width=1.0
)

##############################################
if original_geometry_xyz_for_current_triangles is not None:
    ps.register_surface_mesh(
        "orig with final tris",
        original_geometry_xyz_for_current_triangles,
        deformed_faces,
        enabled=False,
    )
    ps.register_point_cloud(
        "orig with final tris pcl",
        original_geometry_xyz_for_current_triangles,
        enabled=False,
        radius=0.001,
    )

##############################################
if (
    orig_psmesh is not None
    and original_vsel is not None
    and len(original_vsel) == len(original_verts)
):
    # this check is because I messed up saving original_vsel and saved the wrong one (after init remesh of inflation)
    orig_psmesh.add_scalar_quantity(
        "vsel", original_vsel, cmap="plasma", datatype="categorical"
    )
if deformed_vsel is not None and len(deformed_vsel) == len(deformed_verts):
    final_psmesh.add_scalar_quantity(
        "vsel", deformed_vsel, cmap="plasma", datatype="categorical"
    )

##############################################
if new2old_fi is not None and orig_psmesh is not None and final_psmesh is not None:
    orig_psmesh.add_scalar_quantity(
        "fi",
        np.arange(nf := len(original_faces), dtype=int),
        vminmax=(-1, nf),
        defined_on="faces",
    )
    final_psmesh.add_scalar_quantity(
        "new2old_fi", new2old_fi, vminmax=(-1, nf), defined_on="faces"
    )
    if namespace.orig_albmesh:
        with np.load(namespace.orig_albmesh) as vtft_npz:
            vt = vtft_npz["vt"]
            ft = vtft_npz["ft"]
            ct = vt[ft]
            orig_ct = ct.reshape(-1, 2)
            if deformed_vsel is not None:
                deformed_fsel = np.all(deformed_vsel[deformed_faces], axis=1)
                new2old_fi[deformed_fsel] = -1
            final_ct = ct[new2old_fi]
            final_ct[new2old_fi == -1] = -100
            final_ct = final_ct.reshape(-1, 2)
            orig_psmesh.add_parameterization_quantity(
                "orig_ct", orig_ct, defined_on="corners"
            )
            final_psmesh.add_parameterization_quantity(
                "final_ct", final_ct, defined_on="corners"
            )
            tex = vtft_npz.get("tex")
            if tex is not None:
                # guess the range?
                if not isinstance(tex.dtype, np.floating):
                    tex = tex.astype(float) / 255
                if tex.shape[-1] > 3:
                    # rgb only, no alpha
                    tex = tex[:, :, :3]
                ps.add_color_image_quantity("tex", tex, enabled=True)
                orig_psmesh.add_color_quantity(
                    "tex", tex, defined_on="texture", param_name="orig_ct", enabled=True
                )
                final_psmesh.add_color_quantity(
                    "tex", tex, defined_on="texture", param_name="final_ct", enabled=True
                )

ps.show()
