from typing import Optional, Callable, Tuple, Literal, Sequence, Union, List, Dict, Any
from typing import cast
from dataclasses import dataclass, field
from functools import cached_property
import os
import time

import numpy as np
import torch
from .pytorch3d import ops as pt3d_ops
from .pytorch3d import structures as pt3d_structures

import igl

import nvdiffrast.torch as dr

from .nvdiffmodeling.src import mesh as nvdiffmodeling_mesh
from .nvdiffmodeling.src import render as nvdiffmodeling_render
from .nvdiffmodeling.src import texture as nvdiffmodeling_texture

from . import resize_right, td_camera
from . import csd, bkremeshlerps_remesh_with_attrs, vselection_volume
from . import deformations

from .misc_helpers import (
    parse_lr_schedule_string_into_lr_lambda,
    next_increment_path,
    get_bool_env_variable,
)
from thronf import Thronfig, thronfigure, InvalidConfigError
import polyscope as ps
from thlog import (
    Thlogger,
    LOG_INFO,
    LOG_DEBUG,
    VIZ_INFO,
    VIZ_DEBUG,
    PSRSpecialArray,
    _PolyscopeRegisteredStructProxy,
)

thlog = Thlogger(
    LOG_INFO,
    VIZ_INFO,
    "radmesh",
    imports=[deformations.thlog, csd.thlog, vselection_volume.thlog],
)

DRMSHDEBUG_PREPOSTREMESHVIZ = get_bool_env_variable("DRMSHDEBUG_PREPOSTREMESHVIZ")
DRMSHDEBUG_SAVE_YZROT = get_bool_env_variable("DRMSHDEBUG_SAVE_YZROT")
"""
whether to save a rotated version of the initial-inflated, remeshed shape (and a selection,
saved with savetxt) for use with methods that expect the z-up convention
"""

OptimizerTypeName = Literal["Adam", "SGD"]

SupportedOptimizerType = Union[torch.optim.Adam, torch.optim.SGD]


@dataclass(slots=True)
class MeshesDatasetAsFolder_IOSettings(Thronfig):
    """not implemented yet, use MeshesDatasetAsList"""

    path: str
    prompts_file: str


@dataclass(slots=True)
class MeshesDatasetAsList_IOSettings(Thronfig):
    fnames: Sequence[str]
    """
    a list of source .obj filenames which must all exist
    """
    prompts: Sequence[str]
    prompts_negative: Sequence[Optional[str]]

    vertex_selection_fnames: Optional[Sequence[Optional[str]]] = None
    """
    each fname (corresponding to a mesh in fnames) is a path to an .npy boolean array
    with shape (n_verts,), where 1 indicates the vertex is 'selected/enabled' for deformation.
    This is used for localized deformation optimization.
    """
    other_vertex_attributes_fnames: Optional[Sequence[str]] = None

    def __post_typecheck__(self):
        if not (
            (n_fnames := len(self.fnames)) == len(self.prompts)
            and n_fnames == len(self.prompts_negative)
            and (
                self.vertex_selection_fnames is None
                or n_fnames == len(self.vertex_selection_fnames)
            )
        ):
            raise InvalidConfigError(
                "in dataset_cfg.lists, the lists fnames, prompts, prompts_negative, (and vertex_selection_fnames if present) lists must all have the same number of elements"
            )


@dataclass(slots=True)
class MeshesDataset_Settings(Thronfig):
    """
    at least one of folder or lists must be specified.
    """

    folder: Optional[MeshesDatasetAsFolder_IOSettings] = None
    lists: Optional[MeshesDatasetAsList_IOSettings] = None

    def __post_typecheck__(self):
        if self.folder is None and self.lists is None:
            raise InvalidConfigError(
                "one of folder or lists must be non-None in meshes dataset io config"
            )

    def get_dataset_size(self) -> int:
        if self.lists is not None:
            # self.lists's post typecheck already validated it to have the same number of
            # fnames, prompts
            return len(self.lists.fnames)
        elif self.folder is not None:
            raise NotImplementedError("TODO implement folder dataset config")
        else:
            return 0


@dataclass(slots=True)
class BKRemeshLerps_RunSettings:
    method_cfg: "BKRemeshLerps_MethodSettings"
    n_iters: int
    use_avglen: Literal["original", "latest", "post_init_remesh"]
    make_targetlen_from_avglen: Callable[[float], float]
    override__adaptive_epsilon: Union[float, None]


@dataclass(slots=True)
class BKRemeshLerps_MethodSettings(Thronfig):
    targetlen_schedule: str
    n_iters: int
    interp_using_barycoords: bool
    do_smooth_step: bool
    """
    basically should always be true
    """
    targetlen_as_multiplier_of_avglen: bool
    use_avglen: Literal["original", "latest", "post_init_remesh"] = "original"
    """
    if True AND targetlen_as_multiplier_of_avglen is True,
    the targetlen schedule value will be interpreted as a multiplier for
    the current avglen, not the original avglen
    """
    targetlen_for_init_remesh: Optional[float] = None
    """
    this is for the remesh done before any optim loop happens.
    If None, then will be whatever targetlen_schedule indicates for remesh 0
    (the first one in the optim loop)
    """
    n_iters_for_init_remesh: Optional[int] = None
    """
    this is for the remesh doene before any optim loop happens. n_iters if None
    """
    override__adaptive_epsilon: Union[float, str, None] = None
    """ if present will use this epsilon for adaptive remeshing """

    def __post_typecheck__(self):
        if self.use_avglen == "latest" and not self.targetlen_as_multiplier_of_avglen:
            raise InvalidConfigError(
                "use_avglen='latest' must go with targetlen_as_multiplier_of_avglen=True"
            )

    @cached_property
    def targetlen_fn(self) -> Callable[[int], float]:
        return parse_lr_schedule_string_into_lr_lambda(self.targetlen_schedule)

    @cached_property
    def epsilon_fn(self) -> Callable[[int], Optional[float]]:
        if self.override__adaptive_epsilon is None:
            return lambda _: None
        elif isinstance((eps := self.override__adaptive_epsilon), (float, int)):
            return lambda _: eps
        else:
            return parse_lr_schedule_string_into_lr_lambda(eps)

    def _make_targetlen_from_avglen(
        self,
        targetlenfn_out: float,
        avglen: float,
    ) -> Union[float, torch.Tensor]:
        targetlen = (
            (targetlenfn_out * avglen)
            if self.targetlen_as_multiplier_of_avglen
            else targetlenfn_out
        )
        thlog.debug(f"computed targetlen this remesh: {targetlen}")
        return targetlen

    def get_run_settings_for_ith_remesh(
        self,
        remesh_i: int,
        original_local_step_procrustes_settings: Optional[deformations.Procrustes_Settings],
    ) -> Tuple[
        BKRemeshLerps_RunSettings,
        Optional[deformations.Procrustes_Settings],
    ]:
        # -1 indicates init remesh before the optim loop, handle that separately
        targetlenfn_out = (
            (
                self.targetlen_for_init_remesh
                if self.targetlen_for_init_remesh is not None
                else self.targetlen_fn(0)  # use first actual remesh settings
            )
            if remesh_i < 0
            else self.targetlen_fn(remesh_i)
        )
        n_iters = (
            (
                (
                    thlog.debug(f"using {self.n_iters_for_init_remesh} for init remesh"),
                    self.n_iters_for_init_remesh,
                )[-1]
                if self.n_iters_for_init_remesh is not None
                else self.n_iters
            )
            if remesh_i < 0
            else self.n_iters
        )
        epsilonfn_out = self.epsilon_fn(0) if remesh_i < 0 else self.epsilon_fn(remesh_i)
        thlog.debug(f"epsilon {epsilonfn_out}")
        return BKRemeshLerps_RunSettings(
            self,
            n_iters=n_iters,
            use_avglen=self.use_avglen,
            make_targetlen_from_avglen=(
                lambda avglen: self._make_targetlen_from_avglen(targetlenfn_out, avglen)
            ),
            override__adaptive_epsilon=epsilonfn_out,
        ), original_local_step_procrustes_settings


@dataclass(slots=True)
class Inflation_Settings(Thronfig):
    inflate_along_normals: Optional[float]
    inflate_offset_smooth_lambda: float
    inflate_offset_smooth_iters: int
    vsel_smooth_lambda: float
    vsel_smooth_iters: int


@dataclass(slots=True)
class Remeshing_Settings(Thronfig):
    remesh_schedule: Union[int, Sequence[int]]
    """
    if an int, remesh once every that many optim iters; if a sequence of
    ints, run the remeshing method at those iterations of the main optim loop
    """

    use_interpd_optimizer_state: bool
    """
    put the deformation quantity optimizer state into the remeshing run
    as vertex features to interpolate during remeshing and resume with that optim state
    """
    adjust_optimizer_state_using_interpd_deformqty: bool
    """
    put the deformation quantity into the remeshing run as vertex features to
    interpolate during remeshing and resume with that as the start deformation quantity
    """
    bkremeshlerps: BKRemeshLerps_MethodSettings
    """settings for Botsch-Kobbelt remeshing with attribute interpolations"""
    override__remesh_only_once_at_start: bool = False
    """
    if this is True, then IGNORE the remesh_schedule setting above, and only remesh
    ONCE before any deform optimization, and never remesh again.
    """
    override__no_remesh_once_at_start: bool = False
    """
    by default, there is a remesh before deform optimization if the remesh config is present.
    but specify this to disable that remesh (but allow other subsequent remeshes)
    """
    save_at_remesh_i: Sequence[int] = ()
    """
    save result of the remeshes at these remesh_i indices (0-indexed)
    """

    def __post_typecheck__(self):
        if (
            self.adjust_optimizer_state_using_interpd_deformqty
            and not self.use_interpd_optimizer_state
        ):
            raise InvalidConfigError(
                "if adjust_optimizer_state_using_interpd_deformqty is true, then use_interpd_optimizer_state must be true"
            )

        if (
            self.override__no_remesh_once_at_start
            and self.override__remesh_only_once_at_start
        ):
            raise InvalidConfigError(
                "can't have both overrides no_remesh_once_at_start and remesh_only_once_at_start"
            )
            # practically having both will just cause there to be no remeshing at all. but
            # then might as well delete the remeshing config

    def get_run_settings_for_ith_remesh(
        self,
        remesh_i: int,
        original_local_step_procrustes_settings: Optional[deformations.Procrustes_Settings],
    ):
        method_cfg = self.bkremeshlerps
        return method_cfg.get_run_settings_for_ith_remesh(
            remesh_i, original_local_step_procrustes_settings
        )

    def get_remesh_iter_number(self, optim_i: int, optim_n_iters: int) -> Optional[int]:
        if isinstance(self.remesh_schedule, Sequence):
            try:
                # the (remesh_i)th remesh
                remesh_i = self.remesh_schedule.index(optim_i)
            except ValueError:
                remesh_i = None
        else:
            if (optim_i % self.remesh_schedule) == 0 or optim_i == optim_n_iters:
                # force a remesh after the last optim iter, so that the save result really
                # is the final deform+remesh result! (optim_i is 1-indexed, hence ==)
                remesh_i = optim_i // self.remesh_schedule
            else:
                remesh_i = None
        return remesh_i


@dataclass(slots=True)
class SelectionRegionPinning_Settings(Thronfig):
    fpsamps_if_hardpin: Optional[int] = None
    """
    if not None, take this many furthest-point samples out of non-selected vertices to pin
    if None, pin all nonselected
    """
    hardpin: bool = True
    """
    if True, reduces the poisson system by removing the pinned vertices outright
    if False, only sets their matrices to identity before the solve (this is
    also done when True for computation of the rhs)
    """


@dataclass(slots=True)
class VolumeBasedHeuristicTarget_Settings(Thronfig):
    formula: Literal[
        "powercbrt1_replace", "powercbrt1_add", "mulcbrt1_replace", "cbrtdivvol1_replace"
    ]
    """
    choices:
    - powercbrt1_replace:   mul * (a ** (-b * cbrt(vol)))
    - powercbrt1_add:       mul * (a ** (-b * cbrt(vol))) + base
    - mulcbrt1_replace:     mul * (a * cbrt(b * vol))
    - cbrtdivvol1_replace:  mul * cbrt(a / vol)
        (b is unused and ignored. a is meant to be a 'max allowed volume' so
        that the result of this formula represents the cbrt of the ratio between
        the max allowed volume and the current volume)
    """
    a: float
    b: float = 1
    mul: float = 1
    min: float = -np.inf
    max: float = np.inf
    default_on_none_volume: float = 0
    """ value for when volume is unavailable, e.g. volume computation failed """
    use_orig_volume_instead_of_curr_volume: bool = False
    """
    use original (pre-initial-inflate!) volume as input to the formula
    instead of the "current" selection volume, calculated at the latest remesh
    """

    def __post_typecheck__(self):
        if self.default_on_none_volume < self.min or self.default_on_none_volume > self.max:
            raise InvalidConfigError("defualt_on_none_value must fall in [min, max]")

    def __guard_lamb(
        self, fn: Callable[[float, float], float]
    ) -> Callable[[float, Optional[float]], float]:
        def __guard(base: float, vol: Optional[float]):
            if vol is not None:
                return np.clip(fn(base, vol), self.min, self.max)
            else:
                return self.default_on_none_volume

        return __guard

    @cached_property
    def calc(self) -> Callable[[float, Optional[float]], float]:
        if self.formula == "cbrtdivvol1_replace":
            return self.__guard_lamb(
                lambda base, volume: self.mul * np.cbrt(self.a / volume)
            )
        elif self.formula == "powercbrt1_replace":
            return self.__guard_lamb(
                lambda base, volume: self.mul * (self.a ** (-self.b * np.cbrt(volume)))
            )
        elif self.formula == "powercbrt1_add":
            return self.__guard_lamb(
                lambda base, volume: (
                    base + self.mul * (self.a ** (-self.b * np.cbrt(volume)))
                )
            )
        elif self.formula == "mulcbrt1_replace":
            return self.__guard_lamb(
                lambda base, volume: self.mul * self.a * np.cbrt(self.b * volume)
            )
        else:
            raise InvalidConfigError(f"unknown formula {self.formula}")


@dataclass(slots=True)
class VolumeBasedHeuristic_Settings(Thronfig):
    initial_normal_inflate: Optional[VolumeBasedHeuristicTarget_Settings] = None
    vns_scale_id_loss_weight: Optional[VolumeBasedHeuristicTarget_Settings] = None
    vns_scale_max: Optional[VolumeBasedHeuristicTarget_Settings] = None


@dataclass(slots=True)
class Optimizer_Settings(Thronfig):
    type: OptimizerTypeName
    betas: Tuple[float, float] = (0.9, 0.999)


@dataclass(slots=True)
class CSD_Settings(Thronfig):
    """settings specific to csd guidance method"""

    view_batch_size: int
    """
    For each of the patient meshes to be used in a batch, the number of views to render each
    one. This means the final effective batch size that goes in an iteration will be
    (mesh_batch_size * view_batch_size), where each batch istem is a pair (mesh, view).
    """
    adapt_dists: bool
    """
    if True, will apply distance multipliers which are the meshes'
    origin-centered bounding boxes' max-magnitude coord values
    """
    cams_and_lights: td_camera.CamsAndLights_Settings
    background_color: Tuple[float, float, float]
    resize_for_guidance: Optional[Tuple[int, int, resize_right.InterpMethodName]]
    """
    if not None, resize the raw render image to the desired size and with the
    specified kernel before feeding it to the CSD/SDS guidance
    """
    stage_I_weight: float
    """
    should just be 1.0
    """
    stage_II_weight_schedule: str
    """
    a ramp schedule description string (see
    parse_lr_schedule_string_into_lr_lambda in misc_helpers.py); if not None, overrides
    stage_II_weight and uses the function described in the schedule string.
    For a constant value simply write the float value in that string, e.g. "0.0"
    """
    models: csd.GuidanceConfig = field(default_factory=csd.GuidanceConfig)
    """
    further diffusion model-related settings expected by the csd module (model
    choice and config, classifier-free weight, etc)
    """


@dataclass(slots=True)
class DeformByCSD_Settings(Thronfig):
    optimize_deform_via: deformations.DeformOptimQuantityName
    """
    - verts_offsets: only optimize the vertex offsets, no other solve
    - faces_normals: optimize normals to compute axis-angle rot matrices (taking
        face normals from curr normals to the optimized normals) and then
        solving into vertices based on the solve_method (using a face-based solve_method)
    - faces_jacobians: optimize per-face transform matrices, and then solving
        into vertices based on the solve_method (using a face-based solve_method)
    - verts_normals: optimize normals, compute neighborhood rot matrices by Procrustes
        local step then solving into vertices based on the solve_method
        (using a vert-based solve_method)
    - verts_normals_and_scale: adds xyz scale parameters to verts_normals. Procrustes solve
        gives the rotations, then diag(scale) gives the scaling applied after the rotations.
    - verts_3x2rotations: optimize per-vertex 3x2 continuous rotation representation mats,
        to be converted with a soft-diagonalization into 3x3 rotations to be solved into
        vertices based on the solve_method (only works for a vert-based solve_method)
    - faces_3x2rotations: ditto but per-face (for face-based solve_method)
    - verts_jacobians: per-vertex 3x3 transforms.
    """
    solve_method: deformations.DeformSolveMethodName
    """
    the method used to solve per-element transforms into vertex offsets
    """
    arap_energy_type: Optional[deformations.ARAPEnergyTypeName]
    """
    if solve_method has 'arap' in it, then this must be non-None, and vice versa
    """
    postprocess_after_solve: Optional[deformations.PostprocessAfterSolveName]
    """
    postprocess the solve result; 'recenter_only' recommended when a vertex is pinned
    """
    local_step_procrustes: Optional[deformations.Procrustes_Settings]
    """
    if not None, use procrustes solve. Applicable for the verts_normals and
    verts_normals_and_scale optimize_deform_via Since we're always using one
    step of procrustes (i.e. always starts from initial verts) this can probably
    be a bit bigger than 1.0.
    """
    n_iters: int
    lr: Union[float, str]
    """
    use a float for a fixed LR; use a string for a LR schedule spec string
    """

    optimizer: Optimizer_Settings

    n_accum_iters: int
    """
    n of iters of SDS/CSD gradients to accumulate and do backward on per epoch.
    """
    step_after_every_backward: bool
    """
    whether to optimizer.step() after each backward() in every accum iter.
    If False, the optimizer.step() is done only once after the accum loop, i.e. after all
    backward()s.
    """
    optimized_quantity_save_fname: str
    """
    save fname for the optimized quantity (what gets saved depends on optimize_deform_via)
    """

    view_once_every: int
    """
    how often to log to the ps recording during optimization
    """

    mesh_batch_size: int
    """
    number of patient meshes to be used in a batch
    """
    visual_loss_weight_schedule: str
    """
    a ramp schedule description string for the weight of the visual loss.
    applies to CSD loss or inverse rendering loss depending on guidance type
    """

    jacobian_id_loss_weight_schedule: str
    """
    a ramp schedule description string for the weight of the jacobian identity
    regularization loss. Works for all optimize_deform_via not just faces_jacobians
    """
    verts_normals_and_scale__scale_id_loss_weight_schedule: str
    """
    a ramp schedule string for the weight of the scale identity loss for the scale
    parameters in verts_normals_and_scale deformation quantity.
    """
    csd_guidance: CSD_Settings
    """
    settings for CSD guidance
    """
    pin_first_vertex: bool = True
    """
    if True, must be an alias for, and co-occur with, pin_special_vertex = "vertex_0"
    if False, the value of pin_special_vertex will be used.
    """
    pin_special_vertex: deformations.NameOfSpecialVertexToPin = "vertex_0"
    """
    Can specify "vertex_0", "min_z", "max_z", "min_y", "max_y", "min_x", "max_x" to pin those points
    instead of the first vertex in the array index order of each mesh connected component
    """
    pin_nonselected: SelectionRegionPinning_Settings = field(
        default_factory=SelectionRegionPinning_Settings
    )
    """
    settings for what to do with the nonselected vertices w.r.t. pinning them in the system
    """
    adjust_things_via_selection_volume: VolumeBasedHeuristic_Settings = field(
        default_factory=VolumeBasedHeuristic_Settings
    )
    """
    heuristic for adjusting stuff with selection volume
    """
    start_from_epoch: int = 1
    """
    this is 1-indexed, 1 is the first epoch.
    Practically useful for starting the optimization at a certain epoch with respect to the
    loss weight schedule functions
    """
    torch_seed: Optional[int] = None
    numpy_seed: Optional[int] = None
    save_at_epochs: Sequence[int] = ()
    """
    save results at the specified epochs
    """
    initial_inflate: Optional[Inflation_Settings] = None
    """
    because darap tends to preserve volume: inflate the mesh/selected region
    """
    periodic_remeshing: Optional[Remeshing_Settings] = None

    @property
    def view_batch_size(self) -> int:
        return self.csd_guidance.view_batch_size

    def __post_typecheck__(self):
        """some basic config validation"""
        if self.pin_special_vertex != "vertex_0":
            # as a courtesy. i don't want to specify a pin_special_vertex but then also have
            # to specify the DEPRECATED pin_first_vertex in order to not trip the next test.
            # so i will set it here. because the intention with a non-vertex_0
            # pin_special_vertex means that the user took care to change it from the
            # default. so this should also change from the default True.
            self.pin_first_vertex = False

        if self.pin_first_vertex or self.pin_special_vertex == "vertex_0":
            if not (self.pin_special_vertex == "vertex_0" and self.pin_first_vertex):
                raise InvalidConfigError(
                    "if pin_first_vertex, then pin_special_vertex must be vertex_0, and vice versa"
                )

        # arap_energy_type
        has_arap_energy_type = self.arap_energy_type is not None
        solvemethod_is_arap = self.solve_method == "arap"
        if has_arap_energy_type or solvemethod_is_arap:
            if not (has_arap_energy_type and solvemethod_is_arap):
                raise InvalidConfigError(
                    "if deform_by_csd.arap_energy_type is specified, then solve_method must be 'arap' and vice versa"
                )

        if self.local_step_procrustes is not None:
            if (
                self.optimize_deform_via != "verts_normals"
                and self.optimize_deform_via != "verts_normals_and_scale"
            ):
                raise InvalidConfigError(
                    "can only use procrustes lambda with verts_normals optimize_deform_via"
                )

        if self.optimize_deform_via == "verts_offsets" and self.solve_method is not None:
            raise InvalidConfigError(
                "solve_method must be None/null when optimize_deform_via verts_offsets"
            )


@dataclass(slots=True)
class MainConfig(Thronfig):
    dataset: MeshesDataset_Settings
    deform_by_csd: DeformByCSD_Settings
    ps_recording_save_fname: str
    device: Literal["cuda", "cpu"] = "cuda"

    def __post_typecheck__(self):
        if self.dataset.get_dataset_size() > 1:
            raise InvalidConfigError(
                """
    when optimizing multiple mesh deformations via per-element normals/rotations/jacobians
    for each mesh, and not via a shared deformer module to be trained on a dataset, it's
    usually faster and more practical to just launch runs in parallel rather than trying to
    do one run with a mesh dataset of more than one shape, since the optimizations of each
    mesh don't have anything shared between each other.
            """
            )


def make_optimizer(
    quantity_being_optimized: deformations.QuantityBeingOptimized,
    main_init_lr: float,
    optimizer_cfg: Optimizer_Settings,
) -> SupportedOptimizerType:
    if optimizer_cfg.type == "Adam":
        paramgroups = (
            {
                "params": (quantity_being_optimized.tensor,),
                "lr": main_init_lr,
                "this_is": "main",
            },
        )
        return torch.optim.Adam(paramgroups, betas=optimizer_cfg.betas)
    elif optimizer_cfg.type == "SGD":
        thlog.info("initializing SGD")
        paramgroups = (
            {
                "params": (quantity_being_optimized.tensor,),
                "lr": main_init_lr,
                "this_is": "main",
            },
        )
        return torch.optim.SGD(paramgroups, lr=main_init_lr)  # no momentum
    else:
        raise InvalidConfigError(f"unknown optimizer type {optimizer_cfg.type}")


def set_learning_rate(
    optimizer: deformations.torch.optim.Optimizer,
    main_lr: float,
    other_shared_params_lr: float,
):
    for param_group in optimizer.param_groups:
        if (this_is := param_group["this_is"]) == "main":
            param_group["lr"] = main_lr
        elif this_is == "other":
            param_group["lr"] = other_shared_params_lr
        else:
            raise ValueError("unknown this_is value in param")


@dataclass(slots=True)
class BaseMeshGoingIntoOptimLoop:
    """
    all the information pertaining to a single patient mesh (equivalent to a
    single .obj file) to be operated upon.
    """

    nvdm_loaded_mesh: nvdiffmodeling_mesh.Mesh
    prompt: str
    prompt_negative: Optional[str]
    prompt_z: torch.Tensor
    """ embedding of prompt, of shape (1, 77, 4096) """
    prompt_negative_z: torch.Tensor
    """ embedding of negative prompt, of shape (1, 77, 4096) """

    vertex_selection_mask: Optional[torch.Tensor]
    """ if present: (n_verts,) bool array loaded from .npy, 1 where deformation is 'enabled' and 0 otherwise """

    piggyback_vertex_attributes: Optional[torch.Tensor]

    original_selection_sum_volume: Optional[float]
    """
    an approximation of the original loaded-from-file selection sum volume (on a normalized,
    centered mesh), if a selection was available, and if the volume can be approximated.
    """
    current_selection_sum_volume: Optional[float]
    """
    reflects the sum volume of the latest selection, i.e. the current vertex_selection_mask,
    and not the original mask loaded from file. (When we remesh, we replace the mask)
    """
    original_loaded_source_n_faces: int
    """ n of faces of the original source mesh """

    original_average_edge_length: float
    """ average edge length (after initial normalization of the loaded mesh) """

    current_average_edge_length: float
    """ average edge length, this current latest source mesh """

    postinitremesh_avg_edge_length: float
    """ average edge length (after the init normalization AND init remesh, but
    before any optimization)"""

    new2old_fi: np.ndarray
    """
    new2old_fi, a (len(f),) array of indices into the original mesh's face array
    that indicate the origin face of each resultant face in the current mesh,
    or (-1) if a fresh face spawned via splitting or flipping.

    2025-12-16: NOTE that faces that were mangled by collapses may still have a non-(-1)
    value for its entry in this array. To blot out all possible mangled faces whose
    'origin-face index' may be dubious, blot out and manually set to (-1) all selected
    faces in f_remeshed (i.e. faces with all three verts having 1 for their entry in
    v_selection_interpd).
    """


def grab_vselection_mask_packed_from_meshes_structs(
    meshes_structs: Sequence[BaseMeshGoingIntoOptimLoop],
) -> torch.Tensor:
    return torch.cat(
        tuple(
            (
                mesh_struct.vertex_selection_mask
                if mesh_struct.vertex_selection_mask is not None
                else torch.ones(
                    (len(v := cast(torch.Tensor, mesh_struct.nvdm_loaded_mesh.v_pos)),),
                    dtype=torch.bool,
                    device=v.device,
                )
            )
            for mesh_struct in meshes_structs
        ),
        dim=0,
    )


def grab_piggyback_vertex_attributes_packed_from_meshes_structs(
    meshes_structs: Sequence[BaseMeshGoingIntoOptimLoop],
) -> Optional[torch.Tensor]:
    cats = tuple(
        sum(
            (
                (mesh_struct.piggyback_vertex_attributes,)
                if mesh_struct.piggyback_vertex_attributes is not None
                else ()
                for mesh_struct in meshes_structs
            ),
            start=(),
        )
    )
    return torch.cat(cats, dim=0) if len(cats) == len(meshes_structs) else None


def grab_current_and_original_selection_sum_volumes_for_vselvol_heuristics_from_meshes_structs(
    meshes_structs: Sequence[BaseMeshGoingIntoOptimLoop],
) -> Sequence[Tuple[Optional[float], Optional[float]]]:
    return tuple(
        (
            mesh_struct.current_selection_sum_volume,
            mesh_struct.original_selection_sum_volume,
        )
        for mesh_struct in meshes_structs
    )


def calc_deformed_verts_according_to_cfg(
    meshes_structs: Sequence[BaseMeshGoingIntoOptimLoop],
    pt3d_batched_meshes: pt3d_structures.Meshes,
    solve_method: deformations.DeformSolveMethodName,
    arap_energy_type: Optional[deformations.ARAPEnergyTypeName],
    postprocess: Optional[deformations.PostprocessAfterSolveName],
    my_solvers: deformations.SparseLaplaciansSolvers,
    quantity_being_optimized: deformations.QuantityBeingOptimized,
) -> Tuple[Sequence[torch.Tensor], deformations.DeformationIntermediateResults]:
    """
    wraps deformations.calc_deformed_verts_solution_according_to_cfg to
    handle the input-to-deformation-solve-calculating

    returns
    - a list of solution verts tensors, each corresponding to a struct in meshes_structs
    - intermediate results, present or None depending on the quantity_being_optimized, in
        case we wish to penalize or view some intermediate result involved in a deform method

    pt3d_batched_meshes must be a pytorch3d Meshes batch with the same number of
    meshes as len(meshes_structs), and each mesh in pt3d_batched_meshes must
    match the vertex (v_pos) and face (t_pos_idx) array of the corresp. mesh struct's nvdm_loaded_mesh
    """
    inputs_to_deformation_solve_methods = deformations.calc_inputs_to_solve_for_deformation_according_to_cfg(
        pt3d_batched_meshes,
        quantity_being_optimized,
        grab_vselection_mask_packed_from_meshes_structs(meshes_structs),
        grab_current_and_original_selection_sum_volumes_for_vselvol_heuristics_from_meshes_structs(
            meshes_structs
        ),
    )

    return deformations.calc_deformed_verts_solution_according_to_cfg(
        pt3d_batched_meshes,
        solve_method,
        arap_energy_type,
        postprocess,
        my_solvers,
        inputs_to_deformation_solve_methods,
    )


def prep_nvdm_mesh_with_trivial_gray_tex(
    verts: torch.Tensor, faces: torch.Tensor
) -> nvdiffmodeling_mesh.Mesh:
    """
    create trivial UVs and init a nvdiffmodeling mesh structure for rendering
    """
    # NOTE these are trivial degenerate UVs pointing to all (0,0), which is fine since we
    # just need to color everything gray for our use.
    # If we need to learn the texture map, then these should be actual good UVs manually
    # loaded from an obj file, or from some param method.
    mega_trivial_vertex_uvs = torch.zeros(
        (verts.shape[0], 2), dtype=torch.float32, device="cuda"
    )
    mega_trivial_t_tex_idx = faces.clone().cuda()

    grayscale_color = 0.5

    # technically trainable but actually we don't update these
    texture_map = nvdiffmodeling_texture.create_trainable(
        np.full((512, 512, 3), grayscale_color, np.float32), [512] * 2, True
    )
    normal_map = nvdiffmodeling_texture.create_trainable(
        np.array([0, 0, 1]), [512] * 2, True
    )
    specular_map = nvdiffmodeling_texture.create_trainable(
        np.array([0, 0, 0]), [512] * 2, True
    )

    material = {
        "bsdf": "diffuse",
        "kd": texture_map,
        "ks": specular_map,
        "normal": normal_map,
    }
    return nvdiffmodeling_mesh.Mesh(
        v_pos=verts.float().cuda(),
        t_pos_idx=faces.cuda(),
        material=material,
        v_tex=mega_trivial_vertex_uvs,
        t_tex_idx=mega_trivial_t_tex_idx,
    )


def load_meshes_from_dataset_cfg_and_encode_prompts(
    dataset_cfg: MeshesDataset_Settings,
    prompt_encoding_fn: Callable[[str, Optional[str]], Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> Tuple[Sequence[BaseMeshGoingIntoOptimLoop], pt3d_structures.Meshes]:
    """
    loads the dataset as a list of wrapper structs representing patient meshes
    to be operated upon, as well as those same meshes but incorporated into a
    pt3d_structures.Meshes batched meshes object for easy packed operations

    optionally (if shape_prep_cfg is present) apply some preprocessing, namely
    aligning shapes to principal axes plus an extra rotation or matrix for our conventions

    because the per-mesh structs also contain prompts and prompt embeddings,
    we also need a prompt_encoding_fn. In our case that's the `encode_prompt`
    method of the CSD class from `csd`.
    """
    if dataset_cfg.folder is not None:
        raise NotImplementedError(
            "TODO implement read directory of meshes and json prompt specification file"
        )
    if dataset_cfg.lists is None:
        raise InvalidConfigError(
            "only dataset_cfg.lists supported for now, so it must be present"
        )
    fnames = dataset_cfg.lists.fnames
    prompts = dataset_cfg.lists.prompts
    prompts_negative = dataset_cfg.lists.prompts_negative
    vertex_selection_fnames = dataset_cfg.lists.vertex_selection_fnames
    piggyback_vertex_attr_fnames = dataset_cfg.lists.other_vertex_attributes_fnames
    mesh_structs = []
    # we'll use a pytorch3d meshes batch to get a packed form of the quantity to optimize
    pt3d_verts_list = []
    pt3d_faces_list = []
    for i, (fname, prompt, prompt_negative) in enumerate(
        zip(fnames, prompts, prompts_negative)
    ):
        loaded_verts_np, loaded_faces_np = igl.read_triangle_mesh(fname)
        nvdm_loaded_mesh = prep_nvdm_mesh_with_trivial_gray_tex(
            torch.from_numpy(loaded_verts_np).to(device),
            torch.from_numpy(loaded_faces_np).to(device),
        )

        # before we normalize with unit_size, do preprocess (alignments etc) if needed

        # then normalize (this is actually the same normalization method as our
        # sphuncs.normalize_to_side2_cube_inplace, i.e. side-2 cube centered at origin)
        nvdm_loaded_mesh = nvdiffmodeling_mesh.unit_size(nvdm_loaded_mesh)

        assert isinstance(nvdm_loaded_mesh.v_pos, torch.Tensor)
        assert isinstance(nvdm_loaded_mesh.t_pos_idx, torch.Tensor)

        nvdm_loaded_mesh.v_pos -= nvdm_loaded_mesh.v_pos.mean(dim=0)

        # encode prompts
        prompt_z, prompt_negative_z = prompt_encoding_fn(prompt, prompt_negative)

        # load the vertex selection if present
        # (we can make this work for face-based deform quantities: faces with all 3 verts
        # selected are treated as selected)
        vertex_selection_mask = None
        original_selection_sum_volume = None
        if vertex_selection_fnames is not None:
            vertex_selection_fname = vertex_selection_fnames[i]
            if vertex_selection_fname is not None:
                vertex_selection_mask_np = np.load(vertex_selection_fname)
                vertex_selection_mask = torch.from_numpy(vertex_selection_mask_np).to(
                    dtype=torch.bool, device=device
                )
                assert vertex_selection_mask.shape == (
                    (_nv := nvdm_loaded_mesh.v_pos.size(0)),
                ), (
                    f"vertex selection mask file {vertex_selection_fname} does not contain the expected shape (n_verts,) ({_nv},) for mesh {fname}"
                )

                # compute volume. This v_pos has been normalized to side2 cube, centered
                v_pos_np = nvdm_loaded_mesh.v_pos.cpu().detach().numpy()
                original_selection_sum_volume = (
                    vselection_volume.calc_holeclosed_vselection_volume(
                        v_pos_np,
                        cast(np.ndarray, loaded_faces_np),
                        vertex_selection_mask_np,
                    )
                )
                thlog.debug(f"original sel sum vol: {original_selection_sum_volume:.6g}")

        piggyback_vertex_attr = None
        if piggyback_vertex_attr_fnames is not None:
            piggyback_vertex_attr_fname = piggyback_vertex_attr_fnames[i]
            piggyback_vertex_attr = np.load(piggyback_vertex_attr_fname)
            _nv = nvdm_loaded_mesh.v_pos.size(0)
            assert (
                len(piggyback_vertex_attr) == _nv and len(piggyback_vertex_attr.shape) == 2
            ), (
                f"other vertex attribs file {piggyback_vertex_attr_fname} does not contain the expected 2-dim shape (n_verts={_nv},n_other_features) for mesh {fname}"
            )
            piggyback_vertex_attr = torch.from_numpy(piggyback_vertex_attr).to(
                dtype=torch.float, device=device
            )

        mesh_struct = BaseMeshGoingIntoOptimLoop(
            nvdm_loaded_mesh=nvdm_loaded_mesh,
            prompt=prompt,
            prompt_negative=prompt_negative,
            prompt_z=prompt_z,
            prompt_negative_z=prompt_negative_z,
            vertex_selection_mask=vertex_selection_mask,
            piggyback_vertex_attributes=piggyback_vertex_attr,
            original_loaded_source_n_faces=nvdm_loaded_mesh.t_pos_idx.size(0),
            original_average_edge_length=(
                original_average_edge_length := igl.avg_edge_length(
                    nvdm_loaded_mesh.v_pos.cpu().detach().numpy(), loaded_faces_np
                )
            ),
            postinitremesh_avg_edge_length=original_average_edge_length,
            current_average_edge_length=original_average_edge_length,
            original_selection_sum_volume=original_selection_sum_volume,
            current_selection_sum_volume=original_selection_sum_volume,
            new2old_fi=np.arange(len(nvdm_loaded_mesh.t_pos_idx), dtype=int),
        )
        mesh_structs.append(mesh_struct)
        pt3d_verts_list.append(nvdm_loaded_mesh.v_pos)
        pt3d_faces_list.append(nvdm_loaded_mesh.t_pos_idx)

    pt3d_batched_meshes = pt3d_structures.Meshes(
        verts=pt3d_verts_list, faces=pt3d_faces_list
    ).to(device)
    return mesh_structs, pt3d_batched_meshes


def get_batch_of_cameras_and_lights_with_dist_adapt(
    cams_and_lights_cfg: td_camera.CamsAndLights_Settings,
    render_device: torch.device,
    view_batch_size: int,
    adapt_dists: bool,
    verts_to_adapt_dists_to: torch.Tensor,
    just_pick_evenly_spaced_azims: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    extends td_camera.get_batch_of_cameras_and_lights with adaptation to mesh extents maybe

    just_pick_evenly_spaced_azims: turns off randomness for azimuth sampling, just picks
    view_batch_size spaced-out azimuths in the specified azim_minmax
    """
    # adaptive distance scaling based on deformed mesh extents
    if adapt_dists:
        with torch.no_grad():
            v_pos = verts_to_adapt_dists_to
            vmin = v_pos.amin(dim=0)
            vmax = v_pos.amax(dim=0)
            v_pos = v_pos - (vmin + vmax) / 2
            adapt_dists_mult = (
                torch.cat([v_pos.amin(dim=0), v_pos.amax(dim=0)]).abs().amax().item()
            )
    else:
        adapt_dists_mult = 1.0

    # make batch of random camera parameters
    # (this uses numpy global rng, not the torch rng we made above. But the
    # seed_all does set a consistent numpy seed)
    cams_and_lights_batch = td_camera.get_batch_of_cameras_and_lights(
        cams_and_lights_cfg,
        view_batch_size,
        dist_multiplier=adapt_dists_mult,
        just_pick_evenly_spaced_azims=just_pick_evenly_spaced_azims,
    )
    for key in cams_and_lights_batch:
        cams_and_lights_batch[key] = cams_and_lights_batch[key].to(render_device)
    return cams_and_lights_batch


def render_nvdm_mesh_with_new_verts_and_view_batch(
    glctx: Union[dr.RasterizeGLContext, dr.RasterizeCudaContext],
    cams_and_lights_cfg: td_camera.CamsAndLights_Settings,
    cams_and_lights_batch: Dict[str, torch.Tensor],
    background: torch.Tensor,
    nvdm_loaded_mesh: nvdiffmodeling_mesh.Mesh,
    new_verts: torch.Tensor,
) -> torch.Tensor:
    """
    renders a single mesh given as `nvdm_loaded_mesh`, except using `new_verts`
    rather than its verts.

    cams_and_lights_batch is from td_camera.get_batch_of_cameras_and_lights
    OR this script's get_batch_of_cameras_and_lights_with_dist_adapt

    background should be shape (3,) rgb float in range [0,1]

    returns a batch of images of shape `(view_batch_size, channels=3, h, w)`
    where h, w are equal to cams_and_lights_cfg.raster_res
    """
    nvdm_render_mesh = nvdiffmodeling_mesh.Mesh(
        new_verts,  # override only verts,
        base=nvdm_loaded_mesh,  # get everything else from nvdm_loaded_mesh
    )
    # NOTE meshfusion/td code "combines" nvdm_render_mesh into a 1-mesh scene here

    nvdm_render_mesh = nvdiffmodeling_mesh.auto_normals(nvdm_render_mesh)
    nvdm_render_mesh = nvdiffmodeling_mesh.compute_tangents(nvdm_render_mesh)
    # these functions return a lazy chain of computations on the mesh
    # which we have to eval() in order to get back a concrete nvdm Mesh struct.
    # in our case, the eval will feed the chain of computations with the camera
    # parameters

    # eval the lazy chain of computations queued on nvdm_render_mesh
    # to get back a concrete nvdm Mesh struct
    nvdm_render_mesh = nvdm_render_mesh.eval(cams_and_lights_batch)
    train_render = nvdiffmodeling_render.render_mesh(
        glctx,
        nvdm_render_mesh,
        cams_and_lights_batch["mvp"],
        cams_and_lights_batch["campos"],
        cams_and_lights_batch["lightpos"],
        cams_and_lights_cfg.light_power,
        (raster_res := cams_and_lights_cfg.raster_res),
        spp=1,
        num_layers=1,
        msaa=False,
        background=torch.broadcast_to(background, (1, raster_res, raster_res, 3)),
    )
    # ^ (view_batch_size, h, w, channels)
    assert isinstance(train_render, torch.Tensor)
    train_render = train_render.permute(0, 3, 1, 2)
    # ^ (view_batch_size, channels, h, w)
    return train_render


def save_optimized_matrices_or_normals(
    deform_by_csd_cfg: DeformByCSD_Settings,
    dataset_cfg: MeshesDataset_Settings,
    patient_mesh: pt3d_structures.Meshes,
    per_elem_mats_or_normals: torch.Tensor,
    save_fname: str,
) -> str:
    """
    saves the source mesh and optimized deformation quantity for 1 mesh
    (i.e. len(patient_mesh) == 1)
    (can deal with patient_mesh of more than 1 batch size too, but will fuse
    all meshes together in the same packed arrays)
    """
    if os.path.isfile(save_fname):
        fname_noext, ext = os.path.splitext(save_fname)
        save_fname = next_increment_path(fname_noext + "@{:03}" + ext)
    quantity_save_dict = {}

    if (optimize_deform_via := deform_by_csd_cfg.optimize_deform_via) == "faces_normals":
        # due to double backward shenanigans, we actually optimize an offset to
        # add to faces_normals rather than faces_normals itself.
        quantity_save_dict["faces_normals_offset"] = (
            per_elem_mats_or_normals.cpu().detach().numpy()
        )
    else:
        quantity_save_dict[optimize_deform_via] = (
            per_elem_mats_or_normals.cpu().detach().numpy()
        )
    np.savez_compressed(
        save_fname,
        verts=patient_mesh.verts_packed().cpu().detach().numpy(),
        faces=patient_mesh.faces_packed().cpu().detach().numpy(),
        deform_by_csd_cfg=np.array(deform_by_csd_cfg.to_json_string()),
        dataset_cfg=np.array(dataset_cfg.to_json_string()),
        **quantity_save_dict,
    )
    return save_fname


def calc_verts_normals_and_scale__scale_id_loss(
    optim_quantity_this_batch: deformations.QuantityBeingOptimized,
    meshes_structs_this_batch: Sequence[BaseMeshGoingIntoOptimLoop],
    mode: Literal["per_coord", "det"],
    verts_normals_and_scale__scale_id_loss_weight: float,
    vselvol_heuristics_cfg: VolumeBasedHeuristic_Settings,
) -> Tuple[torch.Tensor, float]:
    assert optim_quantity_this_batch.this_is == "verts_normals_and_scale", (
        "cannot do verts_normals_and_scale__scale_id_loss with other optimize_deform_via"
    )
    scale = optim_quantity_this_batch.tensor[:, 3:]
    if mode == "det":
        # this scale is used as diag elements in a pure diag matrix, so the det is its prod
        det = scale.prod(dim=-1)
        # loss is det encouraged to be close to 1
        diff = det - 1
    elif mode == "per_coord":
        diff = scale - 1
    scale_id_loss = diff * diff
    with torch.no_grad():
        scale_id_loss_val = scale_id_loss.detach().mean().item()
    if heuristic_target_cfg := vselvol_heuristics_cfg.vns_scale_id_loss_weight:
        scale_id_loss_wtd = torch.cat(
            tuple(
                scale_id_loss_this_mesh
                * heuristic_target_cfg.calc(
                    verts_normals_and_scale__scale_id_loss_weight,
                    mesh_struct.current_selection_sum_volume,
                )
                for scale_id_loss_this_mesh, mesh_struct in zip(
                    torch.split(
                        scale_id_loss, optim_quantity_this_batch.num_verts_per_mesh, dim=0
                    ),
                    meshes_structs_this_batch,
                )
            ),
            dim=0,
        )
    else:
        scale_id_loss_wtd = scale_id_loss * verts_normals_and_scale__scale_id_loss_weight

    return scale_id_loss_wtd.mean(), scale_id_loss_val


def calc_jacobian_id_loss(
    optim_quantity_this_batch: deformations.QuantityBeingOptimized,
    intermediate_results: deformations.DeformationIntermediateResults,
    jac_id_loss_weight: float,
) -> Tuple[torch.Tensor, float]:
    """
    compute the jacobian identity regularization loss. returns the loss tensor
    and the loss float value.

    `optim_quantity_this_batch.this_is == "faces_jacobians" or "faces_3x2rotations" or "verts_3x2rotations"` required!
    new: "faces_normals" and "verts_normals" also allowed but need intermediate_results.rot_matrices
    """
    device = optim_quantity_this_batch.tensor.device
    if (
        optim_quantity_this_batch.this_is == "faces_normals"
        or optim_quantity_this_batch.this_is == "verts_normals"
        or optim_quantity_this_batch.this_is == "verts_normals_and_scale"
    ):
        assert isinstance(
            intermediate_results, deformations.ElemNormals_IntermediateResults
        )
        mat3x3 = intermediate_results.vert_matrices
    elif optim_quantity_this_batch.tensor.size(-1) == 2:
        mat3x3 = deformations.convert_rot3x2_to_rot3x3(optim_quantity_this_batch.tensor)
    else:
        mat3x3 = optim_quantity_this_batch.tensor
    jac_id_loss = (mat3x3 - torch.eye(3, device=device)).pow(2).mean()
    jac_id_loss_val = jac_id_loss.item()  # the val to print is before weighting
    return (jac_id_loss_weight * jac_id_loss), jac_id_loss_val


def calc_jacobian_det1_loss(
    optim_quantity_this_batch: deformations.QuantityBeingOptimized,
    intermediate_results: deformations.DeformationIntermediateResults,
    jac_det1_loss_weight: float,
) -> Tuple[torch.Tensor, float]:
    """
    compute the jacobian determinant regularization loss. returns the loss tensor
    and the loss float value.

    as of 2025-05-15 this is currently unused but we might need to use this for ablation/comparison
    `optim_quantity_this_batch.this_is == "faces_jacobians" or "faces_3x2rotations" or "verts_3x2rotations"` required!
    new: "faces_normals" and "verts_normals" also allowed but need intermediate_results.rot_matrices
    """
    if (
        optim_quantity_this_batch.this_is == "faces_normals"
        or optim_quantity_this_batch.this_is == "verts_normals"
        or optim_quantity_this_batch.this_is == "verts_normals_and_scale"
    ):
        assert isinstance(
            intermediate_results, deformations.ElemNormals_IntermediateResults
        )
        mat3x3 = intermediate_results.vert_matrices
    elif optim_quantity_this_batch.tensor.size(-1) == 2:
        mat3x3 = deformations.convert_rot3x2_to_rot3x3(optim_quantity_this_batch.tensor)
    else:
        mat3x3 = optim_quantity_this_batch.tensor
    # jac_id_loss = (mat3x3 - torch.eye(3, device=device)).pow(2).mean()
    jac_det1_loss = (torch.det(mat3x3) - 1).pow(2).mean()
    jac_det1_loss_val = jac_det1_loss.item()  # the val to print is before weighting
    return (jac_det1_loss_weight * jac_det1_loss), jac_det1_loss_val


class AdamlikeOptimizerAsPerElemFeatures:
    @staticmethod
    def to_features(
        deformqty_optimizer: torch.optim.Optimizer,
        deformqty: torch.Tensor,
    ):
        """
        deformqty is a per-element (e.g. vertex) quantity being optimized by the optimizer.
        The optimizer keeps some state for each element.

        Turn the optimizer state into flattened per-element features and
        returns a tensor storing the deformqty and all this per-elem state state, of shape
        (deformqty.size(0), n_optimizer_state_arrays * (deformqty.numel() // deformqty.size(0)))
        and an int for the 'last step' of the optimizer.

        for Adam, n_optimizer_state_arrays is 3.
        """
        deformqty_optimizer_state = deformqty_optimizer.state_dict()["state"].get(0)
        if deformqty_optimizer_state is not None:
            # add more optimizer types if needed
            if isinstance(deformqty_optimizer, (torch.optim.Adam,)):
                n_verts_before = deformqty.size(0)
                deformqty_and_optim_state_as_features = torch.cat(
                    (
                        deformqty.detach().view(n_verts_before, -1),
                        deformqty_optimizer_state["exp_avg"].view(n_verts_before, -1),
                        deformqty_optimizer_state["exp_avg_sq"].view(n_verts_before, -1),
                    ),
                    dim=-1,
                )
                deformqty_optimizer_step = deformqty_optimizer_state["step"]
                return deformqty_and_optim_state_as_features, deformqty_optimizer_step
            else:
                thlog.debug(
                    "optimizer not known, AdamlikeOptimizerAsPerElemFeatures will not extract optim state as per-elem features"
                )
                return None, None
        else:
            return None, None

    @staticmethod
    def restore_deformqty_and_optimizer_state(
        deformqty_optimizer: torch.optim.Optimizer,
        features_with_different_dim0_sz: torch.Tensor,
        step: torch.Tensor,
        deformqty_shape_without_dim0: torch.Size,
        restore_optimizer_state_inplace: bool,
        adjust_optimizer_state_inplace_fn: Optional[
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Any]
        ],
        DRMSHDEBUG_PREPOSTREMESHVIZ_nverts_and_psmeshes_to_register_qtys: Sequence[
            Tuple[int, Union[ps.SurfaceMesh, _PolyscopeRegisteredStructProxy]]
        ],
    ):
        """
        step is supposed to be torch.tensor(float(step_as_int))
            (shd be the one returned directly from to_features)
        deformqty_shape_without_dim0 should be deformqty.shape[1:]
        features_with_different_dim0_sz is some per-vertex features but interpolated
        onto the new verts due to remesh

        if not restore_optimizer_state_inplace, just returns the recovered deformqty
        without doing any modification to the optimizer
        """
        deformqty_optimizer_the_state_dict = deformqty_optimizer.state_dict()
        deformqty_numel = deformqty_shape_without_dim0.numel()

        deformqty = features_with_different_dim0_sz[:, :deformqty_numel].view(
            -1, *deformqty_shape_without_dim0
        )

        if restore_optimizer_state_inplace:
            if isinstance(deformqty_optimizer, (torch.optim.Adam,)):
                deformqty_optimizer_exp_avg = features_with_different_dim0_sz[
                    :, deformqty_numel : (2 * deformqty_numel)
                ].view(-1, *deformqty_shape_without_dim0)
                deformqty_optimizer_exp_avg_sq = features_with_different_dim0_sz[
                    :, (2 * deformqty_numel) : (3 * deformqty_numel) :
                ].view(-1, *deformqty_shape_without_dim0)

                if callable(adjust_optimizer_state_inplace_fn):
                    adjust_optimizer_state_inplace_fn(
                        deformqty,
                        deformqty_optimizer_exp_avg,
                        deformqty_optimizer_exp_avg_sq,
                    )

                deformqty_optimizer_the_state_dict["state"][0] = {
                    "step": step,
                    "exp_avg": deformqty_optimizer_exp_avg,
                    "exp_avg_sq": deformqty_optimizer_exp_avg_sq,
                }

                deformqty_optimizer.load_state_dict(deformqty_optimizer_the_state_dict)

                # for visualizing/illustrating
                if DRMSHDEBUG_PREPOSTREMESHVIZ:
                    curr_i = 0
                    for idx_in_batch, (n_verts_this_mesh, psmesh) in enumerate(
                        DRMSHDEBUG_PREPOSTREMESHVIZ_nverts_and_psmeshes_to_register_qtys
                    ):
                        expavgsq = (
                            deformqty_optimizer_exp_avg_sq[
                                curr_i : curr_i + n_verts_this_mesh
                            ]
                            .cpu()
                            .detach()
                            .numpy()
                        )
                        psmesh.add_scalar_quantity(
                            f"newsrclerped__vns_scl_expavgsq_mag{idx_in_batch}",
                            np.linalg.norm(expavgsq[:, 3:6], axis=-1),
                        )
                        psmesh.add_scalar_quantity(
                            f"newsrclerped__vns_dir_expavgsq_mag{idx_in_batch}",
                            np.linalg.norm(expavgsq[:, 0:3], axis=-1),
                        )
                        curr_i += n_verts_this_mesh
                    thlog.psr.show()
            else:
                thlog.debug(
                    "optimizer is not known, so AdamlikeOptimizerAsPerElemFeatures cannot read and restore state despite being told to restore optimizer state in place!"
                )
        return deformqty


def loophelp__init_solver(
    deform_by_csd_cfg: DeformByCSD_Settings,
    pt3d_batched_meshes: pt3d_structures.Meshes,
    vertex_selection_mask_list: Sequence[Optional[torch.Tensor]],
) -> Tuple[deformations.SparseLaplaciansSolvers, Sequence[ps.SurfaceMesh]]:
    psmeshes = []
    for i, (nonsel_verts, faces) in enumerate(
        zip(pt3d_batched_meshes.verts_list(), pt3d_batched_meshes.faces_list())
    ):
        psmeshes.append(
            thlog.psr.register_surface_mesh(
                f"src{i}", nonsel_verts.cpu().detach().numpy(), faces.cpu().detach().numpy()
            )
        )

    pin_special_vertex = deform_by_csd_cfg.pin_special_vertex
    solve_method = deform_by_csd_cfg.solve_method
    if solve_method == "njfpoisson" or solve_method == "arap":
        verts_pinmask_each_mesh: List[Optional[torch.Tensor]] = []
        for i, selmask in enumerate(vertex_selection_mask_list):
            if selmask is not None:
                # sample some furthest points among nonselected verts and pin them?
                if deform_by_csd_cfg.pin_nonselected.hardpin:
                    if (
                        n_fpsamples := deform_by_csd_cfg.pin_nonselected.fpsamps_if_hardpin
                    ) is not None:
                        nonselmask = ~selmask
                        nonsel_verts = pt3d_batched_meshes.verts_list()[i][nonselmask]
                        n_nonsel_verts = len(nonsel_verts)
                        nonsel2verts_idx = nonselmask.nonzero().view(-1)
                        pins, fpsamp_idxs_in_nonsel = pt3d_ops.sample_farthest_points(
                            nonsel_verts.unsqueeze(0),
                            torch.tensor([n_nonsel_verts], device=nonsel_verts.device),
                            K=n_fpsamples,
                        )
                        pinmask = torch.zeros_like(selmask)
                        pinmask[nonsel2verts_idx[fpsamp_idxs_in_nonsel.squeeze(0)]] = True
                        thlog.psr.register_point_cloud(
                            "fp pins", pins.squeeze(0).cpu().detach().numpy()
                        )
                    else:
                        # pin all nonselected
                        pinmask = ~selmask
                else:
                    pinmask = None
                verts_pinmask_each_mesh.append(pinmask)
            else:
                # no selmask
                verts_pinmask_each_mesh.append(None)

        thlog.psr.show()

        # this will handle adding the necessary pins to the pinmask
        return deformations.SparseLaplaciansSolvers.from_meshes(
            pt3d_batched_meshes,
            verts_pinmask_each_mesh=verts_pinmask_each_mesh,
            ensure_each_connected_component_has_at_least_1_pinned_vertex=pin_special_vertex,
            compute_njfpoisson_rhs_lefts=(deform_by_csd_cfg.solve_method == "njfpoisson"),
        ), psmeshes
    else:
        raise InvalidConfigError(f"unknown solve_method {solve_method}")


def loophelp__init_optim_qty(
    deform_by_csd_cfg: DeformByCSD_Settings,
    pt3d_batched_meshes: pt3d_structures.Meshes,
    my_solvers: deformations.SparseLaplaciansSolvers,
    local_step_procrustes_cfg: Optional[deformations.Procrustes_Settings],
) -> deformations.QuantityBeingOptimized:
    thlog.debug(f"init qty with local step procrstes init {local_step_procrustes_cfg}")
    return deformations.QuantityBeingOptimized.init_according_to_cfg(
        meshes=pt3d_batched_meshes,
        optimize_deform_via=deform_by_csd_cfg.optimize_deform_via,
        procrustes_inits_if_procrustes_needed=deformations.ProcrustesInitsForQuantityInit(
            procrustes_cfg=local_step_procrustes_cfg,
            solvers=my_solvers,
            arap_energy_type=deform_by_csd_cfg.arap_energy_type,
            vselvol_heuristics_fns=deformations.VolumeBasedHeuristicsFnsForApplyScaleAfterProcrustes(
                heuristic_target_cfg.use_orig_volume_instead_of_curr_volume,
                heuristic_target_cfg.calc,
            )
            if (
                heuristic_target_cfg
                := deform_by_csd_cfg.adjust_things_via_selection_volume.vns_scale_max
            )
            else None,
        )
        if local_step_procrustes_cfg
        else None,
    )


def loophelp__init_optimizer(
    deform_by_csd_cfg: DeformByCSD_Settings,
    quantity_being_optimized___: deformations.QuantityBeingOptimized,
    main_lr_init___: float,
) -> SupportedOptimizerType:
    optimizer = make_optimizer(
        quantity_being_optimized___,
        main_lr_init___,
        deform_by_csd_cfg.optimizer,
    )
    return optimizer


def loophelp__zero_out_nan_grads_inplace(
    quantity_being_optimized: deformations.QuantityBeingOptimized,
    deformation_intermediate_results: deformations.DeformationIntermediateResults,
):
    # this should rarely actually find nans in grad (i only ran into this with
    # perfectly triangulated cube examples for some reason)
    grad = quantity_being_optimized.tensor.grad
    if grad is not None:
        isnan = grad.isnan()
        # this whole block is for debug printing
        if thlog.logguard(LOG_DEBUG) and isnan.any():
            torch.set_printoptions(precision=4)
            isnanreduced = isnan
            while isnanreduced.ndim > 1:
                isnanreduced = isnanreduced.any(dim=-1)
            isnanwhere = isnanreduced.nonzero().flatten()
            thlog.debug(f"found nan in grad, zeroing out. isnan at\n{isnanwhere}")
            if (
                deformation_intermediate_results is not None
                and deformation_intermediate_results.procrustes_covar is not None
            ):
                thlog.trace(
                    f"procrustes covar matrices for those verts:\n{deformation_intermediate_results.procrustes_covar[isnanwhere]}"
                )

        # do the zeroing
        grad[isnan] = 0


def update_nvdm_mesh_inplace_with_new_verts_faces(
    nvdm_mesh: nvdiffmodeling_mesh.Mesh, verts: torch.Tensor, faces: torch.Tensor
) -> nvdiffmodeling_mesh.Mesh:
    """only remakes trivial UVs, leaves the material settings etc alone"""
    mega_trivial_v_tex = torch.full(
        (verts.size(0), 2), 0.5, dtype=verts.dtype, device=verts.device
    )
    mega_trivial_t_tex_idx = faces.clone()
    # nvdiffmodeling_mesh.Mesh()
    nvdm_mesh.v_pos = verts
    nvdm_mesh.t_pos_idx = faces
    nvdm_mesh.v_tex = mega_trivial_v_tex
    nvdm_mesh.t_tex_idx = mega_trivial_t_tex_idx
    return nvdm_mesh


# need this to replace all quantities on source meshes after periodic remeshing...
def loophelp__replace_source_meshes_structs(
    meshes_structs___: Sequence[BaseMeshGoingIntoOptimLoop],
    verts_list: List[torch.Tensor],
    faces_list: List[torch.Tensor],
    vertex_selection_mask_list: List[Optional[torch.Tensor]],
    we_are_right_after_init_remesh: bool,
):
    # first replace all the meshes_structs v_pos and t_pos_idx,
    # and maybe also the t_tex_idx? i dunno if that will be needed
    # in any case we can always make very trivial UVs
    for verts, faces, selmask, mesh_struct in zip(
        verts_list, faces_list, vertex_selection_mask_list, meshes_structs___
    ):
        mesh_struct.nvdm_loaded_mesh = update_nvdm_mesh_inplace_with_new_verts_faces(
            mesh_struct.nvdm_loaded_mesh, verts, faces
        )
        mesh_struct.vertex_selection_mask = selmask

        # also update its current selection volume
        verts_np = verts.cpu().detach().numpy()
        faces_np = faces.cpu().detach().numpy()
        try:
            mesh_struct.current_selection_sum_volume = (
                vselection_volume.calc_holeclosed_vselection_volume(
                    verts_np,
                    faces_np,
                    (
                        selmask.cpu().detach().numpy()
                        if selmask is not None
                        else np.ones(len(verts), dtype=bool)
                    ),
                )
            )
        except vselection_volume.HolecloserGaveUp:
            thlog.info(
                "hole closer for volume calc gave up, setting mesh_struct.current_selection_sum_volume to None"
            )
            mesh_struct.current_selection_sum_volume = None

        # update avg edge length
        mesh_struct.current_average_edge_length = igl.avg_edge_length(verts_np, faces_np)
        if we_are_right_after_init_remesh:
            mesh_struct.postinitremesh_avg_edge_length = (
                mesh_struct.current_average_edge_length
            )

    # also return updated pytorch3d meshes
    return pt3d_structures.Meshes(verts=verts_list, faces=faces_list)


### func for initial inflation
@torch.no_grad()
def loophelp__initial_inflate(
    initial_inflate_cfg: Inflation_Settings,
    vselvol_heuristics_cfg: VolumeBasedHeuristic_Settings,
    pt3d_batched_meshes___: pt3d_structures.Meshes,  # new value returned
    original_vertex_selection_mask_packed: torch.Tensor,
    meshes_structs___INPLACE: Sequence[BaseMeshGoingIntoOptimLoop],  # modified in place
) -> pt3d_structures.Meshes:
    assert (inflate_multiplier := initial_inflate_cfg.inflate_along_normals) is not None, (
        "other inflation modes not implemented"
    )
    inflate_offsets = []

    # smooth out the vsel itself to avoid border vertices (to not drag along
    # tris at the border of the selected region)
    vself_packed = original_vertex_selection_mask_packed.float().unsqueeze(-1)
    for _ in range(initial_inflate_cfg.vsel_smooth_iters):
        vself_packed += (
            initial_inflate_cfg.vsel_smooth_lambda
            * pt3d_batched_meshes___.laplacian_packed().mm(vself_packed)
        )
    vsel_smoothed_packed = vself_packed.squeeze(-1) > 0.99
    vsel_smoothed_list = torch.split(
        vsel_smoothed_packed, pt3d_batched_meshes___.num_verts_per_mesh().tolist()
    )

    for verts_normals, vsel_smoothed, mesh_struct in zip(
        pt3d_batched_meshes___.verts_normals_list(),
        vsel_smoothed_list,
        meshes_structs___INPLACE,
    ):
        verts_normals[~vsel_smoothed, :] = 0

        if (
            orig_vsel_sum_volume := mesh_struct.original_selection_sum_volume
        ) is not None and (
            heuristic_target_cfg := vselvol_heuristics_cfg.initial_normal_inflate
        ):
            inflate_multiplier = heuristic_target_cfg.calc(
                inflate_multiplier, orig_vsel_sum_volume
            )
            thlog.info(
                f"vselection_volume_heuristic gives inflate_multiplier {inflate_multiplier}"
            )
        inflate_offset = verts_normals * inflate_multiplier
        inflate_offsets.append(inflate_offset)

    inflate_offsets_packed = torch.cat(inflate_offsets, dim=0)
    # smooth out the offsets a bit if lambda is specified
    if (
        l := initial_inflate_cfg.inflate_offset_smooth_lambda
    ) and initial_inflate_cfg.inflate_offset_smooth_iters:
        for _ in range(initial_inflate_cfg.inflate_offset_smooth_iters):
            inflate_offsets_packed += l * pt3d_batched_meshes___.laplacian_packed().mm(
                inflate_offsets_packed
            )
    # apply offsets to pt3d packed meshes, and then meshes_structs
    pt3d_batched_meshes_inflated = pt3d_batched_meshes___.offset_verts(
        inflate_offsets_packed
    )
    for mesh_struct, inflated_verts in zip(
        meshes_structs___INPLACE, pt3d_batched_meshes_inflated.verts_list()
    ):
        mesh_struct.nvdm_loaded_mesh.v_pos = inflated_verts

    return pt3d_batched_meshes_inflated


def project_points_to_mesh_with_igl(
    query_verts_packed: torch.Tensor,
    query_num_verts_per_mesh: List[int],
    meshes: pt3d_structures.Meshes,
) -> torch.Tensor:
    """
    returns face indices corresp
    """
    surface_pts_for_query_verts__list = []
    query_verts_list = torch.split(query_verts_packed, query_num_verts_per_mesh, dim=0)
    for query_verts, verts, faces in zip(
        query_verts_list,
        meshes.verts_list(),
        meshes.faces_list(),
    ):
        _, f_inds_for_each_query_pt, surface_pt_for_each_query_pt = (
            igl.point_mesh_squared_distance(
                query_verts.cpu().detach().numpy(),
                verts.cpu().detach().numpy(),
                faces.cpu().detach().numpy(),
            )
        )
        surface_pts_for_query_verts__list.append(surface_pt_for_each_query_pt)
    surface_pts_for_query_verts__packed = torch.from_numpy(
        np.concatenate(surface_pts_for_query_verts__list, axis=0)
    ).to(query_verts_packed)
    return surface_pts_for_query_verts__packed


#### func to handle remeshing
def loophelp__remesh_and_resume(
    *,
    device: torch.device,
    deform_by_csd_cfg: DeformByCSD_Settings,
    meshes_structs___INPLACE: Sequence[BaseMeshGoingIntoOptimLoop],  # modified in place
    n_meshes: int,
    new_verts_list_and_intermediate_results__alldataset: Sequence[
        Tuple[torch.Tensor, deformations.DeformationIntermediateResults]
    ],
    start_optimizer_at_lr: float,
    remesh_i: int,
    remesh_cfg: Remeshing_Settings,
    optimizer___: Optional[SupportedOptimizerType],  # not modified, new value returned
    quantity_being_optimized___: Optional[
        deformations.QuantityBeingOptimized
    ],  # not modified, new value returuned
    # ^ this can be None for the initial remesh; for resumption, this shouldn't be none
    pt3d_batched_meshes___: pt3d_structures.Meshes,  # not modified, new value returned
    pt3d_batched_meshes_original: pt3d_structures.Meshes,  # (before any deforms)
    original_geometry_xyz_for_current_triangles___: Optional[
        torch.Tensor
    ],  # new value returned
):
    remesh_and_resume_start_time = (
        time.perf_counter() if thlog.logguard(LOG_DEBUG) else None
    )
    resumption_will_be_involved = (
        remesh_cfg.use_interpd_optimizer_state
        and not remesh_cfg.override__remesh_only_once_at_start
    )  # once at start, no point in interpolating

    if not resumption_will_be_involved:
        extra_vertex_attributes_packed, step = None, None
        piggyback_vertex_attributes_packed = None
    else:
        ########### some printouts
        if deform_by_csd_cfg.optimize_deform_via.startswith("faces"):
            raise NotImplementedError(
                "can't interpolate a per-face deform quantity, not implemented"
            )

        ########## build up vertex attributes to be interpolated in the remesh
        # if either of these are None, that means this is an initial remesh, not a resumption
        extra_vertex_attributes_packed, step = (
            (
                AdamlikeOptimizerAsPerElemFeatures.to_features(
                    optimizer___,
                    quantity_being_optimized___.tensor,
                )
            )
            if quantity_being_optimized___ is not None and optimizer___ is not None
            else (None, None)
        )

        # now pack in "original geometry xyz for current triangles" and other attributes
        piggyback_vertex_attributes_packed = (
            grab_piggyback_vertex_attributes_packed_from_meshes_structs(
                meshes_structs___INPLACE
            )
        )
        extra_vertex_attributes_packed = torch.cat(
            (
                (extra_vertex_attributes_packed,)
                if extra_vertex_attributes_packed is not None
                else ()
            )
            + (
                (piggyback_vertex_attributes_packed,)
                if piggyback_vertex_attributes_packed is not None
                else ()
            )
            + (
                (original_geometry_xyz_for_current_triangles___,)
                if original_geometry_xyz_for_current_triangles___ is not None
                else ()
            ),
            dim=-1,
        )
        if extra_vertex_attributes_packed is not None:
            thlog.debug(
                f"extra vertex attributes num features: {extra_vertex_attributes_packed.size(-1)}"
            )
            if original_geometry_xyz_for_current_triangles___ is not None:
                thlog.debug(
                    f"orig geom xyz for current triangles shape {original_geometry_xyz_for_current_triangles___.shape}"
                )
            if piggyback_vertex_attributes_packed is not None:
                thlog.debug(
                    f"other vert attribs shape {piggyback_vertex_attributes_packed.shape}"
                )
    ########### end "if not resumption_will_be_involved"

    # these quantities are per-vertex packed (of shape (sum of
    # all n_verts in dataset,*)), so we have to split into the
    # pt3d-style _list() of one array for each mesh
    extra_vertex_attributes_list = (
        torch.split(
            extra_vertex_attributes_packed,
            pt3d_batched_meshes___.num_verts_per_mesh().tolist(),
        )
        if extra_vertex_attributes_packed is not None
        else tuple(None for _ in range(n_meshes))
    )

    vertex_selection_mask_list = tuple(
        mesh_struct.vertex_selection_mask for mesh_struct in meshes_structs___INPLACE
    )

    remeshed_verts_list = []
    remeshed_faces_list = []
    remeshed_extra_vertex_attributes_list: List[Union[torch.Tensor, None]] = []
    remeshed_vertex_selection_mask_list: List[Union[torch.Tensor, None]] = []

    method_run_cfg, local_step_procrustes_cfg = remesh_cfg.get_run_settings_for_ith_remesh(
        remesh_i,
        deform_by_csd_cfg.local_step_procrustes,
    )

    for mesh_idx_in_dataset, (
        mesh_struct,
        from_verts,
        target_verts_and_intermediate_results,
        from_faces,
        maybe_extra_vertex_attr,
        maybe_vertex_selection_mask,
    ) in enumerate(
        zip(
            meshes_structs___INPLACE,
            pt3d_batched_meshes___.verts_list(),
            new_verts_list_and_intermediate_results__alldataset,
            pt3d_batched_meshes___.faces_list(),
            extra_vertex_attributes_list,
            vertex_selection_mask_list,
        )
    ):
        target_verts, _ = cast(
            Tuple[torch.Tensor, deformations.DeformationIntermediateResults],
            target_verts_and_intermediate_results,
        )
        if isinstance(method_run_cfg, BKRemeshLerps_RunSettings):
            if method_run_cfg.use_avglen == "original":
                avglen = mesh_struct.original_average_edge_length
            elif method_run_cfg.use_avglen == "post_init_remesh":
                avglen = mesh_struct.postinitremesh_avg_edge_length
            else:
                avglen = mesh_struct.current_average_edge_length
            targetlen = method_run_cfg.make_targetlen_from_avglen(avglen)
            bkremeshlerps_start_time = (
                time.perf_counter() if thlog.logguard(LOG_DEBUG) else None
            )
            (
                remeshed_verts,
                remeshed_faces,
                new2old_fi,
                maybe_remeshed_vertex_selection_mask,
                maybe_remeshed_extra_vertex_attr,
            ) = bkremeshlerps_remesh_with_attrs.do_isoremesh_with_vertex_selection_and_attributes(
                targetlen=targetlen,
                n_iters=method_run_cfg.n_iters,
                v=target_verts.detach(),
                f=from_faces.detach(),
                v_selection=maybe_vertex_selection_mask,
                v_attributes=maybe_extra_vertex_attr,
                do_smooth_step=method_run_cfg.method_cfg.do_smooth_step,
                interp_using_barycoords=method_run_cfg.method_cfg.interp_using_barycoords,
                override__adaptive_epsilon=method_run_cfg.override__adaptive_epsilon,
            )
            bkremeshlerps_end_time = (
                time.perf_counter() if bkremeshlerps_start_time is not None else 0.0
            )
            if bkremeshlerps_start_time is not None:
                thlog.info(
                    f"bkremeshlerps took {bkremeshlerps_end_time - bkremeshlerps_start_time:.6f} seconds"
                )
            # the returned new2old_fi maps remeshed faces to pre-remesh faces,
            # and then mesh_struct.new2old_fi maps pre-remesh faces to original faces
            # so to update the mapping to be (remeshed -> original), we compose them
            mesh_struct.new2old_fi = mesh_struct.new2old_fi[new2old_fi]
            # if the remesh->preremesh mapping gives -1, there is no good preremesh face
            # to match with (freshly spawned face, or mangled face without a good corresponding parent face)
            # so its mapping back to a face on the original shape should also say -1
            mesh_struct.new2old_fi[new2old_fi == -1] = -1
            thlog.debug(
                f"mesh_struct.new2oldfi shape after remesh {mesh_struct.new2old_fi.shape}, remeshed faces {remeshed_faces.shape}, new2old_fi shape {new2old_fi.shape}"
            )
            if DRMSHDEBUG_PREPOSTREMESHVIZ:
                preremesh = thlog.psr.register_surface_mesh(
                    "preremesh",
                    target_verts.cpu().detach().numpy(),
                    from_faces.cpu().detach().numpy(),
                )
                if isinstance(targetlen, torch.Tensor):
                    preremesh.add_scalar_quantity(
                        "targetlen", targetlen.cpu().detach().numpy()
                    )
                thlog.psr.register_surface_mesh(
                    "postremesh",
                    remeshed_verts.cpu().detach().numpy(),
                    remeshed_faces.cpu().detach().numpy(),
                )
                thlog.psr.show()
        else:
            raise InvalidConfigError("unknown method_cfg type")

        remeshed_verts_list.append(remeshed_verts)
        remeshed_faces_list.append(remeshed_faces)
        remeshed_extra_vertex_attributes_list.append(maybe_remeshed_extra_vertex_attr)
        remeshed_vertex_selection_mask_list.append(maybe_remeshed_vertex_selection_mask)
    # remesh done
    # now we reinitialize

    # this will change inplace the nvdm meshes inside meshes_structs,
    # (and also replace mesh_struct.vertex_selection_mask for each mesh_struct)
    # and return a new pt3d_batched_meshes with the new remeshed meshes
    pt3d_batched_meshes = loophelp__replace_source_meshes_structs(
        meshes_structs___INPLACE,
        remeshed_verts_list,
        remeshed_faces_list,
        remeshed_vertex_selection_mask_list,
        we_are_right_after_init_remesh=(step is None),
    )
    # grab this post-remesh num_verts_per_mesh while we're here
    nvertspermesh__after_remesh = pt3d_batched_meshes.num_verts_per_mesh().tolist()

    # remake the solver, optim quantity, and optimizer. (this function also
    # registers some psmeshes corresponding to the new source meshes (=the
    # remesh result) and returns them for later registering/viewing)
    my_solvers, new_src_psmeshes = loophelp__init_solver(
        deform_by_csd_cfg, pt3d_batched_meshes, remeshed_vertex_selection_mask_list
    )
    quantity_being_optimized = loophelp__init_optim_qty(
        deform_by_csd_cfg,
        pt3d_batched_meshes,
        my_solvers,
        local_step_procrustes_cfg,
    )
    optimizer = loophelp__init_optimizer(
        deform_by_csd_cfg, quantity_being_optimized, start_optimizer_at_lr
    )

    # if configured to do so (i.e. set extra vertex attributes that were
    # interpd during remesh) restore the interpolated deformqty and
    # deformqty optimizer state, stored in the interpolated "extra vertex
    # attributes" interpd thru the remesh
    original_geometry_xyz_for_remeshed_triangles = None
    if all(map(lambda x: x is not None, remeshed_extra_vertex_attributes_list)):
        remeshed_extra_vertex_attributes_packed = torch.cat(
            cast(List[torch.Tensor], remeshed_extra_vertex_attributes_list),
            dim=0,
        )  # ^ we can cast this because we already checked all is not None (above)

        # extract and pop the extra features carried along for the remesh and interp

        if original_geometry_xyz_for_current_triangles___ is not None:
            original_geometry_xyz_for_remeshed_triangles = project_points_to_mesh_with_igl(
                remeshed_extra_vertex_attributes_packed[:, -3:],
                pt3d_batched_meshes.num_verts_per_mesh().tolist(),
                # ^ these are the new remeshed meshes, we get the num_verts_per_mesh
                pt3d_batched_meshes_original,
            )
            remeshed_extra_vertex_attributes_packed = (
                remeshed_extra_vertex_attributes_packed[:, :-3]
            )
            # ^ cut off the last three which we used to store original-geometry xyz, which will
            # have been extracted out already

        if piggyback_vertex_attributes_packed is not None:
            # these are interpolated during the remesh, not via projection and barycentric
            # coords! if we'd like to do that, use project_points_to_mesh_with_igl here too
            piggyback_n = piggyback_vertex_attributes_packed.size(1)
            remeshed_piggyback_vertex_attributes_packed = (
                remeshed_extra_vertex_attributes_packed[:, -piggyback_n:]
            )
            remeshed_extra_vertex_attributes_packed = (
                remeshed_extra_vertex_attributes_packed[:, :-piggyback_n]
            )
            for mesh_struct, remeshed_piggyback_vertex_attr_this_mesh in zip(
                meshes_structs___INPLACE,
                torch.split(
                    remeshed_piggyback_vertex_attributes_packed, nvertspermesh__after_remesh
                ),
            ):
                mesh_struct.piggyback_vertex_attributes = (
                    remeshed_piggyback_vertex_attr_this_mesh
                )

        if step is not None and remeshed_extra_vertex_attributes_packed.numel() > 0:
            # put the interpolated optimizer state back into the optimizer
            # with adjustment?
            def __adjust_exp_avg_and_exp_avg_sq_inplace(
                interpd_deformqty: torch.Tensor,
                exp_avg: torch.Tensor,
                exp_avg_sq: torch.Tensor,
            ):
                # temporarily disable grad and put the new deformqty in,
                deformqty_before_mangle = quantity_being_optimized.tensor.detach().clone()
                quantity_being_optimized.tensor.detach_().copy_(interpd_deformqty)

                # this runs the local step to give remeshed rotation and scale for adjusting optimizer state
                remeshed_inputs_to_deformation_solve_methods = deformations.calc_inputs_to_solve_for_deformation_according_to_cfg(
                    pt3d_batched_meshes=pt3d_batched_meshes,
                    quantity_being_optimized=quantity_being_optimized,
                    current_vertex_selection_mask_packed=grab_vselection_mask_packed_from_meshes_structs(
                        meshes_structs___INPLACE
                    ),
                    current_and_original_selection_sum_volumes_for_vselvol_heuristics=grab_current_and_original_selection_sum_volumes_for_vselvol_heuristics_from_meshes_structs(
                        meshes_structs___INPLACE
                    ),
                )

                # put back original (since we're not using the interpolated deform quantity)
                # and requires_grad_ again, since that was lost when we wrote inplace above
                quantity_being_optimized.tensor.copy_(deformqty_before_mangle)
                quantity_being_optimized.tensor.requires_grad_()

                # modify exp_avg and exp_avg_sq with the rotation and scale
                intermediate_results = (
                    remeshed_inputs_to_deformation_solve_methods.intermediate_results
                )
                if (
                    intermediate_results is not None
                    and intermediate_results.rotscale_matrices is not None
                ):
                    rot_matrices_packed, scale_matrices_packed = (
                        intermediate_results.rotscale_matrices
                    )
                    expavg_dir = exp_avg[:, :3]
                    expavgsq_dir = exp_avg_sq[:, :3]
                    exp_avg[:, :3] = rot_matrices_packed.bmm(
                        expavg_dir.unsqueeze(-1)
                    ).squeeze(-1)
                    exp_avg_sq[:, :3] = (
                        (rot_matrices_packed**2).bmm(expavgsq_dir.unsqueeze(-1)).squeeze(-1)
                    )
                    if quantity_being_optimized.this_is == "verts_normals_and_scale":
                        expavg_scl = exp_avg[:, 3:]
                        expavgsq_scl = exp_avg_sq[:, 3:]
                        exp_avg[:, 3:] = scale_matrices_packed.bmm(
                            expavg_scl.unsqueeze(-1)
                        ).squeeze(-1)
                        exp_avg_sq[:, 3:] = (
                            (scale_matrices_packed**2)
                            .bmm(expavgsq_scl.unsqueeze(-1))
                            .squeeze(-1)
                        )
                else:
                    raise NotImplementedError(
                        "no optimizer adjust implemented for this optimize_deform_via: no rotscale_matrices intermediate results"
                    )

            # this return is unused because we're not installing the deformqty back into
            # the QuantityBeingOptimized, that is not needed
            _interpd_deformqty = AdamlikeOptimizerAsPerElemFeatures.restore_deformqty_and_optimizer_state(
                optimizer,
                features_with_different_dim0_sz=remeshed_extra_vertex_attributes_packed,
                step=step,
                deformqty_shape_without_dim0=quantity_being_optimized.tensor.shape[1:],
                restore_optimizer_state_inplace=remesh_cfg.use_interpd_optimizer_state,
                adjust_optimizer_state_inplace_fn=__adjust_exp_avg_and_exp_avg_sq_inplace
                if remesh_cfg.adjust_optimizer_state_using_interpd_deformqty
                else None,
                DRMSHDEBUG_PREPOSTREMESHVIZ_nverts_and_psmeshes_to_register_qtys=tuple(
                    zip(nvertspermesh__after_remesh, new_src_psmeshes)
                ),
            )
        # endif step is None
    # end interpolations

    # informative printout of the heuristic-applied targets

    for mesh_struct in meshes_structs___INPLACE:
        if mesh_struct.current_selection_sum_volume:
            if (
                heuristic_target_cfg
                := deform_by_csd_cfg.adjust_things_via_selection_volume.vns_scale_id_loss_weight
            ):
                thlog.info(
                    "vselvol_heuristics calc for vns_scale_id_loss_weight gives "
                    f"{heuristic_target_cfg.calc(1 if heuristic_target_cfg.formula.endswith('_scale') else 0, mesh_struct.current_selection_sum_volume if not heuristic_target_cfg.use_orig_volume_instead_of_curr_volume else mesh_struct.original_selection_sum_volume)}"
                )
            if (
                heuristic_target_cfg
                := deform_by_csd_cfg.adjust_things_via_selection_volume.vns_scale_max
            ):
                thlog.info(
                    "vselvol_heuristics calc for vns_scale_max gives "
                    f"{heuristic_target_cfg.calc(1 if heuristic_target_cfg.formula.endswith('_scale') else 0, mesh_struct.current_selection_sum_volume if not heuristic_target_cfg.use_orig_volume_instead_of_curr_volume else mesh_struct.original_selection_sum_volume)}"
                )

    remesh_and_resume_end_time = (
        time.perf_counter() if remesh_and_resume_start_time is not None else 0.0
    )
    if remesh_and_resume_start_time is not None:
        thlog.info(
            f"remesh_and_resume took {remesh_and_resume_end_time - remesh_and_resume_start_time:.6f} seconds (which includes the bkremeshlerps time reported above, if present)"
        )

    # these are new, for the newly remeshed mesh
    return (
        pt3d_batched_meshes,
        my_solvers,
        quantity_being_optimized,
        optimizer,
        original_geometry_xyz_for_remeshed_triangles,
    )


def loophelp__save_deformremesh_result(
    deform_by_csd_cfg: DeformByCSD_Settings,
    dataset_cfg: MeshesDataset_Settings,
    pt3d_batched_meshes: pt3d_structures.Meshes,
    pt3d_batched_meshes__original: pt3d_structures.Meshes,
    original_geometry_xyz_for_current_triangles: Optional[torch.Tensor],
    original_vertex_selection_mask_packed: Optional[torch.Tensor],
    current_vertex_selection_mask_packed: Optional[torch.Tensor],
    original_piggyback_vertex_attributes_packed: Optional[torch.Tensor],
    current_piggyback_vertex_attributes_packed: Optional[torch.Tensor],
    new2old_fi_packed: np.ndarray,
    n_meshes: int,
    remesh_str_in_fname: Union[int, str],
    optim_i: int,
    _drmshdebug_save_yzrot: bool = False,
):
    if n_meshes == 1:
        save_fname_noext, ext = os.path.splitext(
            deform_by_csd_cfg.optimized_quantity_save_fname
        )
        save_fname = save_fname_noext + f"-{remesh_str_in_fname}-optm{optim_i}" + ext

        np.savez_compressed(
            save_fname,
            deform_by_csd_cfg=np.array(deform_by_csd_cfg.to_json_string()),
            dataset_cfg=np.array(dataset_cfg.to_json_string()),
            deformed_verts=(
                _v := pt3d_batched_meshes.verts_packed().cpu().detach().numpy()
            ),
            deformed_faces=(
                _f := pt3d_batched_meshes.faces_packed().cpu().detach().numpy()
            ),
            original_verts=pt3d_batched_meshes__original.verts_packed()
            .cpu()
            .detach()
            .numpy(),
            original_faces=pt3d_batched_meshes__original.faces_packed()
            .cpu()
            .detach()
            .numpy(),
            new2old_fi=new2old_fi_packed,
            **(
                _vseldict := (
                    {
                        "original_vsel": original_vertex_selection_mask_packed.cpu()
                        .detach()
                        .numpy(),
                        "deformed_vsel": current_vertex_selection_mask_packed.cpu()
                        .detach()
                        .numpy(),
                    }
                    if original_vertex_selection_mask_packed is not None
                    and current_vertex_selection_mask_packed is not None
                    else {}
                )
            ),
            **(
                {
                    "original_geometry_xyz_for_current_triangles": original_geometry_xyz_for_current_triangles.cpu()
                    .detach()
                    .numpy()
                }
                if original_geometry_xyz_for_current_triangles is not None
                else {}
            ),
            **(
                {
                    "original_piggyback_vattrs": original_piggyback_vertex_attributes_packed.cpu()
                    .detach()
                    .numpy(),
                    "deformed_piggyback_vattrs": current_piggyback_vertex_attributes_packed.cpu()
                    .detach()
                    .numpy(),
                }
                if original_piggyback_vertex_attributes_packed is not None
                and current_piggyback_vertex_attributes_packed is not None
                else {}
            ),
        )
        if _drmshdebug_save_yzrot:
            # for comparisons with other methods, to start them off with the same inflated+remeshed shape
            # (many 3D generative methods out there use the z-up convention)
            mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            vrot_np = (mat @ _v.T).T
            yzrot_savename_noext = (
                save_fname_noext + f"rmsh{remesh_str_in_fname}optm{optim_i}-magicclayrot"
            )
            igl.write_triangle_mesh(
                yzrot_savename_noext + "--initinflrmshwithYZrot.obj", vrot_np, _f
            )
            if _vseldict:
                np.savetxt(
                    yzrot_savename_noext + "--allowedvertices.txt",
                    _vseldict["deformed_vsel"].nonzero()[0],
                    fmt="%d",
                )


class CSDGuidanceMethod:
    """
    the only reason this is a class and not just written inline directly in the
    optim loop is because we might want to sub this out for another guidance
    method by subclassing this and overriding init, render_single_mesh, and calc_loss.

    The methods do not modify any state of this class's instance, this is just a
    wrapper for some functions you can override when subclassing

    when subclassing to make a guidance method that doesn't involve
    prompts/prompt embeddings, in your __init__, just
    - set stage_I to csd.DummyCSDClass() (this is needed bc the loop init code
    will still try to run the function stage_I.encode_prompt, and DummyCSDClass
    provides a dummy trivial encode_prompt function)
    - set stage_II to None
    - ignore the prompt_zs and prompt_neg_zs arguments in your calc_loss implementation

    if you don't need to use nvdiffrast for rendering, glctx can also be None
    """

    def __init__(self, cfg: CSD_Settings, device: torch.device, rng: torch.Generator):
        self.cfg = cfg

        #### prep CSD modules
        if get_bool_env_variable("CSD_DUMMY"):
            stage_I = csd.DummyCSDClass()
        else:
            stage_I = csd.CSD(cfg.models, stage=1, generator=rng)
        # stage 2 is loaded only if we're not doing a dummy run AND if the stage 2
        # weight schedule is not a constant zero
        if (not isinstance(stage_I, csd.DummyCSDClass)) and (
            not ((__csdsched := cfg.stage_II_weight_schedule) == "0.0" or __csdsched == "0")
        ):
            stage_II = csd.CSD(cfg.models, stage=2, generator=rng)
        else:
            stage_II = None

        #### rasterizer context
        glctx = dr.RasterizeCudaContext()

        #### background
        background = torch.tensor(cfg.background_color).to(device)

        resize_for_guidance_fn: Callable[[torch.Tensor], torch.Tensor]
        if (resize_for_guidance := cfg.resize_for_guidance) is not None:
            resize_h, resize_w, resize_interp_method_name = resize_for_guidance
            resize_interp_fn = resize_right.get_interp_method(resize_interp_method_name)
            resize_for_guidance_fn = lambda img: resize_right.resize(
                img, out_shape=(resize_h, resize_w), interp_method=resize_interp_fn
            )
        else:
            resize_for_guidance_fn = lambda img: img

        #### stage 2 loss weight schedule func
        stage_II_loss_weight_fn = parse_lr_schedule_string_into_lr_lambda(
            cfg.stage_II_weight_schedule
        )

        self.glctx = glctx
        self.stage_I: Union[csd.CSD, csd.DummyCSDClass] = stage_I
        self.stage_II: Optional[csd.CSD] = stage_II
        self.stage_II_loss_weight_fn: Callable[[int], float] = stage_II_loss_weight_fn
        self.background: torch.Tensor = background
        self.resize_for_guidance_fn: Callable[[torch.Tensor], torch.Tensor] = (
            resize_for_guidance_fn
        )

    def render_single_mesh(
        self,
        optim_i: int,
        nvdm_loaded_mesh: nvdiffmodeling_mesh.Mesh,
        new_verts: torch.Tensor,
    ) -> torch.Tensor:
        # render this mesh with the new verts and a batch of views
        cams_and_lights_batch = get_batch_of_cameras_and_lights_with_dist_adapt(
            self.cfg.cams_and_lights,
            self.background.device,
            self.cfg.view_batch_size,
            (self.cfg.adapt_dists and optim_i > 1),
            new_verts,
        )
        train_render = render_nvdm_mesh_with_new_verts_and_view_batch(
            self.glctx,
            self.cfg.cams_and_lights,
            cams_and_lights_batch,
            self.background,
            nvdm_loaded_mesh,
            new_verts,
        )
        # train_render has shape (view_batch_size, channels, h, w)
        # resize h,w to the config-specified resize_for_guidance size
        train_render = self.resize_for_guidance_fn(train_render)
        return train_render

    def calc_loss(
        self,
        optim_i: int,
        train_renders: torch.Tensor,
        prompt_zs: torch.Tensor,
        prompt_neg_zs: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, float, float]:
        """
        train_renders (B, C=3, H, W)
        prompt_zs should have shape (B, 77, 4096) (the feature shape is from the prompt encoding function,
        and the batch size should be the same as train_renders, one prompt_z for each image in the batch)
        likewise for prompt_neg_zs
        """
        if isinstance(self.stage_I, csd.DummyCSDClass):
            raise RuntimeError("dummy csd can only go this far!")
        stage_II_loss_weight = self.stage_II_loss_weight_fn(optim_i)
        visual_loss, stage_I_loss_val, stage_II_loss_val, total_visual_loss_val = (
            csd.calc_csd_loss(
                self.stage_I,
                self.stage_II,
                train_renders,
                prompt_zs,
                prompt_neg_zs,
                self.cfg.stage_I_weight,
                stage_II_loss_weight,
            )
        )
        return visual_loss, stage_I_loss_val, stage_II_loss_val, total_visual_loss_val


def submain_deformremesh(
    deform_by_csd_cfg: DeformByCSD_Settings,
    dataset_cfg: MeshesDataset_Settings,
    device: torch.device,
    guidance_method_init: Callable[[torch.Generator], CSDGuidanceMethod],
):
    #### set seed
    rng, _, _ = deformations.seed_all(
        deform_by_csd_cfg.torch_seed, deform_by_csd_cfg.numpy_seed
    )

    #### init visual guidance method
    guidance = guidance_method_init(rng)

    remesh_cfg = deform_by_csd_cfg.periodic_remeshing

    #### load meshes and structs and nvdiffmodeling structs
    meshes_structs, pt3d_batched_meshes = load_meshes_from_dataset_cfg_and_encode_prompts(
        dataset_cfg,
        guidance.stage_I.encode_prompt,
        device,
    )
    pt3d_batched_meshes__original = pt3d_batched_meshes.clone()
    original_geometry_xyz_for_current_triangles = (
        pt3d_batched_meshes__original.verts_packed()
    )
    # ^ keep this, to save at the very end (since remeshing will mangle pt3d_batched_meshes
    # to use as the latest 'source mesh for deformation'; we need to keep a separate copy of
    # the actual loaded original source mesh for saving at the very end)
    n_meshes = len(meshes_structs)

    # this tensor is only used for initial inflation and for saving in the drmsh file (all
    # other uses of the selection mask are via the mask stored in each mesh_struct which
    # is updated after each remesh)
    original_vertex_selection_mask_packed = grab_vselection_mask_packed_from_meshes_structs(
        meshes_structs
    )
    # this is also for saving in the drmsh file
    original_piggyback_vertex_attributes_packed = (
        grab_piggyback_vertex_attributes_packed_from_meshes_structs(meshes_structs)
    )

    # pt3d_batched_meshes contains the same exact meshes (verts, faces) as the
    # ones in meshes_structs but allows convenient access to this whole
    # dataset's packed arrays (verts_packed (sum n verts from all meshes, 3),
    # faces_packed (sum n faces from all meshes, 3) and so on).

    # a single lr description string or float
    main_lr_fn = parse_lr_schedule_string_into_lr_lambda(str(deform_by_csd_cfg.lr))
    other_shared_params_lr_fn = main_lr_fn

    #### parse schedule strings and get functions taking epoch and outputting weight/lr
    visual_loss_weight_fn = parse_lr_schedule_string_into_lr_lambda(
        deform_by_csd_cfg.visual_loss_weight_schedule
    )
    jacobian_id_loss_weight_fn = parse_lr_schedule_string_into_lr_lambda(
        deform_by_csd_cfg.jacobian_id_loss_weight_schedule
    )
    verts_normals_and_scale__scale_id_loss_weight_fn = (
        parse_lr_schedule_string_into_lr_lambda(
            deform_by_csd_cfg.verts_normals_and_scale__scale_id_loss_weight_schedule
        )
    )

    #### misc grabs from config
    view_batch_size = deform_by_csd_cfg.view_batch_size
    mesh_batch_size = deform_by_csd_cfg.mesh_batch_size

    #### initial inflation (before any remeshing) to add more volume
    if deform_by_csd_cfg.initial_inflate:
        # modifies meshes_structs inplace and returns new pt3d_batched_meshes
        pt3d_batched_meshes = loophelp__initial_inflate(
            deform_by_csd_cfg.initial_inflate,
            deform_by_csd_cfg.adjust_things_via_selection_volume,
            pt3d_batched_meshes,
            original_vertex_selection_mask_packed,
            meshes_structs,
        )

    #### init solver, quantity being optimized, optimizer; optionally after initial remesh
    if remesh_cfg and not remesh_cfg.override__no_remesh_once_at_start:
        #### remesh and init (it says 'resume' but None for qty and optimizer means init)
        (
            pt3d_batched_meshes,
            my_solvers,
            quantity_being_optimized,
            optimizer,
            original_geometry_xyz_for_current_triangles,
        ) = loophelp__remesh_and_resume(
            device=device,
            deform_by_csd_cfg=deform_by_csd_cfg,
            meshes_structs___INPLACE=meshes_structs,
            n_meshes=n_meshes,
            new_verts_list_and_intermediate_results__alldataset=tuple(
                zip(pt3d_batched_meshes.verts_list(), (None for _ in range(n_meshes)))
            ),
            start_optimizer_at_lr=main_lr_fn(1),  # optim_i is 1-indexed
            remesh_i=-1,  # -1 has special meaning (the init remesh), 0 is the first optim-loop remesh
            remesh_cfg=remesh_cfg,
            optimizer___=None,  # none optimizer also indicates init remesh, since no state yet
            quantity_being_optimized___=None,
            pt3d_batched_meshes___=pt3d_batched_meshes,
            pt3d_batched_meshes_original=pt3d_batched_meshes__original,
            original_geometry_xyz_for_current_triangles___=original_geometry_xyz_for_current_triangles,
        )
    else:
        #### no remeshing config, just initialize without remeshing
        vertex_selection_mask_list = tuple(
            mesh_struct.vertex_selection_mask for mesh_struct in meshes_structs
        )
        my_solvers, _ = loophelp__init_solver(
            deform_by_csd_cfg,
            pt3d_batched_meshes,
            vertex_selection_mask_list,
        )
        quantity_being_optimized = loophelp__init_optim_qty(
            deform_by_csd_cfg,
            pt3d_batched_meshes,
            my_solvers,
            deform_by_csd_cfg.local_step_procrustes,
        )
        optimizer = loophelp__init_optimizer(
            deform_by_csd_cfg, quantity_being_optimized, main_lr_fn(1)
        )

    # save the initialization going into the loop (after inflate + init remesh)
    loophelp__save_deformremesh_result(
        deform_by_csd_cfg=deform_by_csd_cfg,
        dataset_cfg=dataset_cfg,
        pt3d_batched_meshes=pt3d_batched_meshes,
        pt3d_batched_meshes__original=pt3d_batched_meshes__original,
        original_geometry_xyz_for_current_triangles=original_geometry_xyz_for_current_triangles,
        original_vertex_selection_mask_packed=original_vertex_selection_mask_packed,
        current_vertex_selection_mask_packed=grab_vselection_mask_packed_from_meshes_structs(
            meshes_structs
        ),
        original_piggyback_vertex_attributes_packed=original_piggyback_vertex_attributes_packed,
        current_piggyback_vertex_attributes_packed=grab_piggyback_vertex_attributes_packed_from_meshes_structs(
            meshes_structs
        ),
        new2old_fi_packed=np.concatenate(
            tuple(mesh_struct.new2old_fi for mesh_struct in meshes_structs), axis=0
        ),
        n_meshes=n_meshes,
        remesh_str_in_fname="rmsh0-initialization-",
        optim_i=0,
        _drmshdebug_save_yzrot=DRMSHDEBUG_SAVE_YZROT,
    )

    print("LOSSLOG")  # for scripts to know where the loss log starts in stdout file
    from_epoch = deform_by_csd_cfg.start_from_epoch

    detect_anomaly = get_bool_env_variable("TORCH_PLS_DETECT_ANOMALY")
    if detect_anomaly:
        thlog.info("enabling torch anomaly detection")
    torch.autograd.set_detect_anomaly(detect_anomaly)
    for optim_i in range(from_epoch, deform_by_csd_cfg.n_iters + 1):
        this_iter_needs_viz = (
            deform_by_csd_cfg.view_once_every > 0
            and optim_i % deform_by_csd_cfg.view_once_every == 0
        ) or (optim_i == deform_by_csd_cfg.view_once_every)

        # evaluate the LR schedule functions and set the current learning rates
        set_learning_rate(
            optimizer, main_lr_fn(optim_i), other_shared_params_lr_fn(optim_i)
        )

        # evaluate the schedule functions to get the current epoch's loss weights
        visual_loss_weight = visual_loss_weight_fn(optim_i)
        jac_id_loss_weight = jacobian_id_loss_weight_fn(optim_i)
        vns_scale_id_loss_weight = verts_normals_and_scale__scale_id_loss_weight_fn(optim_i)
        stage_II_loss_weight = guidance.stage_II_loss_weight_fn(optim_i)

        # print a line in the polyscope recording
        if this_iter_needs_viz:
            thlog.log_in_ps_recording(
                f"optim iter {optim_i}, visual loss x{visual_loss_weight:.6f} jac id loss x{jac_id_loss_weight:.6f} stage2 loss x{stage_II_loss_weight:.6f}"
            )

        # iterate thru batches in the dataset
        shuffled_mesh_indices = torch.randperm(n_meshes, generator=rng)
        # save the deformed verts lists from all batches in a sequence this will
        # have n_meshes verts arrays + intermediate results, one for each
        # deformed mesh in this optim_i
        nvlir__alldataset: List[
            Optional[Tuple[torch.Tensor, deformations.DeformationIntermediateResults]]
        ] = [None for _ in range(n_meshes)]
        for mesh_indices_this_batch in torch.split(shuffled_mesh_indices, mesh_batch_size):
            # gather mesh_structs and pytorch3d meshes from these indices
            mesh_indices_this_batch = cast(List[int], mesh_indices_this_batch.tolist())
            meshes_structs_this_batch = [meshes_structs[i] for i in mesh_indices_this_batch]
            pt3d_meshes_this_batch = pt3d_batched_meshes[mesh_indices_this_batch]
            optim_quantity_this_batch = quantity_being_optimized[mesh_indices_this_batch]
            my_solvers_this_batch = my_solvers[mesh_indices_this_batch]

            # calculate deformed verts for meshes in the batch
            new_verts_list, intermediate_results = calc_deformed_verts_according_to_cfg(
                meshes_structs_this_batch,
                pt3d_meshes_this_batch,
                deform_by_csd_cfg.solve_method,
                deform_by_csd_cfg.arap_energy_type,
                deform_by_csd_cfg.postprocess_after_solve,
                my_solvers_this_batch,
                optim_quantity_this_batch,
            )

            # render meshes using these new verts
            train_renders = []
            prompt_zs = []
            prompt_neg_zs = []
            psmeshes = []  # polyscope mesh structures (or their psrec proxies) for viz
            for idx_in_batch, (mesh_struct, new_verts) in enumerate(
                zip(meshes_structs_this_batch, new_verts_list)
            ):
                # register the deformed mesh in polyscope for viewing/logging
                if this_iter_needs_viz:
                    psmesh = thlog.psr.register_surface_mesh(
                        f"deformed{idx_in_batch}",
                        new_verts.cpu().detach().numpy(),
                        cast(torch.Tensor, mesh_struct.nvdm_loaded_mesh.t_pos_idx)
                        .cpu()
                        .detach()
                        .numpy(),
                    )
                    psmeshes.append(psmesh)
                    # optimizer running state viz for verts_normals deform qty
                    if (
                        thlog.guard(VIZ_DEBUG, needs_polyscope=True)
                        and optim_i > 1
                        and optim_quantity_this_batch.this_is.startswith("verts_normals")
                        and (deformqtyoptstate := optimizer.state_dict()["state"].get(0))
                        is not None
                    ):
                        expavgsq = deformqtyoptstate["exp_avg_sq"].cpu().detach().numpy()
                        if optim_quantity_this_batch.this_is == "verts_normals_and_scale":
                            _scl = (
                                optim_quantity_this_batch.tensor[:, 3:]
                                .cpu()
                                .detach()
                                .numpy()
                            )
                            psmesh.add_scalar_quantity(
                                f"vns_scl_mag", np.linalg.norm(_scl, axis=-1)
                            )
                            psmesh.add_scalar_quantity(
                                f"vns_scl_expavgsq_mag{idx_in_batch}",
                                np.linalg.norm(expavgsq[:, 3:6], axis=-1),
                            )
                            psmesh.add_scalar_quantity(
                                f"vns_dir_expavgsq_mag{idx_in_batch}",
                                np.linalg.norm(expavgsq[:, 0:3], axis=-1),
                            )

                # # render this mesh with the new verts and a batch of views
                train_render = guidance.render_single_mesh(
                    optim_i, mesh_struct.nvdm_loaded_mesh, new_verts
                )
                train_renders.append(train_render)

                # while we're here looping thru individual `mesh_struct`s: NOTE
                # mesh_struct.prompt_z and prompt_negative_z are shape (1, 77, 4096)
                # so when we cat(prompt_zs) later, we get (mesh_batch_size, 77, 4096)
                # then we'll need repeat_interleave to duplicate the individual
                # zs (interleaved in original order) `view_batch_size` times
                # in order to get the same final batch size as cat(train_renders)
                prompt_zs.append(mesh_struct.prompt_z)
                prompt_neg_zs.append(mesh_struct.prompt_negative_z)
            # end per-mesh render loop

            # concat the render batches from the meshes in the batch.
            # there are mesh_batch_size meshes, each rendered with
            # view_batch_size views, so train_renders should end up with shape
            # (view_batch_size * mesh_batch_size, channels, h, w)
            train_renders = torch.cat(train_renders, dim=0)

            # cat(prompt_zs) has shape (mesh_batch_size, 77, 4096), but the
            # render batches were view_batch_size so we need repeat_interleave
            # to get (mesh_batch_size * view_batch_size, 77, 4096)
            prompt_zs = torch.cat(prompt_zs, dim=0)
            prompt_zs = prompt_zs.repeat_interleave(view_batch_size, dim=0)
            prompt_neg_zs = torch.cat(prompt_neg_zs, dim=0)
            prompt_neg_zs = prompt_neg_zs.repeat_interleave(view_batch_size, dim=0)

            # register train renders in polyscope to view
            if this_iter_needs_viz:
                # train_renders is (b, c, h, w)
                ps_images = train_renders.cpu().detach()[:8]  # just view the first 8 images
                orig_image_width = ps_images.size(3)
                # scale down to save space in the psrec recording
                ps_images = resize_right.resize(
                    ps_images, scale_factors=(128 / orig_image_width)
                ).permute(0, 2, 3, 1)
                # remove any 4th channel if present, keep only rgb
                ps_images = ps_images[..., :3]
                image_batch_size, image_height, image_width, image_n_channels = (
                    ps_images.shape
                )
                thlog.psr.add_color_image_quantity(
                    "renders",
                    PSRSpecialArray.image_array_as_png_bytes(
                        ps_images.reshape(
                            image_batch_size * image_height, image_width, image_n_channels
                        ).numpy()
                    ),
                    enabled=True,
                )

                thlog.psr.show()

            # calc losses and optimize
            optimizer.zero_grad()

            for accum_i in range((n_accum_iters := deform_by_csd_cfg.n_accum_iters)):
                # the first return item is the loss tensor; the rest are just python scalars
                visual_loss, stage_I_loss_val, stage_II_loss_val, total_visual_loss_val = (
                    guidance.calc_loss(optim_i, train_renders, prompt_zs, prompt_neg_zs)
                )

                # add additional losses
                # jacobian identity regularization, for faces_jacobians
                this_is = optim_quantity_this_batch.this_is
                if this_is != "verts_offsets" and jac_id_loss_weight > 0:
                    jac_id_loss, jac_id_loss_val = calc_jacobian_id_loss(
                        optim_quantity_this_batch, intermediate_results, jac_id_loss_weight
                    )
                else:
                    jac_id_loss = None
                    jac_id_loss_val = 0.0

                if is_vns := (this_is == "verts_normals_and_scale"):
                    vns_scale_id_loss, vns_scale_id_loss_val = (
                        calc_verts_normals_and_scale__scale_id_loss(
                            optim_quantity_this_batch,
                            meshes_structs_this_batch,
                            "det",
                            vns_scale_id_loss_weight,
                            deform_by_csd_cfg.adjust_things_via_selection_volume,
                        )
                    )
                else:
                    vns_scale_id_loss, vns_scale_id_loss_val = None, 0.0

                # add up the losses
                loss = visual_loss_weight * visual_loss
                if jac_id_loss is not None:
                    loss = loss + jac_id_loss
                if vns_scale_id_loss is not None:
                    loss = loss + vns_scale_id_loss

                # do backward and step
                is_last_accum_iter = accum_i == (n_accum_iters - 1)
                loss.backward(retain_graph=not is_last_accum_iter)
                if deform_by_csd_cfg.step_after_every_backward or is_last_accum_iter:
                    loophelp__zero_out_nan_grads_inplace(
                        quantity_being_optimized, intermediate_results
                    )
                    optimizer.step()

                # use the loss float values to print
                if is_last_accum_iter:
                    if vns_scale_id_loss_val:
                        id_loss_val_to_show = f"{vns_scale_id_loss_val:.6f}"
                    elif jac_id_loss_val:
                        id_loss_val_to_show = f"{jac_id_loss_val:.6f}"
                    else:
                        id_loss_val_to_show = "0"
                    print(
                        f"{optim_i},{total_visual_loss_val:.6f},{stage_I_loss_val:.6f},{stage_II_loss_val:.6f},{id_loss_val_to_show}",
                        flush=True,
                    )
            # end accum loop

            # save detached deformed verts for this batch
            intermediate_results_list = (
                intermediate_results.get_sequence(
                    pt3d_batched_meshes.num_verts_per_mesh().tolist()
                )
                if intermediate_results is not None
                else tuple(None for _ in new_verts_list)
            )
            for (
                mesh_idx_in_dataset,
                new_verts_this_mesh,
                intermediate_results_this_mesh,
            ) in zip(mesh_indices_this_batch, new_verts_list, intermediate_results_list):
                nvlir__alldataset[mesh_idx_in_dataset] = (
                    new_verts_this_mesh.detach(),
                    intermediate_results_this_mesh,
                )
        # end dataset batches loop

        assert all(map(lambda t: t is not None, nvlir__alldataset)), (
            "somehow none found in new_verts_list_and_intermediate_results__alldataset"
        )

        # can cast to remove the optional now, after assert
        new_verts_list_and_intermediate_results__alldataset = cast(
            List[
                Tuple[
                    torch.Tensor,
                    deformations.DeformationIntermediateResults,
                ]
            ],
            nvlir__alldataset,
        )

        # save if optim_i is one of the indicated save epochs, or is last epoch
        do_save_here = (optim_i in deform_by_csd_cfg.save_at_epochs) or (
            optim_i == deform_by_csd_cfg.n_iters
        )
        if do_save_here:
            pt3d_batched_meshes__forsave = pt3d_batched_meshes.offset_verts(
                torch.cat(
                    tuple(
                        tup[0]
                        for tup in new_verts_list_and_intermediate_results__alldataset
                    ),
                    dim=0,
                )
                - pt3d_batched_meshes.verts_packed()
            )

            loophelp__save_deformremesh_result(
                deform_by_csd_cfg,
                dataset_cfg,
                pt3d_batched_meshes__forsave,
                pt3d_batched_meshes__original,
                original_geometry_xyz_for_current_triangles,
                original_vertex_selection_mask_packed,
                grab_vselection_mask_packed_from_meshes_structs(meshes_structs),
                original_piggyback_vertex_attributes_packed,
                grab_piggyback_vertex_attributes_packed_from_meshes_structs(meshes_structs),
                new2old_fi_packed=np.concatenate(
                    tuple(mesh_struct.new2old_fi for mesh_struct in meshes_structs), axis=0
                ),
                n_meshes=n_meshes,
                remesh_str_in_fname="deformedonly",
                optim_i=optim_i,
            )

        if remesh_cfg and not remesh_cfg.override__remesh_only_once_at_start:
            if (
                remesh_i := remesh_cfg.get_remesh_iter_number(
                    optim_i, deform_by_csd_cfg.n_iters
                )
            ) is not None:
                # replaces these variables declared before the loop, for
                # subsequent iters and modifies meshes_structs inplace
                (
                    pt3d_batched_meshes,
                    my_solvers,
                    quantity_being_optimized,
                    optimizer,
                    original_geometry_xyz_for_current_triangles,
                ) = loophelp__remesh_and_resume(
                    device=device,
                    deform_by_csd_cfg=deform_by_csd_cfg,
                    meshes_structs___INPLACE=meshes_structs,
                    n_meshes=n_meshes,
                    new_verts_list_and_intermediate_results__alldataset=new_verts_list_and_intermediate_results__alldataset,
                    start_optimizer_at_lr=main_lr_fn(optim_i + 1),
                    # ^ this doesn't matter because this will get set again to this value at
                    # the start of the next epoch/optim_i, which begins right after remesh_and_resume
                    remesh_i=remesh_i,
                    remesh_cfg=remesh_cfg,
                    optimizer___=optimizer,
                    quantity_being_optimized___=quantity_being_optimized,
                    pt3d_batched_meshes___=pt3d_batched_meshes,
                    pt3d_batched_meshes_original=pt3d_batched_meshes__original,
                    original_geometry_xyz_for_current_triangles___=original_geometry_xyz_for_current_triangles,
                )
                if do_save_here or remesh_i in remesh_cfg.save_at_remesh_i:
                    thlog.info("saving deformremesh result")
                    loophelp__save_deformremesh_result(
                        deform_by_csd_cfg,
                        dataset_cfg,
                        pt3d_batched_meshes,
                        pt3d_batched_meshes__original,
                        original_geometry_xyz_for_current_triangles,
                        original_vertex_selection_mask_packed,
                        grab_vselection_mask_packed_from_meshes_structs(meshes_structs),
                        original_piggyback_vertex_attributes_packed,
                        grab_piggyback_vertex_attributes_packed_from_meshes_structs(
                            meshes_structs
                        ),
                        new2old_fi_packed=np.concatenate(
                            tuple(mesh_struct.new2old_fi for mesh_struct in meshes_structs),
                            axis=0,
                        ),
                        n_meshes=n_meshes,
                        remesh_str_in_fname=f"rmsh{remesh_i}",
                        optim_i=optim_i,
                    )
            # end "found remesh_i in the list of iters to remesh" branch
        # end remesh branch
    # end optim loop
    print("END LOSSLOG", flush=True)


@thronfigure
def main(config: MainConfig):
    thlog.info(
        f"\nCONFIGPRINT\n{(cfg_string := config.to_json_string(pretty=True))}\nEND CONFIGPRINT"
    )
    # save directories must exist (or create directories on the user's behalf)
    psrec_dir = os.path.dirname(config.ps_recording_save_fname)
    drmsh_dir = os.path.dirname(config.deform_by_csd.optimized_quantity_save_fname)
    thlog.info(f"""Creating save directories if they don't already exist:
    for psrec-*.npz recording: {psrec_dir}
    for drmsh-*.npz result files: {drmsh_dir}
    """)
    os.makedirs(psrec_dir, exist_ok=True)
    os.makedirs(drmsh_dir, exist_ok=True)

    thlog.init_polyscope(start_polyscope_recorder=True)

    def __save_psrec():
        ps_recording_save_fname = config.ps_recording_save_fname
        if not get_bool_env_variable("DRMSH_ALLOW_SAVE_FILE_OVERWRITES"):
            if os.path.isfile(ps_recording_save_fname):
                # append a disambiguating, incrementing number like on windows (Copy (1), etc)
                fname_noext, ext = os.path.splitext(ps_recording_save_fname)
                ps_recording_save_fname = next_increment_path(fname_noext + "@{:03}" + ext)

        thlog.save_ps_recording(ps_recording_save_fname, comment=cfg_string)

    try:
        from functools import partial

        device = torch.device(config.device)
        # can branch on other guidance method configs and define their method
        # init functions here, if any are added
        csd_guidance = config.deform_by_csd.csd_guidance
        guidance_method_init = partial(CSDGuidanceMethod, csd_guidance, device)
        submain_deformremesh(
            config.deform_by_csd,
            config.dataset,
            device=device,
            guidance_method_init=guidance_method_init,
        )
    except ValueError as e:
        import traceback

        thlog.err("submain loop crashed, saving early! this was the traceback")
        traceback.print_exception(e)
        thlog.err("Done printing traceback")
    finally:
        __save_psrec()


if __name__ == "__main__":
    main()
