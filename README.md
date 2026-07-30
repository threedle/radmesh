# RADmesh: Remesh-Aware Mesh Deformation

The official implementation for our **ECCV 2026 (Oral)** paper.

[project page](https://threedle.github.io/radmesh/) | [paper (pdf)](https://people.cs.uchicago.edu/~namanh/papers/radmesh.pdf) | [BibTeX](#bibtex)

![teaser figure. 4 frames, showing 4 prompts that result in deforming and remeshing of a selection region on the Spot cow mesh: the prompt prefix is 'a 3d render of a cute chibi cow', and the prompts continue respectively: wearing a bowler hat (selection on top of head), on big wheels (selection is the feet), with a long lizard tail (selection is tail), with two pegasus wings (selection is back)](http://people.cs.uchicago.edu/~namanh/remoteassets/radmesh/teaser-v7-img.jpg)

---

## Installation

### System requirements
Optimizations have been tested to run on a single A40 (48GB) or single L40S GPU. Your GPU should have enough memory to hold `DeepFloyd/IF-I-XL-v1.0` and
`DeepFloyd/IF-II-L-v1.0` models. You should have a CUDA toolkit installed.
### Environment

Create or start with a conda environment with `python>=3.10,<3.13`. To start fresh,
```sh
conda env create -n radmesh "python>=3.10,<3.13"
```

Inspect the script `env_setup.sh` to change the CUDA version if you need to. Make sure you're in your conda environment, then execute the `env_setup.sh` script (or run the commands manually yourself) to install dependencies.

#### Minimal usage
If you just need `radmesh/deformations.py` to use the geometry processing code, then only `numpy`, `scipy`, `torch`, `libigl>=2.6.2`, `cholespy`, and `thlog` are needed. (See below for a command to install just `thlog`.)

Note that `radmesh/deformations.py` depends on `radmesh/pytorch3d/`, which only requires `torch`. 

> (We've vendored just the useful parts of `pytorch3d` for this, to prevent a bulky installation of the whole `pytorch3d` package.) 

### DeepFloyd IF

To set up HuggingFace Hub and your account for downloading the DeepFloyd stages (instructions from [DeepFloyd IF](https://github.com/deep-floyd/IF)):

1) If you do not already have one, create a [Hugging Face account](https://huggingface.co/join)
2) Accept the license on the model card of [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)
3) Log in to Hugging face locally. In the conda environment you just created, install `huggingface_hub`
```
pip install huggingface_hub --upgrade
```
run the login function in a python shell
```
from huggingface_hub import login

login()
```
and enter your [Hugging Face Hub access token](https://huggingface.co/docs/hub/security-tokens#what-are-user-access-tokens).



## Usage

### Preparing your run
Take a look at `example-config-localized.json`. We've included an example input mesh and prompt in `example-run`, which you can run immediately with this config file (see [Running](#running)).

For your inputs: edit paths to the input mesh, vertex selection file, output files, and the prompt. 

- The `example-config-localized.json` file is for runs with a selection region. Deformation and remeshing will only occur within that region. `example-config-wholemesh.json` is for runs without a selection region, i.e. the whole mesh is allowed to remesh and deform.

The input mesh file can be any format supported by `igl.read_triangle_mesh`. The vertex selection file, if not `null`, should be an `.npy` file of a boolean array of shape `(n_verts,)`, where True indicates the vertex is selected and enabled for deformation and remeshing.

### Hyperparameters

The default hyperparameters in the two files should suffice in most cases (`example-config-localized.json` for localized runs and `example-config-wholemesh.json` for whole-mesh/global runs). However, inspect the hyperparameters and see if there are any you would like to change before running.


**Some adjustments you may wish to consider depending on your mesh:**
- If you can, it is recommended to isotropically remesh your input mesh first to have a face count in the 8k-20k range
    (just so that the average edge length calculation for our default `targetlen_schedule` settings is a reasonable value, since that schedule is, by default, a multiplier on the initial average edge length). However:
    -  Higher-resolution meshes should still work fine, but for best results (since the best overall deformations are achieved with a coarse-to-fine schedule, emphasis on the initial coarseness), some changes to the
    `targetlen_schedule` may be applied as needed.
    - Meshes/regions with non-isotropic triangulations (such as a triangulation of a quad-dominant mesh) should also work, since we will do isotropic remeshing to begin with.
    - Meshes with multiple connected components are supported. However, each component must be manifold.
- For human shapes and other tall, slender shapes that take up little volume when normalized to fit the standard cube bounding box, consider using a `dist_minmax` of `[1.4, 2.6]` and `elev_minmax` of `[0.0, 30.0]`.
- If you find that initial inflation in the normal direction of the selection region is too large, try dilating the selection to have a larger
area/volume (the volume after hole-closing, which you can see in a printout.) Likewise, if you'd like
a larger initial inflation as seed geometry to start optimization with, also consider eroding
the selection so that the initial inflation heuristic computes a bigger length.
    - To judge this, check the initialization mesh (after initial inflation + initial remesh). It is saved as a file with the filename pattern `drmsh-*-rmsh0-initialization--optm0.npz`, which you can view with `view_drmsh_npz.py`. 
    
    - You can also see it as the first frame of the saved `psrec-*.npz` recording of a complete optimization run.

- Optimization in general is a little sensitive to the initialization, so if you aren't getting satisfactory results, try changing the above settings related to this initial inflation and your selection region.

### Running
Run with:
```sh
python run_optimization.py -c example-config-localized.json
```
(you can override fields on the command line; see `python run_optimization.py -h`)

Environment variables to make visible to the python process:
- `NO_POLYSCOPE=1` (required for headless systems)
- `CUDA_HOME` pointing to a CUDA toolkit installation matching the CUDA version you installed packages for. 
 `nvdiffmodeling` will compile an extension module at runtime.

### Viewing results

A recording `.npz` file is saved (e.g. `example-run/psrec-spot-wings.npz`); play the recording with
```
thlog replay example-run/psrec-spot-wings.npz
```
on a system with a monitor (you only need to install `thlog` and `pillow` for playback.)
- The command to install `thlog` with the `pillow` dependency is
    ```bash
    pip install 'thlog[pil] @ https://github.com/namanhd/thlog/archive/main.zip'
    ```


A result `.npz` file is saved regularly (see `save_at_epochs` in the config); these files can be viewed with 
```
python view_drmsh_npz.py <drmsh-filename-here>
```
- (You only need `numpy` and `polyscope`, which are also installed when you install `thlog`.)

## Caveats

- We don't fix the seed in general because this pipeline is nondeterministic even with seeding,
due to nondeterministic GPU algorithms (e.g. sparse `bmm`) compounding over the course of thousands
of epochs. Set the environment variable `TORCH_PLS_BE_DETERMINISTIC=1` for deterministic
algorithms, but this comes with a heavy performance penalty.

- Because of this, you can try doing a run several times to see (possibly better) variant results.

# BibTeX

```
@inproceedings{dinh2026radmesh,
  title     = {RADmesh: Remesh-Aware Mesh Deformation},
  author    = {Dinh, Nam Anh and Lang, Itai and Stein, Oded and Hanocka, Rana},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```
