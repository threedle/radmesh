set -e

# change cuda version to what's on your system
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# if you specified a specific torch version above, you must re-specify that same version here
# to avoid transformers and diffusers fetching a bad latest torch version that doesn't match
# the cuda environment installed above
pip install transformers[sentencepiece] diffusers[torch]

# generic dependencies that don't depend on torch. libigl is unstable, consider pinning to 2.6.2
# if something in the future breaks the API. Do NOT use libigl <2.5.4 with numpy>2.0, this combination
# may lead to silent bad results that are hard to track down.
pip install polyscope scipy "libigl>=2.6.2" cholespy PyGLM imageio pymeshlab

# nvdiffrast 
pip install https://github.com/NVlabs/nvdiffrast/archive/main.zip --no-build-isolation

# remesher
pip install https://github.com/namanhd/botsch-kobbelt-remesher-interps/archive/main.zip

# configs and logger
pip install https://github.com/namanhd/thronf/archive/main.zip 'thlog[pil] @ https://github.com/namanhd/thlog/archive/main.zip'


# for the deformation code itself, only numpy, torch, libigl, cholespy are needed