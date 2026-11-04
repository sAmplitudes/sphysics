# CI image for sphysics with all dependencies preinstalled, such that the CI
# jobs do not install them on the fly. Built and pushed by the `ci-image` job
# in .gitlab-ci.yml whenever this file or pyproject.toml changes.
FROM python:3.11-slim-trixie

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

# tools to build the C++ extension (ROOT is not used, hence no X11 libraries)
RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ make git \
 && rm -rf /var/lib/apt/lists/*

# large packages first, such that they end up in layers that rarely change
# CPU-only torch, the default wheel from PyPI pulls several GB of CUDA libraries
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
RUN pip install 'tensorflow==2.21.0' tensorflow_probability tf-keras torchmetrics torchinfo lightning

# build requirements, dependencies, and test extras of sphysics as well as the
# tools used by the CI jobs
COPY pyproject.toml /tmp/pyproject.toml
RUN python -c "import tomllib; p = tomllib.load(open('/tmp/pyproject.toml', 'rb')); print('\n'.join(p['build-system']['requires'] + p['project']['dependencies'] + p['project']['optional-dependencies']['test']))" > /tmp/requirements.txt \
 && pip install -r /tmp/requirements.txt setuptools_scm \
    pylint==4.0.6 pylint-gitlab==2.1.1 pre-commit build twine sphinx-book-theme sphinx_rtd_theme \
 && rm /tmp/pyproject.toml /tmp/requirements.txt

# the job token is only valid while the image-build job is running
ARG CI_JOB_TOKEN
RUN pip install "modernplotting @ git+https://gitlab-ci-token:${CI_JOB_TOKEN}@gitlab.desy.de/belle2/physics/amplitudesatbelleii/softwarestack/modernplotting.git"
