# Docker image for quantum-isogeny-circuits.
#
# Provides the same environment as described in README.md (Python 3.12,
# SageMath, Qarton), installed the same way: SageMath via a conda/mamba
# environment, and Qarton via the same steps as install_qarton.sh.
#
# Build:
#   docker build -t quantum-isogeny-circuits .
#
# Run a script (see README.md "Using Docker" for the full list):
#   docker run --rm quantum-isogeny-circuits python build_isogeny_chain_circuit.py --level 500
#
# Run the test suite:
#   docker run --rm quantum-isogeny-circuits pytest .
#
# Get an interactive shell:
#   docker run --rm -it quantum-isogeny-circuits bash

FROM condaforge/miniforge3:latest

# git is required to install qarton straight from its repository, exactly like
# install_qarton.sh does.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create a conda environment named "sage" containing SageMath, following the
# same conda-based installation as README.md. This also provides Python 3.12.
RUN mamba create -n sage -y sage \
    && mamba clean -afy

# Make the "sage" environment the default one for every subsequent RUN/CMD,
# instead of having to "conda activate sage" every time.
ENV CONDA_DEFAULT_ENV=sage
ENV PATH=/opt/conda/envs/sage/bin:$PATH

WORKDIR /app

# Install qarton and pytest, mirroring install_qarton.sh exactly.
RUN git clone --depth 1 https://gitlab.inria.fr/capsule/qarton.git /tmp/qarton \
    && pip install /tmp/qarton \
    && rm -rf /tmp/qarton \
    && pip install pytest

# Copy the project last, so that the (slow) environment setup above is cached
# across rebuilds triggered by code changes.
COPY . /app

CMD ["bash"]
