# Quantum circuit for isogenies

This is the code of the paper "Quantum security analysis of unrestricted isogeny-based group actions".

In order to interface it with our results, we included the SageMath implementation
of qt-Pegasis on Fp, taken from [here](https://github.com/Pierrick-Dartois/qt-pegasis-Fp).
This implementation is licensed under an MIT License. Our only modification is that
we included more parameter sets in the sub-folder `params` in order to perform small-scale
experiments.


## Installation

To run this code, you need python 3.12, SageMath, and Qarton. We require no more
dependencies than those of these libraries.

It is possible to setup with Conda.

1. Installing sage with conda

    Follow the instructions [here](https://doc.sagemath.org/html/en/installation/conda.html)
    to install SageMath in a conda virtual environment named `sage`. In detail:

    ```
    conda create -n sage sage
    conda init
    conda activate sage
    ```

2. Installation of Qarton

    You need Qarton >= 1.0.2. Latest version is [here](https://gitlab.inria.fr/capsule/qarton).
    It's simply easier to install it locally. From your virtual environment, use
    the script `install_qarton.sh`. It will download the latest commit on the distant
    qarton repository and install in your virtual environment.

3. Summary

    Using the following commands should setup everything:

    ```
    conda create -n sage sage
    conda init
    conda activate sage
    ./install_qarton.sh
    ```

    Then you can use the `sage` virtual environment running python 3.12.

Changes in the API between versions make this code incompatible with older releases
of Qarton.

## Using Docker

As an alternative to the manual installation above, a `Dockerfile` is provided that
sets up the same environment (Python 3.12, SageMath, and Qarton, installed the same
way as `install_qarton.sh`).

Build the image (this installs SageMath via conda, so it downloads a few GB of data
and can take a while):

```
docker build -t quantum-isogeny-circuits .
```

Run one of the top-level scripts inside a container:

```
docker run --rm quantum-isogeny-circuits python build_isogeny_chain_circuit.py --level 500
docker run --rm quantum-isogeny-circuits python build_norm_equation_circuit.py
docker run --rm quantum-isogeny-circuits python estimate_p1_p2.py
docker run --rm quantum-isogeny-circuits python hsp_exponents.py
docker run --rm quantum-isogeny-circuits python test_norm_equation.py
docker run --rm quantum-isogeny-circuits python test_isogeny_chain.py
```

Reproduce the unit tests with pytest (see "Testing" below):

```
docker run --rm quantum-isogeny-circuits pytest .
```

Get an interactive shell inside the container (with `sage` and `qarton` already
installed and on the `PATH`):

```
docker run --rm -it quantum-isogeny-circuits bash
```

To iterate on the code without rebuilding the image every time, mount the repository
into the container instead:

```
docker run --rm -it -v "$(pwd)":/app quantum-isogeny-circuits bash
```

## Testing

To run all tests with pytest, do: `pytest .`

The tests are written in `*_test.py` files and the test functions that are run are `*_test()`.

## Organization of the code

The following packages contain our new quantum circuits:

* qisogenies/arithmetic: Various arithmetic components such as Cornacchia's algorithm
* qisogenies/montgomery: operations on Montgomery curves
* qisogenies/theta: computing 4-dimensional isogeny chains

The package `qt_pegasis_Fp` is a copy of the implementation of qt-Pegasis on Fp,
available [here](https://github.com/Pierrick-Dartois/qt-pegasis-Fp).

## Running scripts

We provide the following scripts which were used to test our circuits and obtain
our resource counts:

* `test_norm_equation.py`: can be used to test our re-implementation of qlapoti, the
  norm equation solver used in qt-Pegasis. This script will sample random reduced ideals
  and either run the original qlapoti or our new function. It also allows to replace
  some sub-functions by simulated quantum circuits, in order to test these circuits
  at scale.

* `estimate_p1_p2.py`: used to estimate the p1 and p2 probabilities which dictate
  the number of iterations in the norm equation's quantum searches.

* `build_norm_equation_circuit.py`: builds entirely the norm equation circuit (precomputation,
  quantum search and post-computation), counts its resources and displays a table. The
  script can be run with the "chain" strategy or the "pebbled" strategy
  (space-optimized, with more gates).

* `test_isogeny_chain.py`: builds the isogeny chain circuit for n = 500 and simulates it.
  The script relies on the Qarton simulator, which uses "dummified" circuits. It only
  uses the "chain" construction since the "pebbled" construction is outside the reach
  of the Qarton simulator.

* `build_isogeny_chain.py`: builds the isogeny chain and display its resource estimates.
  It can either build the "chain" construction or the "pebbled" construction.

## Statement on AI Tools

We did not use AI in the writing of our paper.

During the writing of this code, we used Claude Sonnet 5 to refactor or modify existing functions
and circuits (for example, starting from the norm equation circuit in the odd case
and adding new code for the even case). This concerns only a few functions and the base
functions and classes were always hand-written. The rest of the code, documentation and 
unit tests were hand-written. In particular, all classical "dummy" functions were hand-written.
