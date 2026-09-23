# qt-PEGASIS-Fp-Sage

This code copies most functionalities from the original qt-Pegasis implementation https://github.com/KULeuven-COSIC/qt-pegasis by Pierrick Dartois, Jonathan Komada Eriksen, Riccardo Invernizzi and Frederik Vercauteren presented in the paper https://eprint.iacr.org/2025/1859. However, we also implement a version of qt-Pegasis relying on the faster new formulae from our paper (Dartois and Duparc, 2026) to work fully over $$\mathbb{F}_p$$.

## Code structure

- Main functions:
    - `qt_pegasis.py`: main algorithm class
    - `norm_eq.py`: solving the norm equation
- Subroutines:
    - `sos.py` and `const_precomp.py`: solving sums of squares
    - `ideals.py`: ideal helpers
    - `xonly.py`: x-only arithmetic
    - `hd_helpers`: wrapper for 4D theta isogenies
    - `theta_lib/`: theta isogenies, contains legacy formulae by Dartois from https://eprint.iacr.org/2024/1180 and the new formulae introduced in our paper (Dartois and Duparc, 2026).
- Parameters:
    - `params.py`: parameters loading
    - `params/`: parameter sets for different security levels
    - `pgen.py`: generate new parameters given p and A
- Benchmarking and testing
    - `benchmarks/`
- Precomputations:
    - `precomp/` to optimize qt-Pegasis (computing rabbit skeletons, gluing and splitting change of theta coordinates matrices)

## Example usage

To run an action with our new faster formulae and work fully over $$\mathbb{F}_p$$, type in a Sage terminal:
```
from qt_pegasis import qtPegasisFp

lvl = '500'
EGA = qtPegasisFp(lvl)
frak_a = EGA.sample_ideal()
EGA.qt_action(frak_a)
```

To run an action with former formulae, type:
```
from qt_pegasis import qtPegasis

lvl = '500'
EGA = qtPegasis(lvl)
frak_a = EGA.sample_ideal()
EGA.qt_action(frak_a)
```

The level `lvl` corresponds to the bitsize of the base prime; available levels are:
- `'500'`
- `'1000'`
- `'1500'`
- `'2000'`
- `'4000'`
- `'500P'`
- `'1000P'`
- `'1500P'`
- `'2000P'`
- `'4000P'`
the `P` at the end denotes PEGASIS primes, while a number without `P` denotes qtPegasis primes.
New parameters can be generated as described below.

The function `qt_action` assumes the ideal to be a pair `[ell, omega-lambda]`.
To act with a `sage` ideal the wrapper `sage_action` can be used as follows:

```
from qt_pegasis import qtPegasisFp

lvl = '500'
EGA = qtPegasisFp(lvl)
frak_a = EGA.sample_sage_ideal()
EGA.sage_action(frak_a)
```

Warning: `sage` structures may cause a significant slowdown especially at
higher levels.

## Benchmarks

To reproduce benchmarks from Table 2 of our paper comparing our new implementation of qt-Pegasis `qtPegasisFp` with the former one `qtPegasis`, type in a Sage terminal:

```
load("benchmarks/compare.py")
```

Warning: This will launch 100 runs for all qt-Pegasis levels (`'500'`, `'1000'`, `'1500'`, `'2000'`, `'4000'`) so the execution of this instruction may take several hours. We recommend to reduce the number of runs to 10 to obtain benchmarks in a reasonable amount of time.

The benchmarks will be appended for all levels to a file `times_compare.txt` located in the main `Sage_Code` folder.

## Testing correctness

To test correctnes of our new implementation of qt-Pegasis `qtPegasisFp`, type in a Sage terminal

```
load("benchmarks/test_qt_Fp.py")
```

This will test the commutativity of the ideal class group action $$[\mathfrak{a}]\cdot[\mathfrak{b}]\cdot E=[\mathfrak{b}]\cdot[\mathfrak{a}]\cdot E$$ 10 times at level `'500'`. To change the number of runs (10 by default) or the level, the file `benchmarks/test_qt_Fp.py` can be modified accordingly.

## Parameters and parameter generation

Parameter sets for different security levels are stored in the `params/` folder
in `json` format. For security level `lvl` the file `params/{lvl}.json`
contains the qt-Pegasis parameter set, while `params/{lvl}P.json` contains the
PEGASIS parameter set. To create a new parameter set, follow the instructions
in `pgen.py`. This generates a new file `params/name.json` that can be later
loaded with `qtPegasis(name)`.

## Precomputations

We explain here how to precompute rabbit skeletons and matrices that appear in Appendices A and B of our paper (Dartois and Duparc, 2026).

To precompute rabbit skeletons, type in a Sage terminal:

```
load("precomp/rabbit_finding.py")
```

To precompute gluing change of theta coordinates matrices, type in a Sage terminal:

```
load("precomp/gluing_matrices.py")
```

To precompute splitting change of theta coordinates matrices, type in a Sage terminal:

```
load("precomp/splitting_matrices.py")
```

## Logging

Additional information is available at runtime by activating logging. Loggers
for `qt_pegasis` and `hd_helpers` can be made verbose with
```
logging.getLogger('qt_pegasis').setLevel(logging.INFO)
logging.getLogger('norm_eq').setLevel(logging.INFO)
```
The default level is `logging.WARNING` (no printing), while `logging.INFO`
prints timings for the main steps and `logging.DEBUG` adds times for
subroutines.