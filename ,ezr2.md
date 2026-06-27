<!-- Copyright (c) 2026 Tim Menzies, MIT License https://opensource.org/licenses/MIT -->
<a href="https://timm.fyi"><img align="right" alt="Author" src="https://img.shields.io/badge/Author-timm-dc143c?logo=readme&logoColor=white"></a><img align="right" alt="Language" src="https://img.shields.io/badge/Language-Python%203.12+-000080?logo=python&logoColor=white"><img align="right" alt="Deps" src="https://img.shields.io/badge/Deps-0-32cd32?logo=checkmarx&logoColor=white"><a href="https://choosealicense.com/licenses/mit/"><img align="right" alt="License" src="https://img.shields.io/badge/License-MIT-32cd32?logo=open-source-initiative&logoColor=white"></a>

### [http://tiny.cc/ezr2](http://tiny.cc/ezr2)

**ezr2** — explainable multi-objective optimization in one file, **zero
dependencies**, pure Python stdlib. Build a large pool of candidate rows,
recursively split it by far-point projection (a descendant of SWAY), label
only a few dozen informative rows, then sort the rest. A regression *or*
classification tree explains which input ranges lead to the best goals.

Runs on **data** (a CSV) or a live **model** (override `labelled()` so a
row's goals are computed on demand — see `dtlz.py`).

**Files:** [ezr2.py](http://tiny.cc/ezr2#file-ezr2-py) | [test_ezr2.py](http://tiny.cc/ezr2#file-test_ezr2-py) | [dtlz.py](http://tiny.cc/ezr2#file-dtlz-py) | [Growing.py](http://tiny.cc/ezr2#file-growing-py) | [pctl.py](http://tiny.cc/ezr2#file-pctl-py) | [ezr2.md](http://tiny.cc/ezr2#file-ezr2-md) | [Random.md](http://tiny.cc/ezr2#file-random-md) | [Growing.md](http://tiny.cc/ezr2#file-growing-md) | [Makefile](http://tiny.cc/ezr2#file-makefile) | [pyproject.toml](http://tiny.cc/ezr2#file-pyproject-toml) | [LICENSE.md](http://tiny.cc/ezr2#file-license-md)

```bash
# sibling data gist supplies the CSVs (no data lives in here)
git clone http://tiny.cc/optimiz                 # optimization data
git clone http://tiny.cc/ezr2 && cd ezr2
python3 test_ezr2.py -h                           # options + test names
python3 test_ezr2.py tree                         # build + show a tree
python3 test_ezr2.py --file=../optimiz/auto93.csv disty
python3 test_ezr2.py all                          # run every self-test
make test                                         # same, via konfig
```

### Knobs (`--key=val`)

`--file --seed --leaf --maxd --grow --budget --cap --check --keepf --round
--landscape`. See `python3 test_ezr2.py -h`.

### Data format

First row names the columns; a name's **last char** sets its role, its
**first char** its type:

- upper-case first letter → numeric (else symbolic)
- `+` / `-` suffix → goal to maximize / minimize
- `!` suffix → klass; `X` → ignore; `~` → protected x-column; else → input x

```
Clndrs,Volume,HpX,Model,origin,Lbs-,Acc+,Mpg+
```
numeric inputs `Clndrs/Volume/Model`, symbolic input `origin`, ignored `HpX`,
goals minimize `Lbs`, maximize `Acc`/`Mpg`.

### Findings

- **Active beats random ~3:1** where landscapes separate, ties at the
  sparsity ceiling; the edge lives in the hard tail, not the mean
  (`Random.md`).
- **`grow` is the dominant knob** and SWAY's `(keepf=0.5, grow=2)` is the
  weakest setting in its own space — growing a little faster gains ~11 win
  points and the surface is a smooth, low-sensitivity plateau (`Growing.md`).

Data lives in sibling DATA gists (`../optimiz`), never inside this tool gist.
