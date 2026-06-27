<!-- Copyright (c) 2026 Tim Menzies, MIT License https://opensource.org/licenses/MIT -->
<a href="https://timm.fyi"><img align="right" alt="Author" src="https://img.shields.io/badge/Author-timm-dc143c?logo=readme&logoColor=white"></a><img align="right" alt="Language" src="https://img.shields.io/badge/Language-Python%203.12+-000080?logo=python&logoColor=white"><img align="right" alt="Deps" src="https://img.shields.io/badge/Deps-0-32cd32?logo=checkmarx&logoColor=white"><a href="https://choosealicense.com/licenses/mit/"><img align="right" alt="License" src="https://img.shields.io/badge/License-MIT-32cd32?logo=open-source-initiative&logoColor=white"></a><img align="right" alt="Purpose" src="https://img.shields.io/badge/Purpose-XAI·Optimization-7b68ee?logo=githubcopilot&logoColor=white">

### [http://tiny.cc/ezr2](http://tiny.cc/ezr2)
ezr2 — explainable multi-objective optimization in one file, **zero
dependencies**, pure Python stdlib. Build a large pool of candidate rows,
recursively split it by far-point projection (a descendant of SWAY), label
only a few dozen informative rows, then sort the rest. A regression *or*
classification tree explains which input ranges lead to the best goals, and
a branch-delta reads as an action. Runs on a CSV, or a live model (override
`labelled()`).

```bash
# sibling data gist supplies the CSVs (no data lives in here)
git clone http://tiny.cc/optimiz                 # optimization data
git clone http://tiny.cc/ezr2 && cd ezr2
python3 test_ezr2.py tree                         # build + show a tree
python3 test_ezr2.py all                          # run every self-test
make test                                         # same, via konfig
```

**Sections:** [NAME](#name) | [SYNOPSIS](#synopsis) | [DESCRIPTION](#description) | [DATA](#data) | [OPTIONS](#options) | [FINDINGS](#findings) | [SEE ALSO](#see-also) | [LICENSE](#license) | [AUTHOR](#author)

**Files:** [ezr2.py](http://tiny.cc/ezr2#file-ezr2-py) | [test_ezr2.py](http://tiny.cc/ezr2#file-test_ezr2-py) | [dtlz.py](http://tiny.cc/ezr2#file-dtlz-py) | [Growing.py](http://tiny.cc/ezr2#file-growing-py) | [pctl.py](http://tiny.cc/ezr2#file-pctl-py) | [ezr2.md](http://tiny.cc/ezr2#file-ezr2-md) | [Random.md](http://tiny.cc/ezr2#file-random-md) | [Growing.md](http://tiny.cc/ezr2#file-growing-md) | [Makefile](http://tiny.cc/ezr2#file-makefile) | [pyproject.toml](http://tiny.cc/ezr2#file-pyproject-toml) | [LICENSE.md](http://tiny.cc/ezr2#file-license-md)

## NAME

    ezr2 - explainable multi-objective optimization via active
           learning and a dual-mode (regression|classification) tree

## SYNOPSIS

    python3 test_ezr2.py [--key=val ...] <test>
    python3 test_ezr2.py -h | all
    python3 dtlz.py [--model=dtlz1..dtlz7] [--M=int] [--N=int]

    Sibling gists (one parent dir; no naked paths):
      ezr2/     this repo (ezr2.py library + test_ezr2.py dispatch)
      optimiz/  optimization CSVs   (tiny.cc/optimiz)
      konfig/   shared Makefile + dotfiles (make help|sh|vi|...)

## DESCRIPTION

    Summarizes a CSV into Num/Sym columns in constant space; scores
    each row by distance-to-heaven over its goals; spends a small
    label budget via SWAY-style far-point projection; then grows a
    tree that minimizes a size-weighted variance (sd for numeric
    goals, entropy for symbolic) so the SAME code yields regression
    or classification trees. A leaf is a real cluster; the delta
    between two leaves' branch tests is a feasible intervention.

    Two modes: a static CSV, or a live model -- override the
    `labelled()` seam so goals are computed on demand (see dtlz.py,
    which drives ezr2 over the DTLZ1-7 benchmarks).

    Tests live in test_ezr2.py (`from ezr2 import *; main(globals())`);
    the library ezr2.py has no tests inside it.

## DATA

    First row names the columns; a name's last char sets its role,
    its first char its type:
      [A-Z]*    numeric        (e.g. "Clndrs")
      [a-z]*    symbolic       (e.g. "origin")
      *+        maximize goal  (e.g. "Mpg+")
      *-        minimize goal  (e.g. "Lbs-")
      *!        class label    (e.g. "sick!")
      *X        ignored        (e.g. "HpX")
      *~        protected x-column

    E.g. Clndrs,Volume,HpX,Model,origin,Lbs-,Acc+,Mpg+ -> numeric
    inputs Clndrs/Volume/Model, symbolic input origin, ignored HpX,
    goals minimize Lbs, maximize Acc/Mpg.

## OPTIONS

    --file   data file             = ../optimiz/misc_auto93.csv
    --seed   random seed           = 1
    --leaf   tree min leaf rows    = 3
    --maxd   tree max depth        = 8
    --grow   add labels/round      = 4
    --budget labeling cap          = 50
    --cap    max rows kept         = 1024
    --check  rows labelled by tree = 5    (at-k trust knob)
    --keepf  keep frac             = 0.66
    --round  decimals shown        = 3
    --landscape  active | random   = active

## FINDINGS

    Active beats random ~3:1 where landscapes separate, ties at the
    sparsity ceiling; the edge lives in the hard tail, not the mean
    (Random.md). `grow` is the dominant knob and SWAY's (keepf=0.5,
    grow=2) is the weakest setting in its own space -- growing a
    little faster gains ~11 win points across a smooth, low-
    sensitivity plateau (Growing.md).

## SEE ALSO

    ezr2.md       the tour -- a build-order textbook of the code
    dtlz.py       drive ezr2 with an external model (DTLZ1-7)
    tiny.cc/optimiz   sibling data gist (CSVs; never bundled here)
    tiny.cc/ezr       the original ezr (v1)

## LICENSE

    MIT (c) 2026 Tim Menzies <timm@ieee.org>.

## AUTHOR

    Tim Menzies, http://timm.fyi
