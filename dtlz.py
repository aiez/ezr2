#!/usr/bin/env python3 -B
"""
dtlz.py: drive ezr2 with an EXTERNAL MODEL instead of a CSV.

The DTLZ1-7 benchmarks are live models: a row's x-values are decision
variables; its goals are computed on demand by overriding ezr2.labelled.
This is how an outside user plugs their own (expensive) model into ezr2.

  python3 dtlz.py                 # default model: dtlz2
  python3 dtlz.py --model=dtlz4   # any of dtlz1..dtlz7
  python3 dtlz.py --model=dtlz7 --M=3 --N=8
"""
import sys, random, ezr2
from math import cos, sin, pi

# -- the models --------------------------------------------------
# Each maps x in [0,1]^N -> M objectives to MINIMIZE. k = N-M+1 of the
# x's form the "distance" group xm; the rest shape the front.
def _g1(xm):  return 100*(len(xm) + sum((v-.5)**2 - cos(20*pi*(v-.5)) for v in xm))
def _g2(xm):  return sum((v-.5)**2 for v in xm)
def _g6(xm):  return sum(v**0.1 for v in xm)

def _sphere(M, g, th):             # cos/sin product shared by dtlz2-6
  f = []
  for i in range(M):
    v = 1 + g
    for j in range(M-1-i): v *= cos(th[j])
    if i > 0: v *= sin(th[M-1-i])
    f += [v]
  return f

def dtlz1(x, M):                   # linear front (sum fi = 0.5)
  g, f = _g1(x[M-1:]), []
  for i in range(M):
    v = 0.5*(1+g)
    for j in range(M-1-i): v *= x[j]
    if i > 0: v *= (1 - x[M-1-i])
    f += [v]
  return f

def dtlz2(x, M): return _sphere(M, _g2(x[M-1:]), [v*pi/2 for v in x[:M-1]])
def dtlz3(x, M): return _sphere(M, _g1(x[M-1:]), [v*pi/2 for v in x[:M-1]])     # multi-modal
def dtlz4(x, M): return _sphere(M, _g2(x[M-1:]), [v**100*pi/2 for v in x[:M-1]]) # biased

def _degen(x, M, g):               # dtlz5/6 theta remap -> degenerate curve
  th = [x[0]*pi/2] + [pi/(4*(1+g))*(1+2*g*x[i]) for i in range(1, M-1)]
  return _sphere(M, g, th)
def dtlz5(x, M): return _degen(x, M, _g2(x[M-1:]))
def dtlz6(x, M): return _degen(x, M, _g6(x[M-1:]))

def dtlz7(x, M):                   # disconnected front
  k = len(x) - M + 1
  g = 1 + 9/k * sum(x[M-1:])
  f = list(x[:M-1])
  h = M - sum((fi/(1+g))*(1+sin(3*pi*fi)) for fi in f)
  return f + [(1+g)*h]

MODELS = {k: v for k, v in globals().items() if k.startswith("dtlz")}

# -- CLI knobs (model, M objectives, N decision vars) ------------
arg = lambda k, d: next((ezr2.thing(a.split("=")[1]) for a in sys.argv
                         if a.startswith("--"+k+"=")), d)
NAME = arg("model", "dtlz2"); MODEL = MODELS[NAME]
M    = arg("M", 2)
N    = arg("N", 6)                 # >4 decision vars by default
names = [f"X{i+1}" for i in range(N)] + [f"F{m+1}-" for m in range(M)]

# -- the unlabelled pool + the label seam ------------------------
def fresh_pool(n=1000):
  return [[random.random() for _ in range(N)] + ["?"]*M for _ in range(n)]

def labelled(row):
  "ezr2's seam: goals come from the model, folded into data.cols so"
  "disty can normalize objectives as labels arrive."
  if "?" in row[N:]:
    row[N:] = MODEL(row[:N], M)
    for at in data.y: data.cols[at] = ezr2.add(data.cols[at], row[at])
  return row
ezr2.labelled = labelled            # labelled() reads the global `data`

def instance(row):
  print("  x  " + " ".join("%.2f" % v for v in row[:N]))
  print("  f  " + " ".join("%.3f" % v for v in row[N:]) +
        "   (disty %.3f, lower=better)" % ezr2.disty(data, row))

print("model %s   N=%d x-vars   M=%d objectives" % (NAME, N, M))
ezr2.the.budget = 30

# (1) pure: landscape ranks the whole pool, no train/test split
random.seed(1); data = ezr2.Data([names] + fresh_pool())
got = ezr2.landscape(data)
print("\nthe best option found (one instance):")
instance(got[0])

# (2) explanatory model: which x-ranges reach good goals
print("\nwhy? an explanatory model -- which x-ranges reach good goals:")
ezr2.show(data, ezr2.tree(data, got))

# (3) test the model on new data: train/test split via holdout
random.seed(1); data = ezr2.Data([names] + fresh_pool())
print("\ndoes that model generalize? best pick on unseen test data:")
instance(ezr2.holdout(data))
