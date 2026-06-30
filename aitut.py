#!/usr/bin/env python3 -B
"""
aitut: a second tour. Six classic AI tools, built on the ezr2 substrate.
(c) 2026, Tim Menzies <timm@ieee.org>, MIT license

USAGE: python3 test_aitut.py [--key=val ...] [test ...]

This file `from ezr2 import *` and carries on: it adds Naive Bayes,
k-means, k-means++, simulated annealing, local search, differential
evolution, and a Bayesian best/rest active learner -- each one a thin
loop over the four ezr2 classes (Num, Sym, Data) and one new sampler
(`pick`, promoted into ezr2.py). Worked examples from the EZR.py paper
("Can AI be Easy?"); the text-mining results are deliberately omitted.

OPTIONS: (extra knobs merged into ezr2's `the`):
  --start   active warm-start labels    = 4
  --known   surrogate known set         = 50
  --evals   optimizer oracle calls      = 1000
  --m       SA fraction of attrs mutated= 0.5
  --p       LS prob of sweeping one attr= 0.5
  --tries   LS sweep candidates         = 20
  --restart LS no-gain reset threshold  = 100
  --np      DE population size           = 20
  --gens    DE generations               = 20
  --f       DE scale factor              = 0.5
  --cr      DE crossover prob            = 0.3
  --bk      bayes smoothing k            = 1
  -h        print this help

TESTS: (run with their bare name):
  bayes     Naive Bayes good/bad classifier accuracy
  kmeans    k-means clusters, sizes and mean disty
  kpp       k-means++ seeding spread
  opt       sa vs ls vs de: best disty found via a surrogate oracle
  active    Bayesian best/rest active learner vs ezr2 landscape
  all       run every test above, reseting the seed each time
"""
from ezr2 import *
from math import pi, log

# merge this file's extra OPTIONS into ezr2's shared `the`
for _k, _v in vars(settings(__doc__)).items(): setattr(the, _k, _v)

#-- Naive Bayes -------------------------------------------------
def like(col, v, prior):
  "P(v | col): Gaussian for a Num, Laplace-smoothed freq for a Sym."
  if v == "?": return 1
  if is_sym(col):
    return (col.get(v,0) + the.bk*prior) / (sum(col.values()) + the.bk)
  mu, s = mu_(col), sd(col)
  z = 2*s*s + TINY
  return exp(-(v - mu)**2 / z) / (z*pi)**0.5

def likes(data, row, nall, nh):
  "Log-likelihood that row belongs to data, summed over the x-columns."
  prior = (len(data.rows) + the.bk) / (nall + the.bk*nh)
  out = log(prior + TINY)
  for at in data.x:
    if (l := like(data.cols[at], row[at], prior)) > 0: out += log(l)
  return out

#-- Clustering: k-means and k-means++ ---------------------------
# (centroids come from ezr2's cached `mids`, invalidated on every add)

def kmeans(data, k=8, loops=10):
  "Lloyd's k-means over the cheap x-distance; returns the non-empty clusters."
  cents = [r[:] for r in some(data.rows, k)]
  out   = []
  for _ in range(loops):
    out = [clone(data, []) for _ in cents]
    for r in data.rows:
      j = min(range(len(cents)), key=lambda j: distx(data, cents[j], r))
      add(out[j], r)
    cents = [mids(c) for c in out if c.rows]
  return [c for c in out if c.rows]

def kpp(data, k=8, few=256):
  "k-means++ seeding: each new centre is picked far from the rest (D^2 weighted)."
  out = [random.choice(data.rows)]
  while len(out) < k:
    t  = some(data.rows, few)
    ws = {i: min(distx(data, t[i], c) for c in out)**2 for i in range(len(t))}
    out.append(t[pick(ws)])      # pick = roulette over the squared-distance dict
  return out

#-- Surrogate oracle (for synthetic candidates) -----------------
def oracle(data, known, row):
  "Label a synthetic row: copy y from its nearest known row, then score disty."
  near = min(known, key=lambda r: distx(data, row, r))
  for at in data.y: row[at] = near[at]
  return disty(data, row)

#-- Optimizers: (1+1) annealing & local search ------------------
def picks(data, row, n):
  "A copy of row with n random x-attributes re-sampled via pick."
  s = row[:]
  for at in some(data.x, n): s[at] = pick(data.cols[at], s[at])
  return s

def oneplus1(data, known, mutate, accept, evals, restart=0):
  "Generic (1+1) loop. restart>0: reset to the start after that many no-gain steps."
  now0 = random.choice(data.rows)[:]
  ne0  = oracle(data, known, now0)
  now, ne, best, beste, h, imp = now0, ne0, now0, ne0, 0, 0
  while h < evals:
    for kid in mutate(now):
      e = oracle(data, known, kid); h += 1
      if accept(e, ne, h, evals): now, ne = kid, e
      if e < beste: best, beste, imp = kid, e, h     # imp = step of last gain
      if restart and h - imp > restart:              # stuck too long: reset-retry
        now, ne, imp = now0[:], ne0, h               # back to the initial solution
        break
      if h >= evals: break
  return best, beste

def sa(data, known=None, evals=None):
  "Simulated annealing: mutate a random m-fraction of attributes; cooling accept."
  known = known or some(data.rows, the.known); evals = evals or the.evals
  n = max(1, int(the.m * len(data.x)))
  def mutate(row): yield picks(data, row, n)
  def accept(en, e, h, b):
    if en <= e: return True                  # downhill: always
    t = 1 - h/(b + 1)                         # temperature cools to 0
    return t > 0 and random.random() < exp((e - en)/(t + TINY))
  return oneplus1(data, known, mutate, accept, evals)

def ls(data, known=None, evals=None):
  "Local search: greedy; mutate one attribute, and p% of the time sweep its range."
  known = known or some(data.rows, the.known); evals = evals or the.evals
  def mutate(row):
    at = random.choice(data.x)               # freeze all but this one attribute
    for _ in range(the.tries if random.random() < the.p else 1):
      s = row[:]; s[at] = pick(data.cols[at], s[at]); yield s    # sweep its range
  return oneplus1(data, known, mutate, lambda en,e,h,b: en <= e, evals, the.restart)

#-- Optimizer: differential evolution ---------------------------
def de(data, known=None):
  "DE/rand/1/bin over the x-columns: kid = a + f*(b-c), oracle scored."
  known = known or some(data.rows, the.known)
  pop = [r[:] for r in some(data.rows, the.np)]
  fit = [oracle(data, known, r) for r in pop]
  for _ in range(the.gens):
    for i in range(len(pop)):
      a, b, c = some(pop, 3)               # 3 peers, sampled straight from pop
      kid, keep = pop[i][:], random.choice(data.x)   # keep one parent attribute
      for at in data.x:
        if at != keep and random.random() < the.cr:
          col = data.cols[at]
          kid[at] = pick(col) if is_sym(col) else a[at] + the.f*(b[at]-c[at])
      if (e := oracle(data, known, kid)) < fit[i]: pop[i], fit[i] = kid, e
  i = min(range(len(pop)), key=lambda j: fit[j])
  return pop[i], fit[i]

#-- Active learning: best/rest acquisition ----------------------
# An acquisition is a factory (data,best,rest,n) -> (row -> score), higher = label-me-next.
def bayes(data, best, rest, n):
  "Acquire by belief: how much more best-like than rest-like a row looks."
  return lambda r: likes(best, r, n, 2) - likes(rest, r, n, 2)

def centroid(data, best, rest, n):
  "Acquire by distance: near the best centroid and far from the rest centroid."
  bm, rm = mids(best), mids(rest)            # cached centroids (see ezr2.mids)
  return lambda r: distx(data, r, rm) - distx(data, r, bm)

def rebalance(best, rest, y):
  "Keep best at the top sqrt(N) rows; demote the worst into rest (add -1)."
  n = len(best.rows) + len(rest.rows)
  while len(best.rows) > int(n**0.5 + 0.5):
    worst = max(best.rows, key=y)
    add(best, worst, -1)                       # subtract worst out of best
    add(rest, worst)                           # and fold it into rest

def actives(data, acquire=bayes):
  "Warm-start, then each round label the row `acquire` most wants (bayes|centroid)."
  y    = lambda r: disty(data, r)
  pool = shuffle(data.rows)
  lab  = sorted(pool[:the.start], key=y)       # warm start: a few random labels
  pool = pool[the.start:]
  cut  = max(1, int(len(lab)**0.5 + 0.5))
  best = clone(data, lab[:cut])                # top sqrt(n) so far ...
  rest = clone(data, lab[cut:])                # ... rest is everything else
  while len(best.rows) + len(rest.rows) < the.budget and pool:
    n     = len(best.rows) + len(rest.rows)
    guess = acquire(data, best, rest, n)
    row, *pool = sorted(pool, key=guess, reverse=True)   # most-wanted first, peel it off
    add(best, row)                             # pay one label -> best
    rebalance(best, rest, y)                   # then demote, incrementally
  return sorted(best.rows + rest.rows, key=y)
