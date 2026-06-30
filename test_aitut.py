#!/usr/bin/env python3 -B
"""
test_aitut.py: the test / CLI lane for aitut (library lives in aitut.py,
which itself rides on ezr2.py). Same contract as test_ezr2.py.

Run any test by its bare name; pass --key=val to override a knob:
  python3 test_aitut.py --file=../optimiz/misc_auto93.csv opt
  python3 test_aitut.py all

TESTS:
  bayes     Naive Bayes good/bad classifier accuracy
  kmeans    k-means clusters, sizes and mean disty
  kpp       k-means++ seeding spread
  opt       sa vs ls vs de: median win over 20 shuffles (+variance)
  active    bayes vs centroid acquisition vs landscape vs random
  all       run every test above, reseting the seed each time
"""
from aitut import *

def _data():
  "Load the file, cap the rows (same prep every test uses)."
  data = Data(csv(the.file)); data.rows = some(data.rows, the.cap)
  return data

def test_bayes():
  "Train NB on half the rows (good=below-median disty), score the other half."
  random.seed(the.seed)
  data = _data(); rows = shuffle(data.rows); h = len(rows)//2
  train, test = rows[:h], rows[h:]
  ys   = sorted(disty(data, r) for r in train); md = ys[len(ys)//2]
  good = lambda r: disty(data, r) <= md
  G    = clone(data, [r for r in train if good(r)])
  B    = clone(data, [r for r in train if not good(r)])
  n    = len(train)
  pred = lambda r: likes(G, r, n, 2) >= likes(B, r, n, 2)
  acc  = sum(pred(r) == good(r) for r in test) / len(test)
  print("bayes good/bad accuracy %.2f  (%s)" % (acc, the.file.split("/")[-1]))
  assert acc > 0.6

def test_kmeans():
  "Cluster with k-means; print each cluster's size and mean disty, best first."
  random.seed(the.seed)
  data = _data(); cl = kmeans(data, k=8)
  md   = lambda c: sum(disty(data, r) for r in c.rows) / len(c.rows)
  print("kmeans: %d clusters  (%s)" % (len(cl), the.file.split("/")[-1]))
  for c in sorted(cl, key=md):
    print("  n=%3d  mean disty=%.3f" % (len(c.rows), md(c)))

def test_kpp():
  "k-means++ seeding: report seed count and mean pairwise x-spread."
  random.seed(the.seed)
  data  = _data(); cents = kpp(data, k=8)
  spread= sum(distx(data, a, b) for a in cents for b in cents) / len(cents)**2
  print("kpp: %d seeds  mean spread=%.3f  (%s)"
        % (len(cents), spread, the.file.split("/")[-1]))

def bench(data, methods, n=20):
  "Median win over n shuffles per method; same() verdict vs the first method."
  W, out = wins(data), {}
  for name, pick in methods:                   # each pick(data) -> one row
    ws = []
    for i in range(n):
      random.seed(the.seed + i); data.rows = shuffle(data.rows)
      ws.append(W(pick(data)))
    out[name] = ws
  base = next(iter(out)); med = lambda z: sorted(z)[len(z)//2]
  print("method          medWin   sd   vs:%s  (%s)"
        % (base, the.file.split("/")[-1]))
  for name, ws in out.items():
    s = (sum((v-sum(ws)/len(ws))**2 for v in ws)/(len(ws)-1))**0.5
    v = "--" if name == base else (
        "tie" if same(out[base], ws) else
        "better" if med(ws) > med(out[base]) else "worse")
    print("%-14s %6.1f %5.1f   %s" % (name, med(ws), s, v))
  return out

def test_opt():
  "sa vs ls vs de over 20 shuffles: median win + variance via the oracle."
  bench(_data(), [("sa", lambda d: sa(d)[0]),
                  ("ls", lambda d: ls(d)[0]),
                  ("de", lambda d: de(d)[0])], n=10)

def _land(mode):
  "landscape best row in the given mode, restoring the default after."
  def f(d):
    the.landscape = mode; r = landscape(d)[0]; the.landscape = "active"; return r
  return f

def test_active():
  "Two acquisitions (bayes, centroid) vs ezr2 landscape vs random, 20 shuffles."
  data = bench(_data(), [
    ("active.bayes", lambda d: actives(d, bayes)[0]),
    ("active.cent",  lambda d: actives(d, centroid)[0]),
    ("landscape",    _land("active")),
    ("random",       _land("random"))])
  assert sorted(data["active.bayes"])[len(data["active.bayes"])//2] >= 50

def test_all():
  "Run every other test_*, reseting the seed before each."
  for n, f in list(globals().items()):
    if n.startswith("test_") and n != "test_all":
      print("\n#", n, "-"*40)
      try: random.seed(the.seed); f()
      except Exception as e: print("FAIL:", n, type(e).__name__, e)

if __name__ == "__main__": main(globals())
