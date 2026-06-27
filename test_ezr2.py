#!/usr/bin/env python3 -B
"""
test_ezr2.py: the test / CLI lane for ezr2 (library lives in ezr2.py).

Run any test by its bare name; pass --key=val to override a knob:
  python3 test_ezr2.py --file=../optimiz/misc_auto93.csv tree
  python3 test_ezr2.py all

TESTS:
  disty       rows by disty: top 5 / bottom 5
  landscape   20 shuffles; active vs random, ranked by delta
  landscapes  one mean-win line (the sweep)
  tree        build+show a tree on acquired rows
  trees       random-trained vs landscape-trained tree
  holdout     50:50 split; tree picks best test row
  holdouts    holdout x20; land vs random verdict
  pure        no tree: best labelled, land vs random
  same        demo+validate the same() stat test
  all         run every test above, reseting seed each
"""
from ezr2 import *

def test_disty():
  "Rows sorted by disty: header, top 5, blank, bottom 5."
  data = Data(csv(the.file))
  rows = sorted(data.rows, key=lambda r: disty(data, r))
  hdr  = list(data.names) + ["disty"]
  fmt  = lambda r: [str(v) for v in r]+["%.3f" % disty(data,r)]
  body = [fmt(r) for r in rows[:5] + rows[-5:]]
  w = [max(len(row[c]) for row in [hdr]+body)
       for c in range(len(hdr))]
  line = lambda cs: print("  ".join(c.rjust(w[i])
                                    for i,c in enumerate(cs)))
  line(hdr)
  for r in body[:5]: line(r)
  print()
  for r in body[5:]: line(r)

def test_landscape():
  "20 shuffles; per-run active vs random landscape, sorted by significant delta."
  the.file = "../optimiz/binary_config_billing10k.csv"  # active reliably beats random
  data = Data(csv(the.file))
  data.rows = some(data.rows, the.cap)
  A, R = [], []
  for i in range(20):
    random.seed(the.seed + i); data.rows = shuffle(data.rows)
    the.landscape = "active"; A += [disty(data, landscape(data)[0])]
    the.landscape = "random"; R += [disty(data, landscape(data)[0])]
  the.landscape = "active"
  sd = lambda z: (sum((v-sum(z)/len(z))**2 for v in z)/(len(z)-1))**0.5
  pooled = (((len(A)-1)*sd(A)**2 + (len(R)-1)*sd(R)**2)/(len(A)+len(R)-2))**0.5
  thr = 0.35 * pooled       # call it a tie below a small effect size; else +ve => active wins
  rows_out = [(a, r, (r-a if abs(r-a) >= thr else 0.0)) for a,r in zip(A, R)]
  up = chr(0x25B2)          # marks whichever side won this run
  print("rank  aDisty  rDisty   delta  win  (%s)" % the.file.split("/")[-1])
  for k,(a,r,d) in enumerate(sorted(rows_out, key=lambda t: -t[2])):
    win = "tie" if d==0 else ("%s active" % up if d>0 else "%s random" % up)
    print("%4d %7.3f %7.3f  %+6.3f  %s" % (k, a, r, d, win))
  win  = sum(d for _,_,d in rows_out if d > 0)
  loss = -sum(d for _,_,d in rows_out if d < 0)
  assert win > loss         # the size of the wins beats the size of the losses

def test_landscapes():
  "One summary line: mean win/disty over 20 runs."
  data = Data(csv(the.file))
  data.rows = some(data.rows, the.cap)
  W, ds, ws, n = wins(data), [], [], 0
  for i in range(20):
    random.seed(the.seed + i)
    data.rows = shuffle(data.rows)
    got = landscape(data)
    ds += [disty(data,got[0])]; ws += [W(got[0])]; n = len(got)
  print("%6.1f %7.3f %4d  %s" % (sum(ws)/len(ws),
        sum(ds)/len(ds), n, the.file.split("/")[-1]))

def test_tree():
  "Build a tree over landscape's rows and print it."
  random.seed(the.seed)
  data = Data(csv(the.file))
  data.rows = some(data.rows, the.cap)
  show(data, tree(data, landscape(data)))

def test_trees():
  "Same budget: random-trained vs landscape-trained tree."
  random.seed(the.seed)
  data = Data(csv(the.file))
  data.rows = some(data.rows, the.cap)
  land = landscape(data)
  rand = some(data.rows, len(land))
  W = wins(data)
  for tag, rows in [("random", rand), ("landscape", land)]:
    best = min(rows, key=lambda r: disty(data,r))
    print("\n== %s  n=%d  best disty=%.3f  win=%.1f ==" %
          (tag, len(rows), disty(data,best), W(best)))
    show(data, tree(data, rows))

def vs(data, pick):
  "active vs random over 20 runs of pick(); stat verdict line."
  W, out = wins(data), {}
  for mode in ("active", "random"):
    the.landscape = mode; out[mode] = []
    for i in range(20):
      random.seed(the.seed + i); out[mode] += [W(pick(data))]
  the.landscape = "active"
  L, R = out["active"], out["random"]
  ml, mr = sum(L)/20, sum(R)/20
  v = "tie" if same(L, R) else ("land" if ml > mr else "rand")
  print("%6.1f %6.1f %-5s %s" % (ml, mr, v,
        the.file.split("/")[-1]))

def test_holdout():
  "One run: the holdout-picked best row's disty and win."
  random.seed(the.seed)
  data = Data(csv(the.file))
  data.rows = some(data.rows, the.cap)
  b = holdout(data)
  print("best disty %.3f  win %.1f  (%s)" % (disty(data,b),
        wins(data)(b), the.file.split("/")[-1]))

def test_holdouts():
  "active vs random landscape, through the holdout pipeline."
  data = Data(csv(the.file))
  data.rows = some(data.rows, the.cap)
  vs(data, holdout)

def test_pure():
  "active vs random landscape; best labelled row, no tree."
  data = Data(csv(the.file))
  data.rows = some(data.rows, the.cap)
  vs(data, lambda d: landscape(d)[0])

def test_same():
  "Validate same(): small shift = same, big shift = differ."
  random.seed(the.seed)
  a = [random.gauss(0, 1) for _ in range(20)]
  shift = lambda d: [x + d for x in a]
  print("shift  same   cliffs cohen")
  for d in (0, 0.1, 0.3, 0.5, 1.0, 2.0):
    b = shift(d)
    print(" %+.1f  %-5s  %.2f   %s" % (d, same(a,b),
          cliffs(a,b), cohen(a,b)))
  assert same(a, a) and not same(a, shift(2))

def test_all():
  "Run every other test_*, reseting the seed before each."
  for n,f in list(globals().items()):
    if n.startswith("test_") and n != "test_all":
      print("\n#", n, "-"*40)
      try: random.seed(the.seed); f()
      except Exception as e: print("FAIL:", n, type(e).__name__, e)

if __name__ == "__main__": main(globals())
