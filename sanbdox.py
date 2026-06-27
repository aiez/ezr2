"""

Here i implement and explore

Grouped by the only axis that matters for your budget: does the knob re-use one labeled batch (free) or demand fresh oracle calls (paid). One line each.

**Free — change tree shape, re-use the same ≤50 labels**

1. Exit-collapse / sub-tree → single exit. The FFT 2^(d-1) family; correlated diversity. FFT origin Martignon & Gigerenzer; SE application Menzies group.
2. Bagging — build each tree on an 80% resample of the 50. Breiman 1996.
3. Random subspace — split on a random subset of x (mtry). Ho 1998; placement at tip vs every level is your choice. Breiman 2001 (Random Forests) combines 2+3.
4. Extra-Trees — pick the cut *threshold* at random, no optimization. Geurts, Ernst & Wehenkel 2006. Max throughput.
5. Mondrian trees — whole skeleton drawn from a process over x only; labels touch leaves alone. Lakshminarayanan, Roy & Teh 2014. The frugal extreme.
6. Oblique splits — split on a random linear combination of x. OC1: Murthy, Kasif & Salzberg 1994.
7. Rotation Forest — PCA-rotate feature subsets before splitting. Rodríguez, Kuncheva & Alonso 2006.
8. Split criterion swap — variance-reduction (CART; Breiman et al. 1984) vs MAE vs Friedman-MSE (Friedman 2001) vs your m2 vs entropy-of-binned-y.
9. Supervised bin edges as candidate cuts — MDL discretization. Fayyad & Irani 1993.
10. Leaf readout — mean / median / kNN / kernel-weighted local regression (LOESS; Cleveland 1979).
11. Model-tree leaves — fit a linear model per leaf instead of a constant. M5: Quinlan 1992; M5′: Wang & Witten 1997.
12. Quantile-regression leaves — fit a leaf quantile, not the mean. Meinshausen 2006 (Quantile Regression Forests).
13. Leaf shrinkage — smooth each leaf toward its parent. m-estimate: Cestnik 1990; hierarchical shrinkage: Agarwal et al. 2022.
14. Monotonic constraints — force splits to respect goal direction. Potharst & Feelders 2002.
15. Scalarization (multi-goal) — random-weight or Chebyshev aggregation of your y's; re-weighting is free since all y are labeled.

**Threshold knobs — slide *along* ROC from the same trees**

16. Target rebinning — sweep the percentile that cuts y into L/H; each percentile = a different ROC operating point. The regression analog of cost-sweeping. This is what scatters your 7 FFTs along the curve.
17. Leaf-mean decision threshold — sweep the predict-High cutoff on the leaf value. Same effect from the readout side.

**Paid — the only label-spending knobs**

18. Fresh skeleton — a new 50 rows; the sole source of *independent* (different-cut) diversity.
19. Which 50 — active acquisition (your landscape; marginal value ≈ 0 on these tasks → treat as dead).

**Ancestors to cite, not knobs**: Holte 1993 (1R, weak-rule-suffices), Schapire 1990 (weak learnability), Provost & Fawcett 2001 (ROC convex hull as the object).

Three I'd flag as uncertain on exact citation: the specific FFT-for-SE paper title, M5′ year (1997), and Potharst & Feelders year — verify those against your own library before they go in a bib. Want this as a `.md` table with a "label cost" column you can drop into the paper?

"""
