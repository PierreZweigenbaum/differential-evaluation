# Differential evaluation

Compares a set of system results

Input = one row per gold standard item, one column per system:
in each cell, 1 if system obtained a true positive (TP) for this item, 0 otherwise.

For each input data item, computes how many systems obtained a TP for it.
Creates as many bins as there are systems (S) plus one (for zero).

  - bin N holds the number of data items for which exactly N systems obtained a TP:
  - bin 0 describes the set of most difficult data items, bin S+1 the easiest data items.

We can then describe a data item by the bin that contains it: its difficulty.
We can finally summmarize a system's result by its number of TPs in each bin,
or by the proportion of items in that bin for which it produced a TP.

## Example:
    # create random system results drawn from a binomial distribution:
    n_sys = 10
    n_data = 100
    proba_tp = 0.8
    rng = np.random.default_rng()
    data = rng.binomial(1, proba_tp, (n_sys, n_data))

    # create bins and system statistics per bin
    r = TPS(data)
    r.compute_bins()
    print(r.results)
    print(r.bins)
    print(r.nps_per_bin)
    print(np.round(r.prop_nps_per_bin, decimals=2)*100)
    print(np.sum(r.results, axis=1))
