#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
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

## Example
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

"""

"""
    Copyright (c) 2020-2021 LISN CNRS
    All rights reserved.
    Pierre Zweigenbaum, LISN, CNRS, Université Paris-Saclay <pz@lisn.fr>
"""

import sys
import os
import csv
import argparse
import logging
import numpy as np

# Program version
version = '0.1'

#================
# 
#================
class Results():
    """Set of results"""

    def __init__(self, iterables):
        """Input is a list of iterables of the same length.
Each iterable represents the results obtained by a system on a series of input data items.
There are as many iterables as examined systems."""
        
        self.results = np.array(iterables)
        return

    @classmethod
    def from_csv(cls, file, delimiter="\t", quotechar='"', newline=''):
        """Create list of Results from unique CSV file."""
        res = np.loadtxt(file, dtype=int, delimiter=delimiter, unpack=True) # unpack to transpose into column vectors
        return cls(res)

    @classmethod
    def from_csvs(cls, files, encoding='utf-8', delimiter=",", quotechar='"', newline=''):
        """Create Results from list of CSV files."""
        res = []
        for file in files:
            r = []
            with open(file, "r", encoding=encoding, newline=newline) as csvfile: # read as a CSV file
                reader = csv.reader(csvfile, delimiter=delimiter, quotechar=quotechar) # with suitable delimiters and quotes
                for fields in reader:
                    r.append(fields)
            res.append(r)
        return cls(res)

class TPS(Results):
    """Set of true positive (TP) results: binary vectors"""

    # maybe do something to post-process the parent class's __init__ function
    # and create numpy binary vectors
    def TODO_binarize(self):
        self.results = np.array(self.results, dtype=int) # bool?

    def compute_bins(self):
        """Given the list of TP results for each system (self.results):
        For each input data item, computes how many systems obtained a true positive (TP) for it.
        Creates as many bins as there are systems (S) plus one (for zero).
        Bin N holds the number of data items for which exactly N systems obtained a true positive:
        bin 0 describes the set of most difficult data items, bin S+1 the easiest data items.
        We can then describe a data item by the bin that contains it: its difficulty.
        We can finally summmarize a system's result by its number of TPs (correct results) in each bin,
        and by the proportion of items in that bin for which it produced a TP (correct result)."""
        # a data item has been found as TP by how many systems
        self.tp_in = np.sum(self.results, axis=0)
        # histogram: nb_sys+1 bins, nb_sys+2 bin edges
        bin_def = range(len(self.results)+2)
        # histogram: number of data points found as TPs by N systems, for N in 0..nb_systems
        self.bins, self.bin_edges = np.histogram(self.tp_in, bins=bin_def)
        # bin of a data item: this is exactly self.tp_in
        # number of data items in the bin of item i: self.bins[self.tp_in[i]]
        # number of data items in the bin of each item: self.bins[self.tp_in]
        # number of data items in the bin of each item for a system sys:
        self.nps_per_bin = np.array([ np.histogram(self.tp_in[sys==1], bins=bin_def)[0]
                                      for sys in self.results ])
        # todo: remove nan's
        self.prop_nps_per_bin = np.nan_to_num(self.nps_per_bin / self.bins)

        # how many TPs in each bin for each system:
        # sum([self.tp_in[sys[i]] for i in range(len())])
        return

    def plot_bins(self):
        plt.plot(.5*(self.bin_edges[1:]+self.bin_edges[:-1]), self.bins)
        plt.show
        return

if __name__ == '__main__':
    def parse_execute_command_line():
        parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                         formatter_class=argparse.RawDescriptionHelpFormatter,
                                         description=__doc__)

        parser.add_argument("systems", help="file containing binary system results (e.g., true positives for each gold positive)")

        groupO = parser.add_argument_group('Options')
        # groupO.add_argument("--ignore-extra-fields", action="store_true", help="Fields beyond NUMFIELDS will be silently ignored: the line will be truncated and processed.  Default is to log an error and skip the line.")

        groupA = parser.add_argument_group('Action')
        groupA.add_argument("--compute-bins", action="store_true", help="Compute bins.")
        groupA.add_argument("--plot-bins", action="store_true", help="Compute and plot bins.")

        groupS = parser.add_argument_group('Special')
        groupS.add_argument("-q", "--quiet", action="store_true", help="suppress reporting progress info.")
        groupS.add_argument("--debug", action="store_true", help="print debug info.")
        groupS.add_argument("-v", "--version", action="version", version='%(prog)s ' + version, help="print program version.")

        args = parser.parse_args()

        FORMAT = '%(levelname)s: %(message)s'
        logging.basicConfig(format=FORMAT)

        logger = logging.getLogger()
        if not args.quiet:
            logger.setLevel(logging.INFO)
        if args.debug:
            logger.setLevel(logging.DEBUG)

        r = TPS.from_csv(args.systems)

        if args.compute_bins:
            logging.info("Computing bins of gold results: bin N holds the number of data items for which exactly N systems obtained a true positive")
            r.compute_bins()
            # print(r.results)
            print(r.bins)
            print(r.nps_per_bin)
            print(np.round(r.prop_nps_per_bin, decimals=2)*100)
            print(np.sum(r.results, axis=1))

    parse_execute_command_line()
