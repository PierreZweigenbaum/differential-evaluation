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

Further services:
Compare two systems according to an already defined set of bins

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

## Command-line example
    python results.py --system-names A1,A2,B1,B2,C1,C2,D,E1,E2 --plot-bins plot.pdf true-positives-per-system.tab.gz

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
import re
import logging
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Program version
version = '0.2'

#================
# Base class for set of results
#================
class Results():
    """Set of results"""

    def __init__(self, iterables, inputs=None, names=None):
        """Argument is a list of iterables of the same length.
Each iterable represents the results obtained by a system on a series of input data items.
There are as many iterables as examined systems.
inputs: list of input data items (or some proxies)
names: list of names of the systems or models or experiments
"""
        
        self.results = np.array(iterables)
        self.inputs = inputs
        self.names = names
        return

    @classmethod
    def from_csv(cls, file, delimiter="\t", quotechar='"', newline=''):
        """Create list of Results from unique CSV file."""
        res = np.loadtxt(file, dtype=int, delimiter=delimiter, unpack=True) # unpack to transpose into column vectors
        return cls(res)

    @classmethod
    def from_csv_with_inputs(cls, file, delimiter="\t", quotechar='"', newline=''):
        """Create list of Results from unique CSV file with input string as first column."""
        df = pd.read_csv(file, sep=delimiter, quotechar=quotechar)
        inputs = list(df["input"]) # inputs only
        names = list(df.columns)[1:]
        res = df.iloc[:, 1:].to_numpy().transpose() # values only; transpose as in from_csv
        return cls(res, inputs=inputs, names=names)

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
        self.tps_per_item = np.sum(self.results, axis=0)
        # histogram: nb_sys+1 bins, nb_sys+2 bin edges
        bin_def = range(len(self.results)+2)
        # histogram: number of data points found as TPs by N systems, for N in 0..nb_systems
        self.bins, self.bin_edges = np.histogram(self.tps_per_item, bins=bin_def)
        # bin of a data item: this is exactly self.tps_per_item
        # number of data items in the bin of item i: self.bins[self.tps_per_item[i]]
        # number of data items in the bin of each item: self.bins[self.tps_per_item]
        # number of data items in the bin of each item for a system sys:
        self.nps_per_bin = np.array([ np.histogram(self.tps_per_item[sys==1], bins=bin_def)[0]
                                      for sys in self.results ])
        # remove nan's
        self.prop_nps_per_bin = np.nan_to_num(self.nps_per_bin / self.bins)

        # Binary vector with 1 if data point is in bin k
        self.bin_indicators = [self.tps_per_item == k for k in range(len(self.results)+1)]
        # how many TPs in each bin for each system:
        # sum([self.tps_per_item[sys[i]] for i in range(len())])
        return

    def plot_bin_size(self):
        plt.plot(.5*(self.bin_edges[1:]+self.bin_edges[:-1]), self.bins)
        plt.show
        return

    def plot_bins(self, file=None, names=None, absolute=False):
        systems = (names if names is not None
                   else [chr(i) for i in range(ord('A'), ord('A')+len(self.results))])
        # remove file name details if they follow the following pattern:
        systems = [ re.sub("/[0-9][^/]+/test_predictions.tsv$", "", s, count=0, flags=0) for s in systems ]
        bins = list(range(len(systems)+1))
        sysbins = np.round(self.prop_nps_per_bin, decimals=2)

        fig, ax = plt.subplots()
        im = ax.imshow(sysbins)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(bins)))
        ax.set_yticks(np.arange(len(systems)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(bins)
        ax.set_yticklabels(systems)
        ax.set_ylim(len(systems)-0.5,-0.5) # show first and last systems with full row height

        # Rotate the tick labels and set their alignment.
        # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.setp(ax.get_xticklabels(), fontsize=4)
        plt.setp(ax.get_yticklabels(), fontsize=4)
        # plt.subplots_adjust(top=0.88, bottom=0.11, left=0.44, right=0.9, hspace=0.2, wspace=0.2)

        # Loop over data dimensions and create text annotations.
        for i in range(len(systems)):
            for j in range(len(bins)):
                val = (self.nps_per_bin[i, j] if absolute else int(sysbins[i, j]*100))
                text = ax.text(j, i, val,
                               ha="center", va="center",
                               color=("b" if val > 50 else "w"),
                               fontsize=(4 if absolute else 5),
                )

        ax.set_title("Performance ({}) of each system on each bin".format(("#TPs" if absolute else "%TPs")), fontsize=5)
        fig.tight_layout()
        plt.show()
        if file is not None:
            fig.savefig(file)
        return

    def plot_bins_v2(self, file=None, names=None, absolute=False):
        systems = (names if names is not None
                   else [chr(i) for i in range(ord('A'), ord('A')+len(self.results))])
        # remove file name details if they follow the following pattern:
        systems = [ re.sub("/[0-9][^/]+/test_predictions.tsv$", "", s, count=0, flags=0) for s in systems ]
        bins = list(range(len(systems)+1))
        sysbins = np.round(self.prop_nps_per_bin, decimals=2)
        title = "Performance ({}) of each system on each bin".format(("#TPs" if absolute else "%TPs"))
        self.plot_bins_internal(file=file, systems=systems, bins=bins, sysbins=sysbins, title=title, absolute=absolute)
        return

    def plot_bins_internal(self, file=None, systems=None, bins=None, sysbins=None, title=None, absolute=False):
        fig, ax = plt.subplots()
        im = ax.imshow(sysbins)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(bins)))
        ax.set_yticks(np.arange(len(systems)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(bins)
        ax.set_yticklabels(systems)
        ax.set_ylim(len(systems)-0.5,-0.5) # show first and last systems with full row height

        # Rotate the tick labels and set their alignment.
        # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.setp(ax.get_xticklabels(), fontsize=4)
        plt.setp(ax.get_yticklabels(), fontsize=4)
        # plt.subplots_adjust(top=0.88, bottom=0.11, left=0.44, right=0.9, hspace=0.2, wspace=0.2)

        # Loop over data dimensions and create text annotations.
        for i in range(len(systems)):
            for j in range(len(bins)):
                val = sysbins[i, j]
                text = ax.text(j, i, val,
                               ha="center", va="center",
                               color=("b" if val > 50 else "w"),
                               fontsize=(4 if absolute else 5),
                )

        ax.set_title(title, fontsize=5)
        fig.tight_layout()
        plt.show()
        if file is not None:
            fig.savefig(file)
        return

if __name__ == '__main__':
    def parse_execute_command_line():
        parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                         formatter_class=argparse.RawDescriptionHelpFormatter,
                                         description=__doc__)

        parser.add_argument("systems", help="file containing binary system results (e.g., true positives for each gold positive)")

        groupO = parser.add_argument_group('Options')
        groupO.add_argument("--with-inputs", action="store_true", help="The input file includes a first column with system inputs.")
        groupO.add_argument("--displayed-values", choices=["percentage", "number"], default="percentage", help="How to display values in the produced table.")

        groupA = parser.add_argument_group('Action')
        groupA.add_argument("--compute-bins", action="store_true", help="Compute bins.")
        groupA.add_argument("--plot-bins", help="Compute bins and plot them to the specified file.")
        groupA.add_argument("--system-names", help="Comma-separated list of system names.")

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

        if args.with_inputs:
            r = TPS.from_csv_with_inputs(args.systems)
        else:
            r = TPS.from_csv(args.systems)

        if args.compute_bins or args.plot_bins:
            logging.info("Computing bins of gold results: bin N holds the number of data items for which exactly N systems obtained a true positive")
            r.compute_bins()
            # print(r.results)
            logging.info("Number of data points in each bin")
            print(r.bins)
            logging.info("Number of true positives in each bin for each system")
            print(r.nps_per_bin)
            logging.info("Percentage of true positives in each bin for each system")
            print(np.round(r.prop_nps_per_bin, decimals=2)*100)
            logging.info("Number of true positives for each system")
            print(np.sum(r.results, axis=1))
            logging.info("Recall of each system")
            print(np.round(np.sum(r.results, axis=1)/np.sum(r.bins), decimals=4)*100)
        if args.plot_bins:
            logging.info("Plotting bins as {} to {}".format(args.displayed_values, args.plot_bins))
            names = r.names
            if args.system_names:
                names = args.system_names.split(sep=',')
            r.plot_bins_v2(file=args.plot_bins, names=names, absolute=(args.displayed_values=="number"))
            

    parse_execute_command_line()
