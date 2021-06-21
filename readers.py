#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Readers: compute True Positive status from various types of input formats.

Given the outputs of N systems in a given format for the same inputs, outputs a table of the form:
L_input<TAB>TP<TAB>TP<TAB>TP ...

where
  "input" is some identifier for the input example
  "L" is the line in the provided files on which the input was found
  for each of the N systems, "TP" is 1 (true positive) or 0 (false negative) depending on whether or not the system prediction matches the gold standard for this input example


EXAMPLE CALL
for CONLL format. Prepare a TP table for the token-level predictions of 12 NER models on the Biocreative 5 CDR chemical dataset:
python readers.py --output-file data/bc5-c-models.tsv data/bc5-chemical/*.tsv


INPUT FORMATS:

1. CONLL sequence tagging format:
token<TAB>gold<TAB>system

where system uses some variant of the BIO schema.
Only tokens with a gold tag other than "O" are retained.
Computes true positives at the token level.

The input TSV file must include a header line and either contain:
- exactly three columns containing token, gold, and system prediction
- any number of columns in any order, with column labels including "input", "gold", and "system"

where bc5-d-model-1.tsv starts with the following lines:
Token	True Label	Predicted Label
famotidine	O	O
-	O	O
associated	O	O
delirium	B-disease	B-disease
.	O	O

Here are the first lines of the resulting True Positive table, in which the "input" column is the concatenation of the input line number (starting with zero) and the input token for which the True Positive status was computed:
input   bc5-d-model-1.tab	bc5-d-model-2.tab	bc5-d-model-3.tab
3_delirium      1       1       1
29_ulcers       0       1       1

2. (more input formats to be added in the future)

"""

"""
    Copyright (c) 2021 LISN CNRS
    All rights reserved.
    Pierre Zweigenbaum, LISN, CNRS, Université Paris-Saclay <pz@lisn.fr>
"""

import sys
import os
import argparse
import logging
import pandas as pd

# Program version
version = '0.1'

#================
# Base class for set of results
#================
class CONLL():
    """CONLL format reader of true positive information"""
    cols = ["input", "gold", "system"]

    def __init__(self, file, delimiter="\t"):
        self.file = file
        # keep blank lines to build correct index
        df = pd.read_csv(file, sep=delimiter, skip_blank_lines=False, keep_default_na=False)
        # assumes column names are those in self.cols;
        # the caller may change their order OR their names, but not both
        if not(all(c in df.columns for c in self.cols)):
            # if changed names, normalize to standard names
            df.columns = self.cols
        self.data = df
        self.compute_tps()
        self.index_tp_input()

    def compute_tps(self, outside_label="O"):
        """Assign 1 or 0 to non-O data points according to system prediction"""
        # need to ignore blank lines
        non_o = self.data[(self.data["gold"] != "") & (self.data["gold"] != outside_label)]
        tps = non_o["gold"] == non_o["system"] # Boolean value
        # keep input and Boolean value (1 or 0)
        self.tps = pd.DataFrame({"input": non_o["input"], self.file: tps * 1})
        return

    def index_tp_input(self):
        """Concatenate index to input string in tps DataFrame"""
        self.tps["input"] = [ "{}_{}".format(i, s) for i, s in zip(list(self.tps.index), list(self.tps["input"])) ]
        return

    def output_tps(self, file, delimiter="\t"):
        self.tps.to_csv(file, sep=delimiter, index=False) # do not print index
        return

    def add_tps(self, others):
        """Add columns for other results for the same inputs"""
        l = len(self.tps)
        assert all(len(df.tps)==l for df in others), "All input files must have the same number of non-O gold entries"
        for df in others:       # create a column with name=file for each other file
            self.tps[df.file] = df.tps[df.file]
        return

if __name__ == '__main__':
    def parse_execute_command_line():
        parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                         formatter_class=argparse.RawDescriptionHelpFormatter,
                                         description=__doc__)

        parser.add_argument("files", nargs="+", help="file containing binary system results (e.g., true positives for each gold positive)")

        groupO = parser.add_argument_group('Options')
        groupO.add_argument("--output-file", required=True, help="Output file")
        groupO.add_argument("--input-format", choices=["CONLL"], help="Input format")

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

        logging.info("Reading input files {}".format(args.files))
        f_tp_l = [CONLL(f) for f in args.files]
        df = f_tp_l[0]       # true positive values for first file
        if len(f_tp_l) > 0:
            logging.info("Merging input files' true positive information")
            df.add_tps(f_tp_l[1:]) # add true positive values of other files

        logging.info("Writing to ouput file {}".format(args.output_file))
        df.output_tps(args.output_file)

    parse_execute_command_line()

