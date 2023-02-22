#!/usr/bin/env python
import argparse
import mdtraj as md
import mdtraj_upside as mu
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--input_file', help='input upside file')
parser.add_argument('--output_file', help='output xtc filename')
parser.add_argument('--align', action='store_true', help='flag to align trj')

def entry_point():

    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file
    align = args.align

    trj = mu.load_upside_traj(input_file)
    if align:
        trj.superpose(trj[0])
    trj.save_xtc(output_file)


if __name__=='__main__':
    entry_point()
