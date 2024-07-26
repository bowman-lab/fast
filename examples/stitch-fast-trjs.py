from enspara import ra
import numpy as np
import mdtraj as md
from glob import glob
import sys
import argparse


"""
Takes an assignment file from FAST and desired MSM state
and traces back to the starting state, yielding a continous
trajectory.

Usage: python stitch-fast-trjs.py '/path/to/fast/trajs/*.xtc' \
/path/to/topology.pdb /path/to/assignments.h5 target_state
"""

def find_connecting_indicies(assigs, start, end):
    """
    Traces from a desired end state to a starting state
    to yield the continuous trajectory that lead to a state.
    """
    
    all_traj_ns, all_frame_ns = [],[]
    
    temp_end = end
    n_trajs_tot = len(assigs)
    iterations = 0
    while temp_end != start and iterations < n_trajs_tot:
        #Find the first instance in assignments file where we see
        #the end state, returns sorted tuple of trajectory #, frame #
        trj_ns, frame_n = ra.where(assigs==temp_end) 
        all_traj_ns.append(trj_ns[0])
        all_frame_ns.append(frame_n[0])
        
        #Grab the state we started that trajectory from
        temp_end = assigs[all_traj_ns[-1]][0]
    
        #Protect against somehow running an infinite loop.
        #Shouldn't happen if this is well behaved FAST data..
        iterations+=1

    if iterations == n_trajs_tot:
        raise ValueError(f"Could not trace from state {end} to {start}, are you sure they're connected?")

    #Stack the traj, frame indicies
    traj_frames = np.stack([all_traj_ns, all_frame_ns],axis=1)

    #Flip so we start from state 0 and lead to end state.
    traj_frames = np.flip(traj_frames, axis=0)
    return traj_frames

def assemble_traj(topfile, traj_files, traj_trace, subsample, align, alignment_selection):
    #Load the topology file first
    top = md.load(topfile).top
    #Make an empty list to put trajs into
    trj_list=[]

    #Load everything in the traj_ trace
    for traj_n, frame_n in traj_trace:
        print(f'Traj_number: {traj_n}')
        print(f'Loading traj: {traj_files[traj_n]}.')
        #Load by stride, only take frames until the break point
        trj_list.append(md.load(traj_files[traj_n], top=top, stride=subsample)
                        [:int(np.ceil(frame_n/subsample))])

    #Join the trajectory list into one trajectory
    final_traj = md.join(trj_list)
    
    #Optionally align
    if align==True:
        final_traj = final_traj.superpose(final_traj, 
                            atom_indices = top.select(alignment_selection))
    return final_traj

def process_command_line(argv):
	parser = argparse.ArgumentParser(
        prog='Make FAST continuous Trajectory',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""Takes an assignment file from FAST and desired MSM state
						and traces back to the starting state, yielding a continous
						trajectory.""")

    # INPUTS
	input_args = parser.add_argument_group("Inputs")
	input_args.add_argument(
        "trjpath",
        help="Globbable path to trajectory files.")
	input_args.add_argument(
        "topfile",
        help="Path to topology file which loads trajectories.")
	input_args.add_argument(
        "assignments_file",
        help="Path to assignments file.")
	input_args.add_argument(
        "end_state", type=int,
        help="MSM state to trace to.")
	input_args.add_argument(
        "--subsample", type=int, default=1,
        help="Amount to downsample by.")
	input_args.add_argument(
        "--start_state", type=int, default=0,
        help="State to trace back to.")
	input_args.add_argument(
        "--out_name", default=None,
        help="What to name your output traj.")
	input_args.add_argument(
        "--align_prot", default=True,
        help="Align protein in output trajectory?")
	input_args.add_argument(
        "--alignment_selection", default='protein',
        help="Atom selection from mdtraj used to align protein.")
    
	args = parser.parse_args(argv[1:])

	return args

def main(argv=None):

    args = process_command_line(argv)
    print(f'Stitching FAST trajectory from MSM state {args.start_state} to {args.end_state}.', flush=True)

    assignments = ra.load(args.assignments_file)

    #Check to make sure start/end states are in the assignments file.
    if np.isin(args.start_state, np.unique(assignments.flatten())) != True:
        raise ValueError(f"Could not find starting state: {args.start_state} in assignments file {args.assignments_file}.")

    if np.isin(args.end_state, np.unique(assignments.flatten())) != True:
        raise ValueError(f"Could not find starting state: {args.end_state} in assignments file {args.assignments_file}.")


    trj_files = sorted(glob(args.trjpath))
    if args.out_name ==  None:
    	args.out_name = f'FAST-traj-{args.start_state}-to-{args.end_state}.xtc'

    traj_trace = find_connecting_indicies(assignments, args.start_state, args.end_state)

    traj = assemble_traj(args.topfile, trj_files, traj_trace, 
    	args.subsample, args.align_prot, args.alignment_selection)

    traj.save_xtc(args.out_name)
    print(f'Done! Saved to: {args.out_name}')


if __name__ == "__main__":
    sys.exit(main(sys.argv))
