import numpy as np
import pandas as pd
import gzip
import random
import argparse
from os.path import exists
import sys
from SpliceNouveau import *

from keras.models import load_model
from pkg_resources import resource_filename
from spliceai.utils import one_hot_encode
import tensorflow as tf

paths = ('models/spliceai{}.h5'.format(x) for x in range(1, 6))
models = [load_model(resource_filename('spliceai', x)) for x in paths]





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aa_seq', required=True)
    parser.add_argument('--intron', type=str, default='GTAAGGAAGGGCACATCAATCTTTGCTTAATTGTCCTTTACTCTAAAGATGTATTTTATCATACTGAATGCTAAACTTGATATCTCCTTTTAG')
    parser.add_argument('--n_removal_attempts', type=int, default=1000)
    parser.add_argument('--window_size', type=int, default=5, help='The window size, in aa, for smoothing during inital optimisation')
    parser.add_argument('--add_to_all', type=float, default=0.05, help='Increase to reduce bias towards mutating positions with splice sites')
    parser.add_argument('--num_insertions', type=int, default=100, help='Number of intron positions to try')
    parser.add_argument('--output', required=True, type=str)
    # parser.add_argument('--initial_3p_utr', default='CTAAGACAGAAATTCGGGAAAAACTAGCCAAAATGTACAAGACCACACCGGATGTCATCTGCacgcgtgtttaaacccgctg')
    # parser.add_argument('--initial_5p_utr', default='actaactttgacctccatagaagacaccgactctactgatacgtagccgccacc')
    args = parser.parse_args()

    # Read in context data
    context_seqs = []
    with open(args.context_dir + "20_good_contexts.csv") as file:
        for line in file:
            context_seqs.append(line.rstrip())

    good_contexts = pd.read_csv(args.context_dir + "6_good_conditions.csv")

    initial_nt_seq = make_nt_seq(args.aa_seq)

    ### Phase 1 - remove unwanted splice sites ###

    new_nt_seq = initial_nt_seq

    best_score = 1000000

    for i in range(args.n_removal_attempts):
        if i > 0:  # then we mutate
            aa = random.choices(np.arange(len(args.aa_seq)), weights=smoothed_aa_badness_pos_best_seq, k=1)

            new_nt_seq = mutate_codons(best_nt_seq, aa_seq=args.aa_seq, start_codon=aa, end_codon=aa)

        acceptor_probs, donor_probs = get_probs(new_nt_seq, good_contexts=good_contexts, context_seqs=context_seqs)

        acceptor_probs = acceptor_probs[0, :]
        donor_probs = donor_probs[0, :]

        badness_pos = acceptor_probs + donor_probs  # Where the splice sites are

        aa_badness_pos = np.mean(badness_pos.reshape(-1, 3), axis=1)
        smoothed_aa_badness_pos = np.convolve(aa_badness_pos,
                                                          np.ones(args.window_size) / args.window_size,
                                                          mode='valid') + args.add_to_all

        score = np.sum(acceptor_probs) + np.sum(donor_probs)

        if i == 0 or score < best_score:
            best_score = score
            smoothed_aa_badness_pos_best_seq = smoothed_aa_badness_pos
            best_nt_seq = new_nt_seq

            if score < 0.1:
                break

    print(best_nt_seq)

    ### Phase 2 - insert introns into random positions and see if they work well ###

    intron_insert_positions = random.sample(np.arange(3*len(args.aa_seq)), k=args.num_insertions)

    with open(args.output, 'w') as file:
        file.write('aa_seq: ' + args.aa_seq + '\n')
        file.write('nt_seq: ' + best_nt_seq + '\n')
        file.write('pos,donor,acceptor\n')
        for intron_insert_position in intron_insert_positions:
            seq_with_intron = best_nt_seq[0:intron_insert_position] + args.intron + best_nt_seq[intron_insert_position:]

            acceptor_probs, donor_probs = get_probs(seq_with_intron, good_contexts=good_contexts,
                                                    context_seqs=context_seqs)
            acceptor_probs = acceptor_probs[0, :]
            donor_probs = donor_probs[0, :]

            file.write(str(intron_insert_position) + ',' + str(donor_probs[intron_insert_position]) + ',' +
                       str(acceptor_probs[intron_insert_position + len(args.intron)]) + '\n')

    
















if __name__ == "__main__":
    main()