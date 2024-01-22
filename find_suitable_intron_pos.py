import numpy as np
import pandas as pd
import gzip
import random
import argparse
import sys
from SpliceNouveau import *
import os

from keras.models import load_model
from pkg_resources import resource_filename
from spliceai.utils import one_hot_encode
import tensorflow as tf

paths = ('models/spliceai{}.h5'.format(x) for x in range(1, 6))
models = [load_model(resource_filename('spliceai', x)) for x in paths]


def mutate_codons(seq, aa_seq, n, start_codon, end_codon, pptness=False):
    assert len(seq) == 3 * len(aa_seq)
    for _ in range(n):
        aa_to_mut = start_codon
        new_codon = make_nt_seq(aa_seq[aa_to_mut], pptness)
        new_seq = seq[0:3 * aa_to_mut] + new_codon + seq[3 * aa_to_mut + 3:]
        assert len(new_seq) == len(seq), "idiot"

        seq = new_seq

    return seq



def get_probs_one_context(input_sequences):

    context = 10000

    x_list = []

    for input_sequence in input_sequences:
        x = one_hot_encode('N' * (context // 2) + input_sequence + 'N' * (context // 2))[None, :]
        x_list.append(x)

    concat_x = tf.concat(x_list, axis=0)

    y = np.mean([models[m].predict(concat_x, steps=1) for m in range(5)],
                axis=0)  # this needs to change? No, it's fine I think

    acceptor_prob = y[:, :, 1]
    donor_prob = y[:, :, 2]

    # Shift acceptor probabilities so that they align correctly (at the G of the AG)
    acceptor_prob = np.roll(acceptor_prob, shift=-1, axis=1)
    acceptor_prob[:, -1] = 0

    # Shift donor probabilities so that they align correctly (at the G of the GT)
    donor_prob = np.roll(donor_prob, shift=1, axis=1)
    donor_prob[:, 0] = 0

    return acceptor_prob, donor_prob


def get_probs(input_sequences, context_seqs, good_contexts, dont_use_contexts=False):
    """
    The purpose of this function is to attach different contextual sequences to each input sequence, then
    take the average score. The contextual sequences used here were found to provide a good generalisation
    in testing during SpliceNouveau development.
    """
    if dont_use_contexts:
        acceptor_prob, donor_prob = get_probs_one_context(input_sequences)
        return acceptor_prob, donor_prob
    else:
        for index, row in good_contexts.iterrows():
            context = row['context']  # length of context
            seq_no = row['seq_no']  # context sequence

            upstream_context = context_seqs[seq_no][5000 - context // 2:5000 + context // 2]
            downstream_context = context_seqs[seq_no + 1][5000 - context // 2:5000 + context // 2]

            seqs_with_context = [upstream_context + s + downstream_context for s in input_sequences]

            acceptor_prob, donor_prob = get_probs_one_context(seqs_with_context)

            # Trim predictions from added context
            if index == 0:
                mean_acceptor_prob = acceptor_prob[:, len(upstream_context):-len(downstream_context)] / len(
                    good_contexts)
                mean_donor_prob = donor_prob[:, len(upstream_context):-len(downstream_context)] / len(good_contexts)
            else:
                mean_acceptor_prob += acceptor_prob[:, len(upstream_context):-len(downstream_context)] / len(
                    good_contexts)
                mean_donor_prob += donor_prob[:, len(upstream_context):-len(downstream_context)] / len(
                    good_contexts)

        return mean_acceptor_prob, mean_donor_prob


def read_fasta(file_path):
    fasta_dict = {}
    current_header = ""
    current_sequence = ""

    with open(file_path, 'r') as fasta_file:
        for line in fasta_file:
            line = line.strip()
            if line.startswith(">"):
                # If a header line is encountered, save the previous entry and start a new one
                if current_header:
                    fasta_dict[current_header] = current_sequence
                current_header = line[1:]
                current_sequence = ""
            else:
                # Concatenate the sequence lines
                current_sequence += line

        # Add the last entry to the dictionary
        if current_header:
            fasta_dict[current_header] = current_sequence

    return fasta_dict


def make_nt_seq(aa_seq, pptness=False):
    import random
    # d = {"A": ["GCA", "GCC", "GCT", "GCG"]}

    d = {"A": ["GCT", "GCC", "GCA", "GCG"], "I": ["ATT", "ATC", "ATA"],
         "R": ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"], "L": ["CTT", "CTC", "CTA", "CTG", "TTA", "TTG"],
         "N": ["AAT", "AAC"], "K": ["AAA", "AAG"], "D": ["GAT", "GAC"], "M": ["ATG"],
         "F": ["TTT", "TTC"], "C": ["TGT", "TGC"], "P": ["CCT", "CCC", "CCA", "CCG"],
         "Q": ["CAA", "CAG"], "S": ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
         "E": ["GAA", "GAG"], "T": ["ACT", "ACC", "ACA", "ACG"],
         "W": ["TGG"],
         "G": ["GGT", "GGC", "GGA", "GGG"], "Y": ["TAT", "TAC"],
         "H": ["CAT", "CAC"], "V": ["GTT", "GTC", "GTA", "GTG"]}

    seq = ""
    for aa in aa_seq:
        if pptness == False:
            seq += random.choice(d[aa])
        else:
            # Find the one with the most pyrimidines
            potentials = d[aa]
            random.shuffle(potentials)
            best = -1
            for codon in potentials:
                pyr = codon.count("T") + codon.count("C")
                if pyr > best:
                    best = pyr
                    best_codon = codon

            seq += best_codon
    # print(seq)

    return seq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', required=True, help='FASTA file with proteome')
    parser.add_argument('--intron', type=str,
                        default='GTAAGGAAGGGCACATCAATCTTTGCTTAATTGTCCTTTACTCTAAAGATGTATTTTATCATACTGAATGCTAAACTTGATATCTCCTTTTAG')
    parser.add_argument('--n_removal_attempts', type=int, default=1000)
    parser.add_argument('--window_size', type=int, default=5,
                        help='The window size, in aa, for smoothing during inital optimisation')
    parser.add_argument('--add_to_all', type=float, default=0.05,
                        help='Increase to reduce bias towards mutating positions with splice sites')
    parser.add_argument('--num_insertions', type=int, default=100, help='Number of intron positions to try')
    parser.add_argument('--output', required=True, type=str, help='CSV that stores all info')
    parser.add_argument("--context_dir", default="data/", help="location ")

    arguments = '--fasta /camp/home/wilkino/home/spliceai/fake_proteome.fa --output out.csv --context_dir /camp/home/wilkino/home/spliceai/SpliceNouveau_gpu/data/'
    arguments += ' --n_removal_attempts 1000'
    args = parser.parse_args(arguments.split())

    # Read in context data
    context_seqs = []
    with open(args.context_dir + "20_good_contexts.csv") as file:
        for line in file:
            context_seqs.append(line.rstrip())

    good_contexts = pd.read_csv(args.context_dir + "6_good_conditions.csv")

    # Read in fasta file:
    fasta_dict = read_fasta(args.fasta)

    # Read in the output CSV

    if not os.path.exists(args.output):
        with open(args.output, 'w') as file:
            file.write('name,sequence,intron_positions')
        completed_proteins = []
    else:
        with open(args.output, 'r') as file:
            completed_proteins = []
            for i, line in enumerate(file):
                if i > 0:
                    completed_proteins.append(line.rstrip().split(',')[0])

    while True:
        still_to_do = [a for a in fasta_dict.keys() if a not in completed_proteins]

        if len(still_to_do) == 0:
            print('complete')
            break

        protein_to_do = random.choice(still_to_do)
        print(protein_to_do)

        ### Phase 1 - remove unwanted splice sites ###

        aa_seq = fasta_dict[protein_to_do]
        initial_nt_seq = make_nt_seq(aa_seq)

        print(initial_nt_seq)

        new_nt_seq = initial_nt_seq

        best_score = 1000000

        for i in range(args.n_removal_attempts):
            if i > 0:  # then we mutate
                aa_pos = random.choices(np.arange(len(aa_seq)), weights=smoothed_aa_badness_pos_best_seq, k=1)[0]

                new_nt_seq = mutate_codons(best_nt_seq, aa_seq=aa_seq, start_codon=aa_pos, end_codon=aa_pos, n=1)

            acceptor_probs, donor_probs = get_probs([new_nt_seq], good_contexts=[], context_seqs=[], dont_use_contexts=True)

            acceptor_probs = acceptor_probs[0, :]
            donor_probs = donor_probs[0, :]

            badness_pos = acceptor_probs + donor_probs  # Where the splice sites are

            aa_badness_pos = [np.mean(badness_pos[3 * i:3 * i + 3]) for i in range(int(len(new_nt_seq) / 3))]

            # Define the downsampling factor
            smoothing_factor = 5  # This will downsample to length 30 / 3 = 10
            # Create a kernel with a stride of 3 for the moving average
            kernel = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 6.4, 3.2, 1.6, 0.8, 0.4, 0.2, 0.1]

            smoothed_aa_badness_pos = np.convolve(aa_badness_pos, kernel, mode='same')

            score = np.max(acceptor_probs) + np.mean(acceptor_probs) + np.max(donor_probs) + np.mean(donor_probs)

            if i == 0 or score < best_score:
                best_score = score
                smoothed_aa_badness_pos_best_seq = smoothed_aa_badness_pos
                best_nt_seq = new_nt_seq
                print(best_score)

                if score < 0.05:
                    break

        print(best_nt_seq)
        assert 0 == 1, 'yo'

        ### Phase 2 - insert introns into random positions and see if they work well ###

        intron_insert_positions = random.sample(np.arange(3 * len(args.aa_seq)), k=args.num_insertions)

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

                # file.write(str(intron_insert_position) + ',' + str(donor_probs[intron_insert_position]) + ',' +
                #            str(acceptor_probs[intron_insert_position + len(args.intron)]) + '\n')

if __name__ == "__main__":
    main()