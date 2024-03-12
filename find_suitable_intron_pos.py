import numpy as np
import pandas as pd
import gzip
import random
import argparse
import sys
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


def translate(cds):
    map = {"TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
           "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
           "TAT": "Y", "TAC": "Y",
           "TGT": "C", "TGC": "C", "TGG": "W",
           "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
           "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
           "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
           "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
           "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
           "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
           "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
           "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
           "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
           "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
           "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
           "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G"}

    aa_seq = ""
    for i in range(int(len(cds) / 3)):
        aa_seq += map[cds[3 * i:3 * i + 3]]

    return aa_seq


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
    parser.add_argument('--num_insertions', type=int, default=1000, help='Number of intron positions to try')
    parser.add_argument('--output', required=True, type=str, help='CSV that stores all info')
    parser.add_argument('--percentile', default=70, type=float, help='Percentile above which to store splice site info')
    parser.add_argument('--nt_fasta', action='store_true', default='False', help='Add this if you are supplying a nucleotide fasta rather than a protein fasta')

    # arguments = '--fasta /camp/home/wilkino/home/spliceai/fake_proteome.fa --output out.csv --context_dir /camp/home/wilkino/home/spliceai/SpliceNouveau_gpu/data/'
    # arguments += ' --n_removal_attempts 1000 --num_insertions 100'
    # args = parser.parse_args(arguments.split())
    args = parser.parse_args()

    # Read in fasta file:
    fasta_dict = read_fasta(args.fasta)

    # Read in the output CSV

    if not os.path.exists(args.output):
        with open(args.output, 'w', newline='') as file:
            file.write('name,sequence,intron_position,don_score,acc_score\n')
        completed_proteins = []
    else:
        with open(args.output, 'r') as file:
            completed_proteins = []
            for i, line in enumerate(file):
                if i > 0:
                    completed_proteins.append(line.rstrip().split(',')[0])

    still_to_do = set([a for a in fasta_dict.keys() if a not in completed_proteins])

    while len(still_to_do) > 0:

        if len(still_to_do) == 0:
            print('complete')
            break

        protein_to_do = random.choice(list(still_to_do))
        print(protein_to_do)

        ### Phase 1 - remove unwanted splice sites ###

        aa_seq = fasta_dict[protein_to_do]
        if args.nt_fasta:
            initial_nt_seq = aa_seq  # it wasn't actually an AA seq...
        else:
            initial_nt_seq = make_nt_seq(aa_seq)

        new_nt_seq = initial_nt_seq

        best_score = 1000000

        for i in range(args.n_removal_attempts):
            if i > 0:  # then we mutate
                aa_pos = random.choices(np.arange(len(aa_seq)), weights=smoothed_aa_badness_pos_best_seq, k=1)[0]

                new_nt_seq = mutate_codons(best_nt_seq, aa_seq=aa_seq, start_codon=aa_pos, end_codon=aa_pos, n=1)

                assert translate(new_nt_seq) == aa_seq

            acceptor_probs, donor_probs = get_probs([new_nt_seq], good_contexts=[], context_seqs=[],
                                                    dont_use_contexts=True)

            acceptor_probs = acceptor_probs[0, :]
            donor_probs = donor_probs[0, :]

            badness_pos = acceptor_probs + donor_probs  # Where the splice sites are

            aa_badness_pos = [np.mean(badness_pos[3 * i:3 * i + 3]) for i in range(int(len(new_nt_seq) / 3))]

            # Create a kernel for smoothing
            kernel = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 6.4, 3.2, 1.6, 0.8, 0.4, 0.2, 0.1]

            smoothed_aa_badness_pos = np.convolve(aa_badness_pos, kernel, mode='same')

            score = np.max(acceptor_probs) + np.mean(acceptor_probs) + np.max(donor_probs) + np.mean(donor_probs)

            if i == 0 or score < best_score:
                best_score = score
                smoothed_aa_badness_pos_best_seq = smoothed_aa_badness_pos
                best_nt_seq = new_nt_seq

                if score < 0.05:
                    break

        ### Phase 2 - insert introns into random positions and see if they work well ###

        intron_insert_positions = list(np.arange(min([len(best_nt_seq), args.num_insertions])))

        intron_pos_d = {}

        for intron_insert_position in intron_insert_positions:
            seq_with_intron = best_nt_seq[0:intron_insert_position] + args.intron + best_nt_seq[intron_insert_position:]

            acceptor_probs, donor_probs = get_probs([seq_with_intron], good_contexts=[],
                                                    context_seqs=[], dont_use_contexts=True)
            acceptor_probs = acceptor_probs[0, :]
            donor_probs = donor_probs[0, :]

            don_score = float(donor_probs[intron_insert_position])
            acc_score = float(acceptor_probs[intron_insert_position - 1 + len(args.intron)])

            intron_pos_d[intron_insert_position] = [don_score, acc_score]

        all_dons = [a[0] for a in intron_pos_d.values()]
        all_accs = [a[1] for a in intron_pos_d.values()]

        top_dons = set([k for k, a in enumerate(all_dons) if a >= np.percentile(all_dons, args.percentile)])
        top_accs = set([k for k, a in enumerate(all_accs) if a >= np.percentile(all_accs, args.percentile)])

        top_both = top_dons & top_accs

        j = 0
        k = -1
        with open(args.output, 'a', newline='') as file:
            for key, value in intron_pos_d.items():
                k += 1

                if k not in top_both:
                    continue

                don_score = value[0]
                acc_score = value[1]
                intron_insert_position = key

                if j == 0:
                    file.write(','.join([protein_to_do, best_nt_seq, str(intron_insert_position), str(don_score),
                                         str(acc_score)]) + '\n')
                else:
                    file.write(','.join(
                        [protein_to_do, '', str(intron_insert_position), str(don_score), str(acc_score)]) + '\n')

                j += 1

        still_to_do.discard(protein_to_do)

if __name__ == "__main__":
    main()