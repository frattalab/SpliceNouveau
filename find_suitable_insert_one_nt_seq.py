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
    parser.add_argument('--nt_seq', required=True, help='The nucleotide sequence to be tested')
    parser.add_argument('--intron', type=str,
                        default='GTAAGGAAGGGCACATCAATCTTTGCTTAATTGTCCTTTACTCTAAAGATGTATTTTATCATACTGAATGCTAAACTTGATATCTCCTTTTAG')
    parser.add_argument('--output', required=True)

    args = parser.parse_args()

    best_nt_seq = args.nt_seq

    intron_insert_positions = list(np.arange(len(best_nt_seq)))

    intron_pos_d = {}


    for intron_insert_position in intron_insert_positions:
        seq_with_intron = best_nt_seq[0:intron_insert_position] + args.intron + best_nt_seq[intron_insert_position:]
        acceptor_probs, donor_probs = get_probs([seq_with_intron], good_contexts=[],
                                                context_seqs=[], dont_use_contexts=True)
        acceptor_probs = acceptor_probs[0, :]
        donor_probs = donor_probs[0, :]

        don_score = float(donor_probs[intron_insert_position])
        acc_score = float(acceptor_probs[intron_insert_position - 1 + len(args.intron)])

        intron_pos_d[intron_insert_position] = [don_score, acc_score, seq_with_intron]


    with open(args.output, 'w') as file:
        file.write('position,don,acc,seq\n')
        for key, value in intron_pos_d.items():
            file.write(str(key) + ',' + str(value[0]) + ',' + str(value[1]) + ',' + value[2] + '\n')



if __name__ == "__main__":
    main()