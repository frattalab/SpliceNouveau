import numpy as np
import pandas as pd
import gzip
import random
import argparse
from os.path import exists
import sys
import random

# conda activate keras-gpu
# A test: python3 SpliceNouveau.py --initial_cds atggCgagaACAATGGTTGCTATGGTGTCCAAAGGTGAGGCAGTCATAAAGGAGTTTATGAGGTTCAAGGTGCACATGGAAGGGTCAATGAACGGACATGAGTTCGAAATTGAAGGTGAGGGCGAGGGCCGCCCCTATGAAGGGACACAAACTGCCAAGCTCAAAGTGACCAAGGGCGGGCCTCTGCCCTTCTCTTGGGATATCCTGAGCCCGCAGTTTATGTACGGCAGCCGGGCTTTCACCAAACACCCTGCCGATATCCCAGACTACTATAAACAGTCCTTTCCAGAAGGATTTAAGTGGGAGCGAGTCATGAATTTCGAGGACGGAGGTGCCGTGACGGTTACTCAGGACACCAGCCTGGAGGACGGCACCCTGATCTACAAGGTGAAGCTGAGGGGCACCAACTTCCCCCCCGACGGCCCCGTGATGCAGAAGAAGACCATGGGCTGGGAGGCCAGCACCGAGAGGCTGTACCCCGAGGACGGCGTGCTGAAGGGCGACATCAAGATGGCCCTGAGGCTGAAGGACGGCGGCAGGTACCTGGCCGACTTCAAGACCACCTACAAGGCCAAGAAGCCCGTGCAGATGCCCGGCGCCTACAACGTGGACAGGAAGCTGGACATCACCAGCCACAACGAGGACTACACCGTGGTGGAGCAGTACGAGAGGAGCGAGGGCAGGCACAGCACCGGCGGCATGGACGAGCTGTACAAGGACTACAAGGACGATGATGACAAG --initial_intron1 GTAAGNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTGTGTGTGTGTGTGTGTGAATGTGTGTGTGTGTGTGTGNNAG --initial_intron2 GTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYAG --ce_start 43 --ce_end 159 --ce_mut_chance 1 --five_utr CGGCCGCTTCTTGGTGCCAGCTTATCAtagcgctaccggtcgccacc --three_utr TGATAAACAAATGGTAAGGAAGGGCACATCAATCTTTGCTTAATTGTCCTTTACTCTAAAGATGTATTTTATCATACTGAATGCTAAACTTGATATCTCCTTTTAGGTCATTGATGTCCTTCACCCCGGGAAGGCGACAGTGCCTAAGACAGAAATTCGGGAAAAACTAGCCAAAATGTACAAGACCACACCGGATGTCATCTTTGTATTTGGATTCAGAACTCAGTAAACTGGATCCGCAGGCCTCTGCTAGCTTGACTGACTGAGATACAGCGTACCTTCAGCTCACAGACATGATAAGATACATTGATGAGTTTGGACAAACCACAACTAGAATGCAGTGAAAAAAATGCTTTATTTGTGAAATTTGTGATGCTATTGCTTTATTTGTAACCATTATAAGCTGCAATAAACAAGTTAACAACAACAATTGCATTCATTTTATGTTTCAGGTTCAGGGGGAGGTGTGGGAGGTTTTTTAA --ignore_end 470 --aa generate_it --upstream_mut_chance 0.2 --downstream_mut_chance 0.2 --output mscar/aars1_inspired_closer_aim_0p5.csv --target_cryptic_donor 0.5 --target_cryptic_acc 0.5 -a 5 --intron1_mut_chance 0.5 --intron2_mut_chance 0.5 -n 2000 --cds_mut_end_trim 569

if "--skip_tf" in sys.argv:
    load_tf = False
else:
    load_tf = True

if load_tf:
    from keras.models import load_model
    from pkg_resources import resource_filename
    from spliceai.utils import one_hot_encode
    import tensorflow as tf

    paths = ('models/spliceai{}.h5'.format(x) for x in range(1, 6))
    models = [load_model(resource_filename('spliceai', x)) for x in paths]


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


else:
    print("NOT LOADING TENSOR FLOW! THIS IS JUST FOR TESTS")


    def get_probs(input_sequence, context_seqs, good_contexts):
        return [0] * len(input_sequence), [0] * len(input_sequence)


def translate_cds(cds):
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


def make_random_seq(l):
    nts = ["A", "C", "G", "T"]

    return ''.join(random.choices(nts, k=l))


def make_ppt(l, frac):
    s = ""
    for _ in range(l):
        if random.uniform(0, 1) <= frac:
            s += random.choice(["C", "T"])
        else:
            s += random.choice(["A", "G"])

    return s


def random_mut(seq, rate):
    out = []
    nts = ["A", "C", "G", "T"]
    for s in seq:
        if random.uniform(0, 1) <= rate:
            out.append(random.choice(nts))
        else:
            out.append(s)

    return ''.join(out)


def remove_NY(initial_seq, pyrimidine_chance):
    # Convert string to list
    seq = list(initial_seq)
    new_seq = seq

    for i, character in enumerate(seq):
        if character.upper() == "N":
            new_seq[i] = random.choice(["a", "t", "c", "g"])
        elif character.upper() == "Y":
            new_seq[i] = make_ppt(1, pyrimidine_chance).lower()

    return ''.join(new_seq)


def mutate_single_codon(codon):
    aa = translate_cds(codon)

    d = {"A": ["GCT", "GCC", "GCA", "GCG"], "I": ["ATT", "ATC", "ATA"],
         "R": ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"], "L": ["CTT", "CTC", "CTA", "CTG", "TTA", "TTG"],
         "N": ["AAT", "AAC"], "K": ["AAA", "AAG"], "D": ["GAT", "GAC"], "M": ["ATG"],
         "F": ["TTT", "TTC"], "C": ["TGT", "TGC"], "P": ["CCT", "CCC", "CCA", "CCG"],
         "Q": ["CAA", "CAG"], "S": ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
         "E": ["GAA", "GAG"], "T": ["ACT", "ACC", "ACA", "ACG"],
         "W": ["TGG"],
         "G": ["GGT", "GGC", "GGA", "GGG"], "Y": ["TAT", "TAC"],
         "H": ["CAT", "CAC"], "V": ["GTT", "GTC", "GTA", "GTG"]}

    possible_codons = d[aa]

    possible_codons = [a for a in possible_codons if a != codon]

    if len(possible_codons) == 0:
        return codon

    else:
        return random.choice(possible_codons)


def mutate_codons(seq, aa_seq, n, start_codon, end_codon, pptness=False):
    assert len(seq) == 3 * len(aa_seq)
    for _ in range(n):
        aa_to_mut = random.randint(start_codon, end_codon)
        new_codon = make_nt_seq(aa_seq[aa_to_mut], pptness)
        new_seq = seq[0:3 * aa_to_mut] + new_codon + seq[3 * aa_to_mut + 3:]
        assert len(new_seq) == len(seq), "idiot"

        seq = new_seq

    return seq


def mut_cds(old_cds, mut_n, mut_start, mut_end, pptness=False):
    """
	Note that mut_start and mut_end are in nucleotide coordinates
	"""
    # Determine which codons can be mutated
    first_codon_mut = int(mut_start / 3)  # round down
    last_codon_mut = int(mut_end / 3)  # round down

    aa_seq = translate_cds(old_cds)

    for i in range(100):

        for _ in range(5):
            new_cds = mutate_codons(old_cds, aa_seq, mut_n, first_codon_mut, last_codon_mut, pptness)
            if new_cds != old_cds:
                continue

        if new_cds[0:mut_start] != old_cds[0:mut_start]:
            continue
        if new_cds[mut_end + 1:] != old_cds[mut_end + 1:]:
            continue
        return new_cds

    assert 0 == 1, "Unable to make new sequence"


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("-c", "--csv", type=str, required=True,
    #                   help="A csv that contains all the sequence details")
    parser.add_argument("-n", "--n_iterations_per_attempt", default=1000, type=int)
    parser.add_argument("-a", "--attempts", default=1, type=int, help="The number of evolution attempts")
    parser.add_argument("-e", "--early_stop", default=200, type=int, help="The number of iterations with no "
                                                                          "improvement before stopping")
    parser.add_argument("-o", "--output", required=True, type=str, help="output value")
    parser.add_argument("--aa", type=str, required=False,
                        help="The amino acid sequence encoded", default="generate_it")
    parser.add_argument("--initial_cds", type=str, required=True,
                        help="The initial CDS sequence, WITHOUT stop codon - state generate_it to make it")
    parser.add_argument("--initial_intron1", type=str, required=True)
    parser.add_argument("--initial_intron2", type=str, required=False, default="")
    parser.add_argument("--five_utr", default="")
    parser.add_argument("--three_utr", default="TAA")
    parser.add_argument("--ignore_start", default=0, type=int,
                        help="Ignore nucleotides before this. Eg ignore_start=100 would ignore the "
                             "first 100 nucleotides when scoring sequences")
    parser.add_argument("--ignore_end", default=0, type=int,
                        help="Ignore nucleotides after this. Eg ignore_end=100 would ignore the last 100 nucleotides "
                             "when scoring sequences")
    parser.add_argument("--ce_start", type=int, required=True,
                        help="The position (0 based) of the first CE nucleotide in the CDS. Also use this for "
                             "defining the splice site position within CDS when using --one_intron mode.")
    parser.add_argument("--ce_end", type=int, required=False, default=0,
                        help="The position (0 based) of the last CE nucleotide in the CDS")
    parser.add_argument('--mutate_bad_regions_factor', type=float, default=10,
                        help='Regions which score badly are more likely to be mutated. Higher value results in '
                             'stronger bias towards mutating regions which score badly. 0 sets this to off.')
    parser.add_argument("--ce_mut_weight", type=float, default=1,
                        help="Chance per iteration that CE is mutated")
    parser.add_argument("--ce_mut_n", type=int, default=1,
                        help="Number of mutations per iteration")
    parser.add_argument("--CDS_mut_weight", type=float, default=1,
                        help="Chance per iteration that non-CE CDS is mutated")
    parser.add_argument("--upstream_mut_n", type=int, default=1,
                        help="Number of mutations per iteration")
    parser.add_argument("--downstream_mut_n", type=int, default=1,
                        help="Number of mutations per iteration")
    parser.add_argument("--intron_mut_weight", default=1, type=float)
    parser.add_argument("--intron1_mut_n", default=1, type=int)
    parser.add_argument("--intron2_mut_n", default=1, type=int)
    parser.add_argument("--target_const_donor", default=1, type=float, help="Target spliceAI for upstream/IR donor")
    parser.add_argument("--target_const_acc", default=1, type=float,
                        help="Target spliceAI score for downstream/IR acceptor")
    parser.add_argument("--target_cryptic_donor", default=0.3, type=float)
    parser.add_argument("--target_cryptic_acc", default=0.3, type=float)
    parser.add_argument("--pyrimidine_chance", default=0.85, type=float, help="When replacing 'P's, the chance "
                                                                              "of a C or a T")
    parser.add_argument("--min_improvement", default=0, type=float, help="Minimum improvement to be selected for")
    parser.add_argument("--cds_mut_start_trim", default=0, type=int, help="Don't mutate the first N nucleotides of CDS")
    parser.add_argument("--cds_mut_end_trim", default=0, type=int, help="Don't mutation the last N nucleotides of CDS")
    parser.add_argument("--skip_tf", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ce_score_weight", default=2, type=float,
                        help="Increase the weighting of the cryptic exon. Default is 2")
    parser.add_argument("--one_intron", action="store_true", help="Use if only wanting a single intron")
    parser.add_argument("--ir", action="store_true", help="Only interested in intron retention")
    parser.add_argument("--alt_5p", action="store_true", help="Use alternative 5p (donor) splice sites")
    parser.add_argument("--alt_3p", action="store_true", help="Use alternative 3p (acceptor) splice sites")
    parser.add_argument("--alt_position", type=str, default="",
                        help="When using --alt_5p or --alt_3p, either 'in_intron' or 'in_exon'")
    parser.add_argument("--min_intron_l", type=int, default=70,
                        help="Minimum intron length, when using --alt_3p or --alt_5p")
    parser.add_argument("--alt_5p_start_trim", type=int, default=0,
                        help="When using --alt_5p and --alt_position in_exon,"
                             " this is the first position at which the "
                             " donor can be")
    parser.add_argument("--alt_3p_end_trim", type=int, default=0, help="When using --alt_3p and --alt_position in_exon,"
                                                                       " this is the last position at which the "
                                                                       "acceptor can be. It trims from the end,"
                                                                       " e.g. 100 would mean it won't look in "
                                                                       "the last 100 bases")
    parser.add_argument("--min_alt_dist", type=int, default=1,
                        help="Minimum distance between repressed and alternative splice site. "
                             "Only applicable when using --alt_3p or --alt_5p")
    parser.add_argument("--alt_weight", type=float, default=2,
                        help="Weight of the non-repressed (alternative) splice site in scoring")
    parser.add_argument("--make_codons_ppt", action="store_true", help="Will convert the mutated downstream (for alt_3p) region to a \
		polypyrimidine rich region")
    parser.add_argument("--track_splice_scores", action="store_true", default=False,
                        help="Write out splice scores each time it improves")
    parser.add_argument("--n_seqs_per_it", type=int, help="How many sequences to try in parallel per iteration; "
                                                          "higher numbers will help reduce mutational load",
                        default=16)
    parser.add_argument("--context_dir", default="data/", help="location ")
    parser.add_argument('--dont_use_contexts', default=False, action='store_true', help='Use this command '
                                                                                        'to avoid using the additional '
                                                                                        'contextual flanking sequences'
                                                                                        ' i.e. behave like the original '
                                                                                        'versions of SpliceNouveau.')
    parser.add_argument('--three_p_but_no_pptness', default=False, action='store_true',
                        help='By default, when using alt 3prime mode, it generates a pyrmidine-rich coding sequence. '
                             'Use this flag to turn off this behaviour.')
    parser.add_argument('--conv_window_size', default=9, type=int, help='When prioritising which positions to \
     mutate, this is the with the the triangular convolution window. Larger values spread this across a larger region')

    args = parser.parse_args()

    args.initial_cds = args.initial_cds.upper()

    if args.initial_cds == "GENERATE_IT":
        args.initial_cds = make_nt_seq(args.aa)

    if args.aa == "generate_it":
        args.aa = translate_cds(args.initial_cds)

    if args.ir or args.alt_5p or args.alt_3p:
        print("Setting option --one_intron to True")
        args.one_intron = True

    if args.ir:
        print("Using only ce_start")
        args.ce_end = args.ce_start

        if args.target_const_donor > 0.9 or args.target_const_acc > 0.9:
            print("Warning! Intron retention may be improved by specifying weaker 'constitutive' splice sites")

    if args.alt_5p or args.alt_3p:
        assert args.alt_position in ["in_intron", "in_exon"], "Must choose a valid option for --alt_position!"
        assert args.ignore_start <= args.alt_5p_start_trim, "Need to tweak your trim settings"
        assert args.ignore_end <= args.alt_3p_end_trim, "Need to tweak your trim settings"
    else:
        assert args.alt_position == "", "Can only use alt_position in conjunction with alt_5p or alt_3p!"

    if args.one_intron:
        assert len(args.initial_intron2) == 0, "If using just one intron, intron2 must not be supplied!"
        assert args.intron2_mut_chance == 0, "If using just one intron, intron2 cannot be mutated..."
        assert args.ir or args.alt_5p or args.alt_3p, "Need to use --ir, --alt_5p or --alt_3p!"
        assert int(args.ir) + int(args.alt_5p) + int(args.alt_3p) == 1, "Must only select one of --ir, --alt_5p or " \
                                                                        "--alt_3p!"
        args.ce_mut_chance = 0
    else:
        assert len(args.initial_intron2) > 0, "Need to supply an intron"

    assert args.pyrimidine_chance <= 1
    assert args.intron1_mut_chance <= 1
    assert args.intron2_mut_chance <= 1

    print(args.initial_cds)
    print(args.aa)

    assert len(args.initial_cds) == 3 * len(args.aa)
    assert args.ce_end < len(args.initial_cds)
    assert "N" not in args.initial_cds.upper()
    # assert args.three_utr[0:3] in ["TAG", "TAA", "TGA"], "three prime UTR should start with a stop codon"
    assert translate_cds(args.initial_cds) == args.aa
    if not args.overwrite:
        assert False == (exists(args.output)), "Use a different name file or delete previous output!"

    assert args.min_intron_l <= len(args.initial_intron1), "Intron 1 is too short!"
    if not args.one_intron:
        assert args.min_intron_l <= len(args.initial_intron2), "Intron 2 is too short!"
        assert args.ce_start < args.ce_end
        if (args.ce_end - args.ce_start) % 3 == 0:
            print("WARNING - cryptic exon does not induce frame shift!")

    return args


def mut_noncoding(seq, positions_to_mut, n_mut):
    assert len([1 for a in list(seq) if a.islower()]) > 0, "Only lower case bases can be mutated!"

    seq = list(seq)
    new_seq = seq
    choices = random.choices(positions_to_mut, k=n_mut)
    for i in choices:
        new_seq[i] = random.choice(["a", "t", "c", "g"])

    return ''.join(new_seq)


def mutate_sequence(prev_best_sequence, transcript_structure_array, prev_best_score_contribution_array,
                    mutate_bad_regions_factor, ce_mut_weight, CDS_mut_weight, intron_mut_weight,
                    convolution_window_size):
    # Create dictionary with assignments linked to third row of transcript_structure_array
    # Intron is weighted 3-fold because there is much more freedom in intron
    weight_d = {0: 0, 1: CDS_mut_weight, 2: ce_mut_weight, 3: intron_mut_weight * 3, 4: 0}

    # Normalise contribution array so the max value is 1
    prev_best_score_contribution_array = prev_best_score_contribution_array / np.max(prev_best_score_contribution_array)

    chances = []
    for i in range(len(transcript_structure_array[0, :])):
        # chance = can_be_mutated * (1 + factor * score_contribution * weight_for_this_type_of_sequence)
        chances.append(transcript_structure_array[0, i] * (
                1 + mutate_bad_regions_factor * prev_best_score_contribution_array[i] * weight_d[
            transcript_structure_array[2, i]]))

    chances = np.asarray(chances)
    triangle_filter = np.convolve(np.ones(convolution_window_size), np.ones(convolution_window_size), mode='full')
    smoothed_chances = np.convolve(chances, triangle_filter, mode='same') / np.sum(triangle_filter)

    # Set anything that can't be mutated to zero
    smoothed_chances = smoothed_chances * transcript_structure_array[0, :]

    position_to_mutate = random.choices(list(range(len(prev_best_sequence))),
                                        k=1,
                                        weights=smoothed_chances)[0]

    # What type of region is this?
    position_type = transcript_structure_array[2, position_to_mutate]

    if position_type == 3:  # intron i.e. non-coding
        current_nt = prev_best_sequence[position_to_mutate]
        new_nts = [a for a in ['A', 'T', 'C', 'G'] if a != current_nt.upper()]
        new_seq = list(prev_best_sequence)
        new_seq[position_to_mutate] = random.choice(new_nts)
        new_seq = ''.join(new_seq)

    else:  # it's a coding sequence
        current_codon_pos = transcript_structure_array[5, position_to_mutate]
        codon_number = int(current_codon_pos)

        positions_of_this_codon_in_sequence = [i for i in range(len(prev_best_sequence)) if
                                               int(transcript_structure_array[5, i]) == codon_number]

        current_codon = ''.join(
            [a for i, a in enumerate(list(prev_best_sequence)) if i in positions_of_this_codon_in_sequence])

        new_codon = mutate_single_codon(current_codon)

        new_seq = list(prev_best_sequence)
        for i, p in enumerate(positions_of_this_codon_in_sequence):
            new_seq[p] = new_codon[i]

        new_seq = ''.join(new_seq)

    return new_seq


def make_transcript_structure_array(five_utr, cds, intron1, intron2, three_utr, ce_start, ce_end,
                                    ignore_start=0, ignore_end=0, cds_mut_start_trim=0, cds_mut_end_trim=0,
                                    one_intron=False, ir=False, alt_5p=False, alt_3p=False, min_alt_distance=0,
                                    alt_position="", alt_3p_end_trim=0, min_intron_l=70):
    """
	This array stores information about the structure of the transcript that is being created.

	The first row records whether a given position can be mutated 0 = cannot be mutated, 1 = can be mutated

	The second row records the position of intended splice sites. 0 = not a splice site, 1 = the constant donor,
	2 = the constant acceptor, 3 = cryptic donor, 4 = cryptic acceptor, 5 = alternative donor, 6 = alternative
	acceptor.

	The third row records the type of exonic/intronic region we are in. 0 = 5' UTR, 1 = constitutively expressed
	coding sequence, 2 = cryptically-expressed coding sequence, 3 = intron, 4 = 3' utr

	The fourth row records whether a given region should be scored (0 = false, 1 = true)

	The fifth row records whether a given position is a valid position for an alternative donor/acceptor (0 = false,
	1 = true)

	The sixth row records the codon sub position of each position within the coding sequence. Eg 14.0 is the first
	nucleotide of the 14th codon (starting from zero-th codon, i.e. it's actually the 15th). 3.2 is the third codon
	of the 3rd codon. If it's not coding then this is set to -1. All indexing is zero based (the first
	nucleotide of CDS is 0.0)
	"""
    total_l = len(five_utr) + len(cds) + len(three_utr) + len(intron1) + len(intron2)

    transcript_structure_array = np.zeros((6, total_l))
    transcript_structure_array[3, :] = 1  # By default assume all positions are important for scoring

    to_mut_intron1 = [i for i, a in enumerate(list(intron1)) if a.islower() or a.upper() in ["Y", "N"]]
    to_mut_intron2 = [i for i, a in enumerate(list(intron2)) if a.islower() or a.upper() in ["Y", "N"]]

    cds_counter = -1
    intron1_counter = -1
    intron2_counter = -1
    ce_counter = -1

    for i in range(total_l):
        this_type = None
        if i < len(five_utr):
            this_type = 'five_utr'
            transcript_structure_array[2, i] = 0  # five utr
            transcript_structure_array[5, i] = -1

        elif i < total_l - len(three_utr):  # not in UTRs as UTRs are never mutated
            if i < len(five_utr) + ce_start:
                cds_counter += 1
                this_type = 'upstream_cds'

            elif i < len(five_utr) + ce_start + len(intron1):
                intron1_counter += 1
                this_type = 'intron1'

                transcript_structure_array[2, i] = 3  # intron

                if intron1_counter == 0:
                    if ir or alt_5p:
                        transcript_structure_array[1, i] = 3  # cryptic donor
                    else:
                        transcript_structure_array[1, i] = 1  # constant donor

                if intron1_counter == len(intron1) - 1:
                    if alt_5p:
                        transcript_structure_array[1, i] = 2  # constant acceptor
                    else:  # ir or alt_3p (or normal CE)
                        transcript_structure_array[1, i] = 4  # cryptic acceptor

            elif i < len(five_utr) + len(intron1) + ce_end:
                ce_counter += 1
                cds_counter += 1
                this_type = 'CE'

            elif i < len(five_utr) + len(intron1) + ce_end + len(intron2) and not one_intron:
                intron2_counter += 1
                this_type = 'intron2'

                transcript_structure_array[2, i] = 3  # intron

                if not one_intron:
                    if intron2_counter == 0:
                        transcript_structure_array[1, i] = 3  # cryptic donor
                    elif intron2_counter == len(intron2) - 1:
                        transcript_structure_array[1, i] = 2  # constant acceptor

            elif i < len(five_utr) + len(intron1) + len(cds) + len(intron2):
                cds_counter += 1
                this_type = 'downstream_cds'

            else:
                assert 0 == 1, 'unexpected position'

            if this_type in ['upstream_cds', 'CE', 'downstream_cds']:
                codon_pos = cds_counter // 3 + 0.1 * (cds_counter % 3)
                transcript_structure_array[5, i] = codon_pos
            else:
                codon_pos = -1  # not a codon
                transcript_structure_array[5, i] = codon_pos

        else:
            this_type = 'three_utr'
            transcript_structure_array[2, i] = 4  # intron
            transcript_structure_array[5, i] = -1

            # Fill out first and third rows (index 0 and 2)
        if this_type in ['upstream_cds', 'CE', 'downstream_cds']:
            if cds_mut_start_trim <= cds_counter < len(cds) - cds_mut_end_trim:
                transcript_structure_array[0, i] = 1  # can be mutated

            if this_type == 'CE':
                transcript_structure_array[2, i] = 2  # cryptically expressed coding sequencing
            else:
                transcript_structure_array[2, i] = 1  # constitutively expressed coding sequence

        if this_type == 'intron1':
            if intron1_counter in to_mut_intron1:
                transcript_structure_array[0, i] = 1

        if this_type == 'intron2':
            if intron2_counter in to_mut_intron2:
                transcript_structure_array[0, i] = 1

        # Fill out fourth row (index 3)
        if i < ignore_start or i >= total_l - ignore_end:
            transcript_structure_array[3, i] = 0  # Ignored during scoring
            transcript_structure_array[0, i] = 0  # Also shouldn't mutate stuff that is ignored

    # Find valid positions of alternative splice sites
    if alt_5p:
        if alt_position == 'in_exon':
            transcript_structure_array[4, ignore_start + 1:len(five_utr) + ce_start - min_alt_distance] = 1
        elif alt_position == 'in_intron':
            transcript_structure_array[4,
            len(five_utr) + ce_start + min_alt_distance:len(five_utr) + ce_start + len(intron1) - min_intron_l] = 1

    if alt_3p:
        if alt_position == 'in_exon':
            start_of_region = len(five_utr) + ce_start + len(intron1) + min_alt_distance
            end_of_region = min([total_l - ignore_end, total_l - alt_3p_end_trim])
            transcript_structure_array[4, start_of_region:end_of_region] = 1
        elif alt_position == 'in_intron':
            transcript_structure_array[4,
            len(five_utr) + ce_start + min_intron_l:len(five_utr) + ce_start + len(intron1) - min_alt_distance] = 1

    return transcript_structure_array


def make_score_contribution_array(transcript_structure_array, donor_probs, acceptor_probs, target_const_donor,
                                  target_const_acceptor, target_cryptic_donor, target_cryptic_acceptor,
                                  target_alternative_donor, target_alternative_acceptor, ce_score_weight,
                                  alt_score_weight, alt_5p, alt_3p):
    """
	This function makes a (2,seq_length) array/matrix that gives the score contribution of each donor and acceptor
	probability. The first row is for donors, the second is for acceptors. The 2-sum(whole_array)=total_score
	(don't ask me why I chose 2 as a constant)
	"""

    # First, if alternative 3' or 5', identify the position of the alternative splice site

    if alt_5p:
        # Find the strongest donor in the region of valid alt donor positions
        max_valid_donor = max([a for i, a in enumerate(donor_probs) if transcript_structure_array[4, i] == 1])
        alt_5p_pos = \
            [i for i, a in enumerate(donor_probs) if a == max_valid_donor and transcript_structure_array[4, i] == 1][0]
        transcript_structure_array[1, alt_5p_pos] = 5  # alt donor

    if alt_3p:
        # Find the strongest acc in the region of valid alt acc positions
        max_valid_acc = max([a for i, a in enumerate(acceptor_probs) if transcript_structure_array[4, i] == 1])
        alt_3p_pos = \
            [i for i, a in enumerate(acceptor_probs) if a == max_valid_acc and transcript_structure_array[4, i] == 1][0]
        transcript_structure_array[1, alt_3p_pos] = 6  # alt acceptor

    # Now, score the values in a consistent way

    seq_length = len(donor_probs)

    score_contribution_array = np.zeros((2, seq_length))

    target_donor_values_d = {0: 0, 1: target_const_donor, 2: 0, 3: target_cryptic_donor, 4: 0,
                             5: target_alternative_donor, 6: 0}

    target_acceptor_values_d = {0: 0, 1: 0, 2: target_const_acceptor, 3: 0, 4: target_cryptic_acceptor, 5: 0,
                                6: target_alternative_acceptor}

    scoring_weights_d = {0: 1, 1: 1, 2: 1, 3: ce_score_weight, 4: ce_score_weight, 5: alt_score_weight,
                         6: alt_score_weight}

    for seq_pos in range(seq_length):
        position_type = transcript_structure_array[1, seq_pos]

        # contribution = is_position_scored * abs(target - actual) * score_weight

        score_contribution_array[0, seq_pos] = transcript_structure_array[3, seq_pos] * \
                                               abs(donor_probs[seq_pos] - target_donor_values_d[position_type]) * \
                                               scoring_weights_d[position_type]

        score_contribution_array[1, seq_pos] = transcript_structure_array[3, seq_pos] * \
                                               abs(acceptor_probs[seq_pos] - target_acceptor_values_d[position_type]) * \
                                               scoring_weights_d[position_type]

    score_contribution_array = np.sum(score_contribution_array, axis=0)

    return score_contribution_array


def main():
    print("Reading arguments")
    args = get_args()

    # Read in context data
    context_seqs = []
    with open(args.context_dir + "20_good_contexts.csv") as file:
        for line in file:
            context_seqs.append(line.rstrip())

    good_contexts = pd.read_csv(args.context_dir + "6_good_conditions.csv")

    transcript_structure_array = make_transcript_structure_array(five_utr=args.five_utr,
                                                                 cds=args.initial_cds,
                                                                 intron1=args.initial_intron1,
                                                                 intron2=args.initial_intron2,
                                                                 three_utr=args.three_utr,
                                                                 ce_start=args.ce_start,
                                                                 ce_end=args.ce_end,
                                                                 ignore_start=args.ignore_start,
                                                                 ignore_end=args.ignore_end,
                                                                 cds_mut_start_trim=args.cds_mut_start_trim,
                                                                 cds_mut_end_trim=args.cds_mut_end_trim,
                                                                 one_intron=args.one_intron,
                                                                 ir=args.ir,
                                                                 alt_5p=args.alt_5p,
                                                                 alt_3p=args.alt_3p,
                                                                 min_alt_distance=args.min_alt_dist,
                                                                 alt_position=args.alt_position,
                                                                 alt_3p_end_trim=args.alt_3p_end_trim,
                                                                 min_intron_l=args.min_intron_l)

    print("CHECK THIS IS WHAT'S EXPECTED")
    print("Upstream region:")
    print(args.initial_cds[0:args.ce_start])
    if not args.one_intron:
        print("CE region:")
        print(args.initial_cds[args.ce_start:args.ce_end])
    print("Downstream region:")
    print(args.initial_cds[args.ce_end:])

    print()

    results = {}

    if args.track_splice_scores:
        splice_score_tracker_filename = args.output + ".tracked_scores.csv"
        with open(splice_score_tracker_filename, 'w') as sst:
            sst.write("attempt,iteration,position,donor_prob,acceptor_prob,score,sequence\n")

    for attempt in range(args.attempts):
        best_score = -1000
        print("Attempt " + str(attempt))
        bored = 0
        # Replace Ns and Ps
        starting_5utr = remove_NY(args.five_utr, args.pyrimidine_chance)
        starting_intron1 = remove_NY(args.initial_intron1, args.pyrimidine_chance)
        starting_intron2 = remove_NY(args.initial_intron2, args.pyrimidine_chance)
        starting_3utr = remove_NY(args.three_utr, args.pyrimidine_chance)

        if args.alt_3p:
            if args.three_p_but_no_pptness:
                make_like_ppt = False
            else:
                make_like_ppt = True

            assert len(
                args.initial_cds) - 1 - args.cds_mut_end_trim >= args.ce_end + 1, "Need to tweak your mut_cds trims"

            args.initial_cds = mut_cds(args.initial_cds, 1000, args.ce_end + 1,
                                       len(args.initial_cds) - 1 - args.cds_mut_end_trim,
                                       pptness=make_like_ppt)

        for i in range(args.n_iterations_per_attempt):
            bored += 1


            new_seqs = []
            for j in range(args.n_seqs_per_it):
                if i == 0:
                    new_combined_seq = starting_5utr + args.initial_cds[0:args.ce_start] + starting_intron1 + \
                    args.initial_cds[args.ce_start:args.ce_end] + starting_intron2 + \
                    args.initial_cds[args.ce_end:] + starting_3utr

                if i > 0:

                    new_combined_seq = mutate_sequence(best_seq,
                                                       transcript_structure_array,
                                                       best_score_contribution_array,
                                                       args.mutate_bad_regions_factor,
                                                       args.ce_mut_weight, args.CDS_mut_weight, args.intron_mut_weight,
                                                       args.conv_window_size)


                new_seqs.append(new_combined_seq)

            acceptor_probs, donor_probs = get_probs(new_seqs, good_contexts=good_contexts, context_seqs=context_seqs,
                                                    dont_use_contexts=args.dont_use_contexts)

            all_scores = {}
            for seq_no in range(args.n_seqs_per_it):
                score_contribution_array = \
                    make_score_contribution_array(transcript_structure_array,
                                                  donor_probs[seq_no, :],
                                                  acceptor_probs[seq_no, :],
                                                  target_const_donor=args.target_const_donor,
                                                  target_const_acceptor=args.target_const_acc,
                                                  target_cryptic_donor=args.target_cryptic_donor,
                                                  target_cryptic_acceptor=args.target_cryptic_acc,
                                                  target_alternative_donor=args.target_const_donor,
                                                  target_alternative_acceptor=args.target_const_acc,
                                                  ce_score_weight=args.ce_score_weight,
                                                  alt_score_weight=args.alt_weight,
                                                  alt_5p=args.alt_5p,
                                                  alt_3p=args.alt_3p)

                score = 2 - np.sum(score_contribution_array)
                all_scores[seq_no] = [score, score_contribution_array]

            # find the best sequence
            best_seq_no = max(all_scores, key=lambda k: all_scores[k][0])
            score = all_scores[best_seq_no][0]
            this_best_seq = new_seqs[best_seq_no]
            this_best_acceptor_prob = acceptor_probs[best_seq_no, :]  # note this is the probs across whole sequence
            this_best_donor_prob = donor_probs[best_seq_no, :]

            this_best_score_contribution_array = all_scores[best_seq_no][1]

            if i == 0 and args.track_splice_scores:
                print("writing")
                write_to_tracker(attempt, i, this_best_acceptor_prob, this_best_donor_prob, score, this_best_seq,
                                 splice_score_tracker_filename)

            if score > best_score + args.min_improvement:
                best_score_contribution_array = this_best_score_contribution_array

                if i > 0 and args.track_splice_scores:
                    write_to_tracker(attempt, i, this_best_acceptor_prob, this_best_donor_prob, score, new_combined_seq,
                                     splice_score_tracker_filename)
                best_score = score
                best_seq = this_best_seq
                best_donor_prob = this_best_donor_prob
                best_acceptor_prob = this_best_acceptor_prob
                bored = 0
                print(score)

            else:
                if bored > args.early_stop:
                    break

        results[attempt] = {'score': best_score, 'sequence': best_seq, 'donor': best_donor_prob,
                            'acceptor': best_acceptor_prob}

    with open(args.output, 'w') as file, open(args.output + ".predictions.csv", 'w') as file2:
        file.write("attempt,score,seq,ce_length,ce_frameshift\n")
        for key, value in results.items():
            file.write(str(key) + "," + str(value["score"]) + "," + str(value["sequence"]) + "," + \
                       str(int(args.ce_end - args.ce_start)) + "," + str((args.ce_end - args.ce_start) % 3 != 0) + "\n")

        file2.write("attempt,pos,donor,acceptor\n")
        for key, value in results.items():
            bruv = 0

            for d, a in zip(value["donor"], value["acceptor"]):
                file2.write(str(key) + "," + str(bruv) + "," + str(d) + "," + str(a) + "\n")
                bruv += 1


def write_to_tracker(attempt, i, acceptor_prob, donor_prob, score, new_combined_seq, splice_score_tracker_filename):
    """
	"attempt,iteration,position,donor_prob,acceptor_prob,score,sequence\n"
	"""
    sst = open(splice_score_tracker_filename, 'a')
    l = len(acceptor_prob)
    for p in range(l):
        to_write = ','.join([str(attempt), str(i), str(p), str(acceptor_prob[p]), str(donor_prob[p]), str(score)])

        if p == 0:
            to_write += "," + new_combined_seq + "\n"  # only write sequence on first position to save storage...
        else:
            to_write += ",\n"

        sst.write(to_write)
    sst.close()


if __name__ == "__main__":
    main()
