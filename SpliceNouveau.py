import numpy as np
import pandas as pd
import gzip
import random
import argparse
from os.path import exists
import sys

# conda activate keras-gpu
# A test: python3 SpliceNouveau.py --aa MCCGMM --initial_cds generate_it --initial_intron1 GTcACNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNGACTAGCACTAGCAGAGACTAGCACGACGAGACTACGACACTAGCACTccAG --initial_intron2 GTAAaACGAAAGCTACGAaaacGaGCaTCaGCagaaAcgagcTAGAGCATCaaaaaGCCGcTcTcGaaAAAG --ce_start 3 --ce_end 5 --intron1_mut_chance 0.5 --skip_tf -o test

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


	def get_probs(input_sequences):

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

		return acceptor_prob, donor_prob
else:
	print("NOT LOADING TENSOR FLOW! THIS IS JUST FOR TESTS")


	def get_probs(input_sequence):
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
	parser.add_argument("--five_utr", default="TGA")
	parser.add_argument("--three_utr", default="")
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
	parser.add_argument("--ce_mut_chance", type=float, default=0,
	                    help="Chance per iteration that CE is mutated")
	parser.add_argument("--ce_mut_n", type=int, default=1,
	                    help="Number of mutations per iteration")
	parser.add_argument("--upstream_mut_chance", type=float, default=0,
	                    help="Chance per iteration that CE is mutated")
	parser.add_argument("--upstream_mut_n", type=int, default=1,
	                    help="Number of mutations per iteration")
	parser.add_argument("--downstream_mut_chance", type=float, default=0,
	                    help="Chance per iteration that CE is mutated")
	parser.add_argument("--downstream_mut_n", type=int, default=1,
	                    help="Number of mutations per iteration")
	parser.add_argument("--intron1_mut_chance", default=0, type=float)
	parser.add_argument("--intron2_mut_chance", default=0, type=float)
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


def mutate_all(prev_best_5utr, prev_best_cds, prev_best_intron1, prev_best_intron2, prev_best_3utr, args,
               to_mut_intron1, to_mut_intron2, no_mut=False):
	my_choice = random.choices([1, 2, 3, 4, 5], k=1, weights=[args.ce_mut_chance, args.upstream_mut_chance,
	                                                          args.downstream_mut_chance,
	                                                          args.intron1_mut_chance,
	                                                          args.intron2_mut_chance])[0]

	if no_mut:
		my_choice = None

	# initialise
	new_cds = prev_best_cds
	new_intron1 = prev_best_intron1
	new_intron2 = prev_best_intron2

	if my_choice == 1:
		new_cds = mut_cds(prev_best_cds, args.ce_mut_n, mut_start=args.ce_start, mut_end=args.ce_end)

	elif my_choice == 2:
		new_cds = mut_cds(prev_best_cds, args.upstream_mut_n, args.cds_mut_start_trim, args.ce_start - 1)

	elif my_choice == 3:
		assert len(
			new_cds) - 1 - args.cds_mut_end_trim >= args.ce_end + 1, "Need to tweak your mut_cds trims"
		new_cds = mut_cds(prev_best_cds, args.downstream_mut_n, args.ce_end + 1,
		                  len(prev_best_cds) - 1 - args.cds_mut_end_trim)

	elif my_choice == 4:
		new_intron1 = mut_noncoding(prev_best_intron1, to_mut_intron1, args.intron1_mut_n)

	elif my_choice == 5:
		new_intron2 = mut_noncoding(prev_best_intron2, to_mut_intron2, args.intron2_mut_n)

	new_combined_seq = prev_best_5utr + new_cds[0:args.ce_start] + new_intron1 + \
	                   new_cds[args.ce_start:args.ce_end] + new_intron2 + new_cds[args.ce_end:] + prev_best_3utr

	separate_parts_d = {"utr5": prev_best_5utr, "cds": new_cds, "intron1": new_intron1, "intron2": new_intron2, "utr3": prev_best_3utr}

	return new_combined_seq, separate_parts_d


def main():
	print("Reading arguments")
	args = get_args()

	# Find which positions can be mutated
	to_mut_intron1 = [i for i, a in enumerate(list(args.initial_intron1)) if a.islower() or a.upper() in ["Y", "N"]]
	to_mut_intron2 = [i for i, a in enumerate(list(args.initial_intron2)) if a.islower() or a.upper() in ["Y", "N"]]

	# Find positions of the splice sites
	intron1_start = len(args.five_utr) + args.ce_start
	intron1_end = intron1_start + len(args.initial_intron1)
	intron2_start = intron1_end - args.ce_start + args.ce_end
	intron2_end = intron2_start + len(args.initial_intron2)

	# find positions where we don't want splice sites
	total_l = len(args.five_utr) + len(args.initial_cds) + len(args.three_utr) + len(args.initial_intron1) + \
	          len(args.initial_intron2)

	print(total_l)
	print(total_l - args.ignore_end)

	dont_want_ss = [i for i in range(args.ignore_start, total_l - args.ignore_end) if i \
	                not in [intron1_start, intron1_end, intron2_start, intron2_end]]

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
			assert len(
				args.initial_cds) - 1 - args.cds_mut_end_trim >= args.ce_end + 1, "Need to tweak your mut_cds trims"
			args.initial_cds = mut_cds(args.initial_cds, 1000, args.ce_end + 1,
			                           len(args.initial_cds) - 1 - args.cds_mut_end_trim, pptness=True)

		for i in range(args.n_iterations_per_attempt):
			bored += 1

			if i == 0:
				prev_best_5utr = starting_5utr
				prev_best_intron1 = starting_intron1
				prev_best_intron2 = starting_intron2
				prev_best_3utr = starting_3utr
				prev_best_cds = args.initial_cds

			new_seqs_ds = []
			for j in range(args.n_seqs_per_it):
				if i > 0:
					new_combined_seq, separate_parts_d = mutate_all(prev_best_5utr, prev_best_cds, prev_best_intron1, prev_best_intron2,
					                              prev_best_3utr, args, to_mut_intron1, to_mut_intron2)

				else:
					new_combined_seq, separate_parts_d = mutate_all(prev_best_5utr, prev_best_cds, prev_best_intron1, prev_best_intron2,
					                              prev_best_3utr, args, to_mut_intron1, to_mut_intron2, no_mut=True)

				new_seqs_ds.append({"seq": new_combined_seq, "separate_parts_d": separate_parts_d})

			acceptor_probs, donor_probs = get_probs([a["seq"] for a in new_seqs_ds])

			# Is it good??
			all_scores = {}
			for seq_no in range(args.n_seqs_per_it):
				acceptor_prob = acceptor_probs[seq_no, :]
				donor_prob = donor_probs[seq_no, :]  # TODO dims may be wrong

				if not args.one_intron:  # two introns...
					const_donor = donor_prob[intron1_start - 1]
					ce_acceptor = acceptor_prob[intron1_end]
					ce_donor = donor_prob[intron2_start - 1]
					const_acceptor = acceptor_prob[intron2_end]

					score = 2 - abs(const_donor - args.target_const_donor) - abs(
						const_acceptor - args.target_const_acc) - \
					        abs(ce_donor - args.target_cryptic_donor) * args.ce_score_weight - abs(
						ce_acceptor - args.target_cryptic_acc) * args.ce_score_weight

					bad_acc = max([acceptor_prob[a] for a in dont_want_ss])
					bad_don = max([donor_prob[a - 1] for a in dont_want_ss])
					score += -2 * bad_acc
					score += -2 * bad_don

				else:  # only one intron
					if args.ir:
						ir_donor = donor_prob[intron1_start - 1]
						ir_acceptor = acceptor_prob[intron1_end]

						score = 2 - abs(ir_donor - args.target_const_donor) - abs(ir_acceptor - args.target_const_acc)
						bad_acc = max([acceptor_prob[a] for a in dont_want_ss])
						bad_don = max([donor_prob[a - 1] for a in dont_want_ss])
						score += -2 * bad_acc
						score += -2 * bad_don

					elif args.alt_5p:
						cryptic_donor = donor_prob[intron1_start - 1]
						const_acceptor = acceptor_prob[intron1_end]

						# Alternative, i.e. constitutive, donor will be somewhere else
						if args.alt_position == "in_intron":
							valid_positions = [k for k in range(intron1_start, intron1_end - args.min_intron_l)]
						elif args.alt_position == "in_exon":
							valid_positions = [k for k in
							                   range(args.alt_5p_start_trim, intron1_start - args.min_alt_dist)]

						const_donor_pos = \
							[k for k in valid_positions if
							 donor_prob[k] == max([donor_prob[j] for j in valid_positions])][
								0]
						const_donor = donor_prob[const_donor_pos]

						score = 2 - abs(const_donor - args.target_const_donor) * args.alt_weight - abs(
							const_acceptor - args.target_const_acc)
						score += -abs(cryptic_donor - args.target_cryptic_donor) * args.ce_score_weight

						bad_acc = max([acceptor_prob[a] for a in dont_want_ss])
						bad_don = max([donor_prob[a - 1] for a in dont_want_ss if a != (const_donor_pos + 1)])
						score += -2 * bad_acc
						score += -2 * bad_don


					elif args.alt_3p:
						const_donor = donor_prob[intron1_start - 1]
						cryptic_acceptor = acceptor_prob[intron1_end]

						# Alternative, i.e. constitutive, donor will be somewhere else
						if args.alt_position == "in_intron":
							valid_positions = [k for k in range(intron1_start + args.min_intron_l, intron1_end)]
						elif args.alt_position == "in_exon":
							valid_positions = [k for k in
							                   range(intron1_end + args.min_alt_dist, total_l - args.alt_3p_end_trim)]

						const_acceptor = max([acceptor_prob[j] for j in valid_positions])
						const_acceptor_pos = [k for k in valid_positions if
						                      acceptor_prob[k] == max([acceptor_prob[j] for j in valid_positions])][0]
						const_acceptor = acceptor_prob[const_acceptor_pos]

						score = 2 - abs(const_donor - args.target_const_donor) - abs(
							const_acceptor - args.target_const_acc) * args.alt_weight
						score += -abs(cryptic_acceptor - args.target_cryptic_acc) * args.ce_score_weight

						bad_acc = max([acceptor_prob[a] for a in dont_want_ss if a != const_acceptor_pos])
						bad_don = max([donor_prob[a - 1] for a in dont_want_ss])
						score += -2 * bad_acc
						score += -2 * bad_don

				all_scores[seq_no] = score

			# find the best sequence
			best_seq_no = max(all_scores, key=lambda k: all_scores[k])
			score = all_scores[best_seq_no]
			new_combined_seq = new_seqs_ds[best_seq_no]["seq"]


			# print(' '.join([str(const_donor), str(ce_acceptor), str(ce_donor), str(const_acceptor), str(bad_acc), str(bad_don)]))

			if i == 0 and args.track_splice_scores:
				print("writing")
				write_to_tracker(attempt, i, acceptor_prob, donor_prob, score, new_combined_seq,
				                 splice_score_tracker_filename)

			if score > best_score + args.min_improvement:
				prev_best_5utr = new_seqs_ds[best_seq_no]["separate_parts_d"]["utr5"]
				prev_best_intron1 = new_seqs_ds[best_seq_no]["separate_parts_d"]["intron1"]
				prev_best_intron2 = new_seqs_ds[best_seq_no]["separate_parts_d"]["intron2"]
				prev_best_3utr = new_seqs_ds[best_seq_no]["separate_parts_d"]["utr3"]
				prev_best_cds = new_seqs_ds[best_seq_no]["separate_parts_d"]["cds"]



				if i > 0 and args.track_splice_scores:
					write_to_tracker(attempt, i, acceptor_prob, donor_prob, score, new_combined_seq,
					                 splice_score_tracker_filename)
				best_score = score
				best_seq = new_combined_seq
				best_donor_prob = donor_prob  # TODO FIX
				best_acceptor_prob = acceptor_prob # TODO FIX
				bored = 0
				print(score)
				if args.alt_5p:
					print(' '.join(
						[str(const_donor), str(cryptic_donor), str(const_acceptor), str(bad_acc), str(bad_don)]))
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
