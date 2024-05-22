# SpliceNouveau

## Description

SpliceNouveau is a python algorithm to help users generate vectors containing introns and splice sites. At its core, it is an 'in silico directed evolution' algorithm: based on a set of user-defined sequences (which may be amino acid and/or nucleotide sequences) and constraints (splice-site strengths and types), it attempts to identify synonymous and non-coding mutations which result in the SpliceAI splicing predictions matching those requested by the user.

## Features

SpliceNouveau can generate several different types of splicing events:
- Cassette exons
- Alternative 3' splice sites
- Alternative 5' splice sites
- Intron retention

In each case, it attempts to alter the sequence to set the SpliceAI predictions at the defined splice sites to the user-requested level, while minimizing the presence of off-target splice sites which could cause mis-splicing.

## Applications

In theory, SpliceNouveau can be used to create constitutively-spliced vectors. However, if the user specifies an enrichment of certain motifs that bind a given splicing regulator (or are known to influence splicing in, e.g., a tissue-specific manner) near a given splice site, then SpliceNouveau can help generate alternative spliced vectors. We have used this algorithm extensively to generate vectors which undergo alternative splicing in response to loss of TDP-43 nuclear function.

Sure, here's a 'Getting Started' user guide for the `SpliceNouveau` tool:

# Getting Started with SpliceNouveau

SpliceNouveau is a Python algorithm designed to help users generate vectors containing introns and splice sites. It is an 'in silico directed evolution' algorithm that identifies synonymous and non-coding mutations to match the SpliceAI splicing predictions requested by the user.

## Prerequisites

- Python 3.x
- Required Python packages: `numpy`, `pandas`, `gzip`, `random`, `argparse`, `os`, `sys`
- Optional: `keras`, `tensorflow`, `spliceai` (for running predictions locally)

## Installation

1. Clone the repository or download the source code.
2. Install the required Python packages using pip:

```
pip install numpy pandas gzip
```

3. Install SpliceAI by following the instructions on the SpliceAI github page. Note that although a CUDA-enabled GPU significantly increases performance, it is feasible to run SpliceAI/SpliceNouveau using a CPU. 

## Usage

The `SpliceNouveau` tool is executed from the command line with various arguments. Here's an example usage:

```
python3 SpliceNouveau.py --initial_cds ATGGCGAGAACAATGGTTGCTATGGTGTCCAAAGGTGAGGCAGTCATAAAG... \
                         --initial_intron1 GTAAGNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTGTGTGTG... \
                         --ce_start 43 \
                         --ce_end 159 \
                         --ce_mut_chance 1 \
                         --five_utr CGGCCGCTTCTTGGTGCCAGCTTATCAT... \
                         --three_utr TGATAAACAAATGGTAAGGAAGGGCACAT... \
                         --ignore_end 470 \
                         --output mscar/aars1_inspired_closer_aim_0p5.csv \
                         --target_cryptic_donor 0.5 \
                         --target_cryptic_acc 0.5 \
                         -a 5 \
                         --intron1_mut_chance 0.5 \
                         --intron2_mut_chance 0.5 \
                         -n 2000 \
                         --cds_mut_end_trim 569
```

This example command specifies the initial CDS, intron sequences, cryptic exon start and end positions, UTR sequences, splice site target scores, and other parameters for the algorithm.

For a complete list of available arguments and their descriptions, run:

```
python3 SpliceNouveau.py -h
```

## Output

The `SpliceNouveau` tool generates two output files:

1. `<output_filename>.csv`: Contains the attempt number, score, sequence, cryptic exon length, and frameshift information for each attempt.
2. `<output_filename>.predictions.csv`: Contains the attempt number, position, donor probability, and acceptor probability for each position in the sequence.

If the `--track_splice_scores` option is used, an additional file `<output_filename>.tracked_scores.csv` is generated, which tracks the splice scores for each iteration.
