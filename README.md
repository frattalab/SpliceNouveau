# SpliceNouveau

## Description

SpliceNouveau is a python algorithm to help users generate vectors containing introns and splice sites. At its core, it is an 'in silico directed evolution' algorithm: based on a set of user-defined sequences (which may be amino acid and/or nucleotide sequences) and constraints (splice-site strengths and types), it attempts to identify synonymous and non-coding mutations which result in the SpliceAI splicing predictions matching those requested by the user.

## Features

SpliceNouveau can generate several different types of splicing events:
- Cassette exons
- Alternative 3' splice sites
- Alternative 5' splice sites

In each case, it attempts to alter the sequence to set the SpliceAI predictions at the defined splice sites to the user-requested level, while minimizing the presence of off-target splice sites which could cause mis-splicing.

## Applications

In theory, SpliceNouveau can be used to create constitutively-spliced vectors. However, if the user specifies an enrichment of certain motifs that bind a given splicing regulator (or are known to influence splicing in, e.g., a tissue-specific manner) near a given splice site, then SpliceNouveau can help generate alternative spliced vectors. We have used this algorithm extensively to generate vectors which undergo alternative splicing in response to loss of TDP-43 nuclear function.

## Getting Started

Instructions on how to install and use SpliceNouveau will be provided here.
