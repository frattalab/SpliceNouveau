# SpliceNouveau

## Description

SpliceNouveau is a Python algorithm designed to assist users in generating vectors containing introns and splice sites. At its core, it is an 'in silico directed evolution' algorithm. Based on a set of user-defined sequences (which may be amino acid and/or nucleotide sequences), it attempts to identify synonymous and non-coding mutations that result in the SpliceAI splicing predictions matching those requested by the user.

## Features

- Generates several different types of splicing events:
 - Cassette exons
 - Alternative 3' splice sites
 - Alternative 5' splice sites
- Alters the sequence to set the SpliceAI predictions at the defined splice sites to the user-requested level
- Minimizes the presence of off-target splice sites that could cause mis-splicing

## Applications

- Can be used to create constitutively-spliced vectors
- By specifying an enrichment of certain motifs that bind a given splicing regulator (or are known to influence splicing in, e.g., a tissue-specific manner) near a given splice site, SpliceNouveau can help generate alternative spliced vectors
- Extensively used to generate vectors that undergo alternative splicing in response to loss of TDP-43 nuclear function

## Getting Started

Instructions on how to install and use SpliceNouveau will be provided here.

