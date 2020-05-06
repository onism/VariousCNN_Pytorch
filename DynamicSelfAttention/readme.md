ref: https://arxiv.org/pdf/1808.07383.pdf

# Dynamic Self-Attention: Computing Attention over Words Dynamically for Sentence Embedding

## Abstract

In this paper, we propose DSA, a new self-attention mechanism for sentence embedding. **We design DSA by modifying dynamic routing in capsule network for NLP.**


## Introduction

Self-attention computes attention weights by the inner product between words and the learnable weight vector.

## DSA

DSA is stacked on CNN with Dense connection, computes attention weights over words.

### CNN with Dense Connection
The goal of this module is to encode each word into a meaningful representation space while capturing local information.

### DSA
DSA iteratively computes attention weights over words with the dynamic weight vector, which varies with inputs.

