# GCDT: A Global Context Enhanced Deep Transition Architecture for Sequence Labeling

## Contents
* [Introduction](#introduction)
* [Usage](#usage)
* [Requirements](#requirements)
* [Citation](#citation)
* [FAQ](#faq)

## Introduction

The code of our proposed GCDT, which deepens the state transition path at each position in a sentence, and further assign every token with a global representation learned from the entire sentence. \[[paper](https://arxiv.org/abs/undone)\]. The implementation is based on [THUMT](https://github.com/thumt/THUMT).

## Usage

+ Training

```
sh train.sh task_name
```

`task_name` is the name of tasks between `ner` and `chunking`.

+ Evaluation and Testing

```
sh test.sh task_name test_type
```

`task_name` is same as above, and `test_type` indicates `testa` or `testb`.		
Notice to choose `testa` for evaluation before the `testb` set.


## Requirements

+ tensorflow 1.2 
+ python 3.5 

## Citation

Please cite the following paper if you use the code:

```
@InProceedings{Liu:19,
  author    = {Yijin Liu, Fandong Meng, Jinchao Zhang, Jinan Xu, Yufeng Chen and Jie Zhou},
  title     = {GCDT: A Global Context Enhanced Deep Transition Architecture for Sequence Labeling},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  year      = {2019}
}
```

## FAQ

+ Why not evaluate along with training?

  For training efficiency, we firstly train a model for specified steps, and restore checkpoints for evaluation and testing.   Finally, we report the result on the test set at the best-performing checkpoints on the evaluation set.

+ How to get BERT embeddings?

  Since the WordPiece adopted in [BERT](https://github.com/google-research/bert#using-bert-to-extract-fixed-feature-vectors-like-elmo) may cut a word into pieces, we investigate three strategies (`first`, `mean` and `max`) to maintain alignments between input tokens and their corresponding labels. We provide a simple [tools](undone) to gennerate the BERT embedding with specified layer and aggregation strategy.


