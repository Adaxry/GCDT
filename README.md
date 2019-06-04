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

Set `test_type` to `testa` for evaluation and `testb` for testing.
Please note there is no evaluation set for the `chunking` task.


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

  For training efficiency, we firstly train a model for specified steps, and restore checkpoints for evaluation and testing.   For the CoNLL03, we compute the score on the test set at the best-performing checkpoints on the evaluation set. For the CoNLL2000, we compute the score on the test set directly.

+ How to get BERT embeddings?

  We provide a simple [tool](https://github.com/Adaxry/get_aligned_BERT_emb) to gennerate the BERT embedding for sequence labeling tasks. And then assign `bert_emb_path` with correct path and set `use_bert` to True in `train.sh`.


