# coding=utf-8
# Copyright 2018 The THUMT Authors


import math
import operator
import functools
import numpy as np
import tensorflow as tf
from pathlib import Path


def batch_examples(example, batch_size, max_length, mantissa_bits,
                   shard_multiplier=1, length_multiplier=1, constant=False,
                   num_threads=4, drop_long_sequences=True):
    """ Batch examples

    :param example: A dictionary of <feature name, Tensor>.
    :param batch_size: The number of tokens or sentences in a batch
    :param max_length: The maximum length of a example to keep
    :param mantissa_bits: An integer
    :param shard_multiplier: an integer increasing the batch_size to suit
        splitting across data shards.  shard_multiplier=len(params.device_list)??
    :param length_multiplier: an integer multiplier that is used to
        increase the batch sizes and sequence length tolerance.
    :param constant: Whether to use constant batch size
    :param num_threads: Number of threads
    :param drop_long_sequences: Whether to drop long sequences

    :returns: A dictionary of batched examples
    """

    with tf.name_scope("batch_examples"):
        max_length = max_length or batch_size
        min_length = 8
        mantissa_bits = mantissa_bits

        # Compute boundaries
        x = min_length
        boundaries = []

        # add stages [8, 10, ..., 112]
        while x < max_length:
            boundaries.append(x)
            x += 2 ** max(0, int(math.log(x, 2)) - mantissa_bits)

        # Whether the batch size is constant
        if not constant:
            batch_sizes = [max(1, batch_size // length)
                           for length in boundaries + [max_length]]
            # [512, 409, ..., 32]                           
            batch_sizes = [b * shard_multiplier for b in batch_sizes]
            bucket_capacities = [2 * b for b in batch_sizes]
            # [1024, ..., 64]
        else:
            batch_sizes = batch_size * shard_multiplier
            bucket_capacities = [2 * n for n in boundaries + [max_length]]

        # length_multiplier default is 1
        max_length *= length_multiplier
        boundaries = [boundary * length_multiplier for boundary in boundaries]
        max_length = max_length if drop_long_sequences else 10 ** 9

        # The queue to bucket on will be chosen based on *maximum length*
        max_example_length = 0
        for v in list(example.values()):
            if v.shape.ndims > 0:
                seq_length = tf.shape(v)[0]
                max_example_length = tf.maximum(max_example_length, seq_length)

        (_, outputs) = tf.contrib.training.bucket_by_sequence_length(
            max_example_length,  # detemine which bucket
            example,  # representing a single element to bucket
            batch_sizes,
            [b + 1 for b in boundaries],
            num_threads=num_threads,
            capacity=2,  # Number of full batches to store, we don't need many.
            bucket_capacities=bucket_capacities,  # one larger than bucket_boundaries
            dynamic_pad=True,  # How to specify pad symbol ?? 
            keep_input=(max_example_length <= max_length) # whether the input example is added to the queue or not??
        )
        # A tuple (sequence_length, outputs) where sequence_length is a 1-D Tensor
        #  of size batch_size and outputs is a list or dictionary of batched, bucketed

    return outputs  


def get_training_input(filenames, params):
    """ Get input for training stage

    :param filenames: A list contains [source_filename, target_filename]
    :param params: Hyper-parameters

    :returns: A dictionary of pair <Key, Tensor>
    """

    with tf.device("/cpu:0"):
        def parse_fn(line_words, line_tags):
            # Encode in Bytes for TF
            words = [w.encode() for w in line_words.strip().split()] + ["<eos>".encode()]
            tags = [t.encode() for t in line_tags.strip().split()] + ["<eos>".encode()]
            assert len(words) == len(tags) 
        
            # Chars
            #   just padd chars to the same length in one sentence 
            #   chars will be padd into the samme length in one batch at "bucket_by_sequence_length"
            chars = [[c.encode() for c in w] + ["<eos>".encode()] for w in line_words.strip().split()]
            chars.append(["<eos>".encode()])
            char_lengths = [len(c) for c in chars]
            chars = [c + [params.pad.encode()] * (max(char_lengths) - l) for c, l in zip(chars, char_lengths)]
            return (words, chars, char_lengths), tags


        def generator_fn(words, tags):
            with Path(words).open('r') as f_words, Path(tags).open('r') as f_tags:
                for line_words, line_tags in zip(f_words, f_tags):
                    yield parse_fn(line_words, line_tags)
        

        shapes = (([None], [None, None], [None]), # words, chars, char_lengths
                  [None])                         # tags              

        types = ((tf.string, tf.string, tf.int32),
                 tf.string)

        dataset = tf.data.Dataset.from_generator(
            functools.partial(
                generator_fn,
                filenames[0],
                filenames[1]
            ),
            output_shapes=shapes,
            output_types=types
        )
        dataset = dataset.shuffle(params.buffer_size)
        dataset = dataset.repeat() # ??? how much times

        # Convert to dictionary
        dataset = dataset.map(
            lambda src, tgt: {
                "source": src[0],
                "target": tgt,
                "chars": src[1],
                "source_length": tf.shape(src[0]),
                "target_length": tf.shape(tgt),
                "char_length": src[2],
            },
            num_parallel_calls=params.num_threads
        )

        # Create iterator
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()  # get one ???

        # Create lookup table
        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["source"]),
            default_value=params.mapping["source"][params.unk] # index of unk
        )
        tgt_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["target"]),
            default_value=0
        )
        char_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["char"]),
            default_value=params.mapping["char"][params.unk]
        )
        # String to index lookup
        features["source"] = src_table.lookup(features["source"])
        features["target"] = tgt_table.lookup(features["target"])
        features["chars"] = char_table.lookup(features["chars"])

        # Batching according the sentence length
        # NOTE: char has been padded ! But it does't has infuence on sequence length
        features = batch_examples(features, params.batch_size,
                                  params.max_length, params.mantissa_bits,
                                  shard_multiplier=len(params.device_list),
                                  length_multiplier=params.length_multiplier,
                                  constant=params.constant_batch_size,
                                  num_threads=params.num_threads)

        # Convert to int32
        features["source"] = tf.to_int32(features["source"])
        features["target"] = tf.to_int32(features["target"])
        features["source_length"] = tf.to_int32(features["source_length"])
        features["target_length"] = tf.to_int32(features["target_length"])
        features["char_length"] = tf.to_int32(features["char_length"])
        # Removes dimensions of size 1 from the shape of a tensor
        features["source_length"] = tf.squeeze(features["source_length"], 1)
        features["target_length"] = tf.squeeze(features["target_length"], 1)

        return features


def get_training_input_with_bert(filenames, params):
    """ Get input for training stage

    :param filenames: A list contains [source_filename, target_filename, bert_embding_file]
    :param params: Hyper-parameters

    :returns: A dictionary of pair <Key, Tensor>
    """

    with tf.device("/cpu:0"):
        def parse_fn(line_words, line_tags, line_berts):
            # Encode in Bytes for TF
            words = [w.encode() for w in line_words.strip().split()] + ["<eos>".encode()]
            tags = [t.encode() for t in line_tags.strip().split()] + ["<eos>".encode()]
            berts = [[float(value) for value in t.split()] for t in line_berts.strip().split("|||")] + [[0.0] * params.bert_size] 
            assert len(words) == len(tags) 
            assert len(words) == len(berts) 
            assert len(berts[0]) == params.bert_size
        
            # Chars
            #   just padd chars to the same length in one sentence 
            #   chars will be padd into the samme length in one batch at "bucket_by_sequence_length"

            chars = [[c.encode() for c in w] + ["<eos>".encode()] for w in line_words.strip().split()]
            chars.append(["<eos>".encode()])
            char_lengths = [len(c) for c in chars]
            chars = [c + [params.pad.encode()] * (max(char_lengths) - l) for c, l in zip(chars, char_lengths)]
            return (words, chars, char_lengths, berts), tags


        def generator_fn(words, tags, berts):
            with Path(words).open('r') as f_words, Path(tags).open('r') as f_tags, Path(berts).open('r') as f_berts:
                for line_words, line_tags, line_berts in zip(f_words, f_tags, f_berts):
                    yield parse_fn(line_words, line_tags, line_berts)
        

        shapes = (([None], [None, None], [None], [None, params.bert_size]), # words, chars, char_lengths, berts
                  [None])                         # tags              

        types = ((tf.string, tf.string, tf.int32, tf.float32),
                 tf.string)

        dataset = tf.data.Dataset.from_generator(
            functools.partial(
                generator_fn,
                filenames[0],
                filenames[1],
                filenames[2]
            ),
            output_shapes=shapes,
            output_types=types
        )

        # dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
        dataset = dataset.shuffle(params.buffer_size)
        dataset = dataset.repeat() 


        # Convert to dictionary
        dataset = dataset.map(
            lambda src, tgt: {
                "source": src[0],
                "target": tgt,
                "chars": src[1],
                "source_length": tf.shape(src[0]),
                "target_length": tf.shape(tgt),
                "char_length": src[2],
                "bert": src[3]
            },
            num_parallel_calls=params.num_threads
        )

        # Create iterator
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()  # get one ???
        # Create lookup table
        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["source"]),
            default_value=params.mapping["source"][params.unk] # index of unk
        )
        tgt_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["target"]),
            default_value=0
        )
        char_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["char"]),
            default_value=params.mapping["char"][params.unk]
        )
        # String to index lookup
        features["source"] = src_table.lookup(features["source"])
        features["target"] = tgt_table.lookup(features["target"])
        features["chars"] = char_table.lookup(features["chars"])

        # Batching according the sentence length
        # NOTE: char has been padded ! But it does't has infuence on sequence length
        features = batch_examples(features, params.batch_size,
                                  params.max_length, params.mantissa_bits,
                                  shard_multiplier=len(params.device_list),
                                  length_multiplier=params.length_multiplier,
                                  constant=params.constant_batch_size,
                                  num_threads=params.num_threads)

        # Convert to int32
        features["source"] = tf.to_int32(features["source"])
        features["target"] = tf.to_int32(features["target"])
        features["source_length"] = tf.to_int32(features["source_length"])
        features["target_length"] = tf.to_int32(features["target_length"])
        features["char_length"] = tf.to_int32(features["char_length"])
        # Removes dimensions of size 1 from the shape of a tensor
        features["source_length"] = tf.squeeze(features["source_length"], 1)
        features["target_length"] = tf.squeeze(features["target_length"], 1)

        return features



def sort_input_file(filename, reverse=True):
    # Read file
    with tf.gfile.Open(filename) as fd:
        inputs = [line.strip() for line in fd]

    input_lens = [
        (i, len(line.strip().split())) for i, line in enumerate(inputs)
    ]

    sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1),
                               reverse=reverse)
    sorted_keys = {}
    sorted_inputs = []

    for i, (index, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[index])
        sorted_keys[index] = i

    return sorted_keys, sorted_inputs


def sort_and_zip_files(names):
    inputs = []
    input_lens = []
    files = [tf.gfile.GFile(name) for name in names]

    count = 0

    for lines in zip(*files):
        lines = [line.strip() for line in lines] # one [src , tar]

        input_lens.append((count, len(lines[0].split())))
        inputs.append(lines)
        count += 1

    # Close files
    for fd in files:
        fd.close()
    # sort by sentence length
    sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1),
                               reverse=True)
    sorted_inputs = []

    for i, (index, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[index])

    return [list(x) for x in zip(*sorted_inputs)] # [[src...] [tar...]




def get_evaluation_input(filenames, params):
    pass



def get_inference_input(filename, params):  # unchange for char !!

    def parse_fn(line):
        # Encode in Bytes for TF
        words = [w.encode() for w in line.strip().split()] + ["<eos>".encode()]
        
        # Chars
        chars = [[c.encode() for c in w] + ["<eos>".encode()] for w in line.strip().split()]
        chars.append(["<eos>".encode()])
        char_lengths = [len(c) for c in chars]
        chars = [c + [params.pad.encode()] * (max(char_lengths) - l) for c, l in zip(chars, char_lengths)]
        return words, chars, char_lengths


    def generator_fn(filename):
        with Path(filename).open('r') as src_f:
            for line in src_f:
                yield parse_fn(line)
        

    shapes = ([None], [None, None], [None]) # words, chars, char_lengths

    types = (tf.string, tf.string, tf.int32)

    dataset = tf.data.Dataset.from_generator(
        functools.partial(
            generator_fn,
            filename[0]
        ),
        output_shapes=shapes,
        output_types=types
    )

    #dataset = dataset.shuffle(params.buffer_size)

    # Convert to dictionary
    dataset = dataset.map(
        lambda words, chars, char_length: {
            "source": words,
            "source_length": tf.shape(words)[0],
            "chars": chars,
            "char_length": char_length
        },
        num_parallel_calls=params.num_threads
    )


    dataset = dataset.padded_batch(
        params.decode_batch_size,
        {
            "source": [None],
            "source_length": [],
            "chars" : [None, None],
            "char_length" : [None]
        },
        {
            "source": params.pad,
            "source_length": 0,
            "chars" : params.pad,
            "char_length" : 0
        }
    )

    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()

    src_table = tf.contrib.lookup.index_table_from_tensor(
        tf.constant(params.vocabulary["source"]),
        default_value=params.mapping["source"][params.unk]
    )
    char_table = tf.contrib.lookup.index_table_from_tensor(
        tf.constant(params.vocabulary["char"]),
        default_value=params.mapping["char"][params.unk]
    )    


    features["source"] = src_table.lookup(features["source"])
    features["chars"] = char_table.lookup(features["chars"])

    return features

def get_inference_input_with_bert(filenames, params):  

    def parse_fn(line_words, line_berts):
        # Encode in Bytes for TF
        words = [w.encode() for w in line_words.strip().split()] + ["<eos>".encode()]
        berts = [[float(value) for value in t.split()] for t in line_berts.strip().split("|||")] + [[0.0] * params.bert_size]
        assert len(words) == len(berts)
        assert len(berts[0]) == params.bert_size
        
        # Chars
        chars = [[c.encode() for c in w] + ["<eos>".encode()] for w in line_words.strip().split()]
        chars.append(["<eos>".encode()])
        char_lengths = [len(c) for c in chars]
        chars = [c + [params.pad.encode()] * (max(char_lengths) - l) for c, l in zip(chars, char_lengths)]
        return words, chars, char_lengths, berts


    def generator_fn(words, berts):
        with Path(words).open('r') as f_words, Path(berts).open("r") as f_berts:
            for line_words, line_berts in zip(f_words, f_berts):
                yield parse_fn(line_words, line_berts)

    shapes = ([None], [None, None], [None], [None, params.bert_size]) # words, chars, char_lengths

    types = (tf.string, tf.string, tf.int32, tf.float32)

    dataset = tf.data.Dataset.from_generator(
        functools.partial(
            generator_fn,
            filenames[0],
            filenames[1]
        ),
        output_shapes=shapes,
        output_types=types
    )

    # Convert to dictionary
    dataset = dataset.map(
        lambda words, chars, char_length, berts: {
            "source": words,
            "source_length": tf.shape(words)[0],
            "chars": chars,
            "char_length": char_length,
            "bert": berts
        },
        num_parallel_calls=params.num_threads
    )


    dataset = dataset.padded_batch(
        params.decode_batch_size,
        {
            "source": [None],
            "source_length": [],
            "chars" : [None, None],
            "char_length" : [None],
            "bert": [None, params.bert_size]
        },
        {
            "source": params.pad,
            "source_length": 0,
            "chars" : params.pad,
            "char_length" : 0,
            "bert": 0.0
        }
    )

    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()

    src_table = tf.contrib.lookup.index_table_from_tensor(
        tf.constant(params.vocabulary["source"]),
        default_value=params.mapping["source"][params.unk]
    )
    char_table = tf.contrib.lookup.index_table_from_tensor(
        tf.constant(params.vocabulary["char"]),
        default_value=params.mapping["char"][params.unk]
    )    


    features["source"] = src_table.lookup(features["source"])
    features["chars"] = char_table.lookup(features["chars"])

    return features
