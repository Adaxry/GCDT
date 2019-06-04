# coding=utf-8
# Copyright 2018 The THUMT Authors




import types
import numpy as np
import tensorflow as tf

def load_glove(glove_path):
    with open(glove_path, "r", encoding="utf-8") as glove_f:
        all_vectors = []
        for line in glove_f:
            try:
                vectors = [float(word) for word in line.strip().split()[1:]]
                all_vectors.append(vectors)
                assert len(vectors) == 300

            except Exception as e:
                print("Warning : incomplete glove vector!")
                print(line.strip().split())
        return np.asarray(all_vectors, dtype=np.float32)



def session_run(monitored_session, args):
    # Call raw TF session directly
    return monitored_session._tf_sess().run(args)


def zero_variables(variables, name=None):
    ops = []

    for var in variables:
        with tf.device(var.device):
            op = var.assign(tf.zeros(var.shape.as_list()))
        ops.append(op)

    return tf.group(*ops, name=name or "zero_op")


def replicate_variables(variables, device=None):
    new_vars = []

    for var in variables:
        device = device or var.device
        with tf.device(device):
            name = "replicate/" + var.name.split(":")[0]
            new_vars.append(tf.Variable(tf.zeros(var.shape.as_list()),
                                        name=name, trainable=False))

    return new_vars


def collect_gradients(gradients, variables):
    ops = []

    for grad, var in zip(gradients, variables):
        if isinstance(grad, tf.Tensor):
            ops.append(tf.assign_add(var, grad))
        elif isinstance(grad, tf.IndexedSlices):
            ops.append(tf.scatter_add(var, grad.indices, grad.values))
        else:
            print("grad : ", grad, " with type : ", type(grad)) 
    return tf.group(*ops)


def scale_gradients(grads_and_vars, scale):
    scaled_gradients = []
    variables = []

    for grad, var in gradients:
        if isinstance(grad, tf.IndexedSlices):
            slices = tf.IndexedSlices(scale * grad.values, grad.indices)
            scaled_gradients.append(slices)
            variables.append(var)
        elif isinstance(grad, tf.Tensor):
            scaled_gradients.append(scale * grad)
            variables.append(var)
        else:
            pass
        print("grad : ", grad, " with type : ", type(grad))      
 
    return scaled_gradients, variables
