import tensorflow as tf


def get_loss(labels, predictions, loss_type, scope=None, **kwargs):
    """ Calculates a compensated loss for all 11 labels.
      The losses available are:
        * absolute_difference
        * mean_squared_error
        * log_loss
        * huber_loss - requires huber_delta
    """
    '''labels=labels,
        predictions=net["output"],
        loss_type="log_loss"
        scope="log_loss"
        huber_delta=1
    '''
    if loss_type == "absolute_difference":
        loss_func = lambda x, y, z: tf.compat.v1.losses.absolute_difference(labels=x,
                                                                            predictions=y,
                                                                            weights=z,
                                                                            reduction=tf.losses.Reduction.NONE)
    elif loss_type == "mean_squared_error":
        loss_func = lambda x, y, z: tf.compat.v1.losses.mean_squared_error(labels=x,
                                                                           predictions=y,
                                                                           weights=z,
                                                                           reduction=tf.losses.Reduction.NONE)
    elif loss_type == "log_loss":
        loss_func = lambda x, y, z: tf.compat.v1.losses.log_loss(labels=x,
                                                                 predictions=y,
                                                                 weights=z,
                                                                 reduction=tf.losses.Reduction.NONE)
    elif loss_type == "huber_loss":
        loss_func = lambda x, y, z: tf.compat.v1.losses.huber_loss(labels=x,
                                                                   predictions=y,
                                                                   weights=z,
                                                                   delta=kwargs["huber_delta"],
                                                                   reduction=tf.losses.Reduction.NONE)
    else:
        print("*" * 20)
        print("Not valid loss function was defined")
        return tf.zeros((1,))

    shape = tf.shape(labels)
    axes = tf.range(tf.shape(shape)[0] - 1)
    loss_1 = loss_func(labels, predictions, labels)
    nonzero = tf.reduce_sum(labels, axis=axes) + 1e-9
    loss_1 = tf.reduce_sum(loss_1, axis=axes) / nonzero
    loss_1 = tf.reduce_mean(loss_1)
    loss_2 = loss_func(labels, predictions, 1 - labels)
    nonzero = tf.reduce_sum(1 - labels, axis=axes) + 1e-9
    loss_2 = tf.reduce_sum(loss_2, axis=axes) / nonzero
    loss_2 = tf.reduce_mean(loss_2)
    loss = (loss_1 + loss_2) / 2
    return loss
