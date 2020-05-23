import tensorflow as tf


def get_loss(y_true, y_pred):
    """labels=labels,
       predictions=net["output"],
       loss_type="log_loss"
       scope="log_loss"
       huber_delta=1
    """
    shape = tf.shape(y_true)
    axes = tf.range(tf.shape(shape)[0] - 1)
    loss_1 = tf.compat.v1.losses.log_loss(labels=y_true, logits=y_pred)  # weights = 1
    nonzero = tf.reduce_sum(y_true, axis=axes) + 1e-9  # sum
    loss_1 = tf.reduce_sum(loss_1, axis=axes) / nonzero
    loss_1 = tf.reduce_mean(loss_1)
    loss_2 = tf.compat.v1.losses.log_loss(labels=y_true, logits=y_pred, weights=1 - y_true)
    nonzero = tf.reduce_sum(1 - y_true, axis=axes) + 1e-9
    loss_2 = tf.reduce_sum(loss_2, axis=axes) / nonzero
    loss_2 = tf.reduce_mean(loss_2)
    loss = (loss_1 + loss_2) / 2
    return loss


def dice_coefficient(y_true, y_pred, smooth=.1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    """
    y_true = tf.keras.layers.Flatten(y_true)
    y_pred = tf.keras.layers.Flatten(y_pred)
    intersection = tf.reduce_sum(tf.abs(y_true * y_pred))
    return (2. * intersection) / (tf.reduce_sum(tf.square(y_true)) + tf.reduce_sum(tf.square(y_pred)) + smooth)