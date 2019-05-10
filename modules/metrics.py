from keras import backend as K
from keras.layers.core import Lambda
CTC_LOSS_STR='ctc_loss'
# Actual ctc loss calculation
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # From Keras example image_ocr.py:
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def CTC(args, name=CTC_LOSS_STR):
	return Lambda(ctc_lambda_func, output_shape=(1,), name=name)(args)

# We are masking the loss calculated for padding
def vspnet_loss(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)
    