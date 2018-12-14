from keras import backend as K
from keras.layers.core import Lambda

# Actual ctc loss calculation
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # From Keras example image_ocr.py:
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def CTC(args, name='ctc'):
	return Lambda(ctc_lambda_func, output_shape=(1,), name=name)(args)
