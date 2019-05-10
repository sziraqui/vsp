import os
import time
from time import localtime
from time import strftime as timeformat
import tensorflow as tf
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv3D, ZeroPadding3D
from keras.layers.pooling import MaxPooling3D
from keras.layers.core import SpatialDropout3D, Flatten, Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras import backend as K
from keras.models import load_model
from .textprocessing import ints2word, wordCollapse, sentenceCollapse, CODE_BLANK
from .generators import GeneratorInterface
from .metrics import CTC, CTC_LOSS_STR
from .layers import Encoder, Decoder
from .utils import load_tokenizer
from .utils import Log


class WordReader:
    """
        WordReader constructor
        Required args
        - params: A dict-like object containing model parameters
    """

    def __init__(self, params):
        self.name = params['model_file']
        self.model = None
        try:
            self.frameLength = params['frame_length']
        except KeyError:
            self.frameLength = 7
        self.frameWidth = 100
        self.frameHeight = 50
        self.create_model(params)
        if params['resume']:
            self.model.load_weights(params['model_file_checkpoint'])

        try:
            self.batchSize = params['batch_size']
            self.sampleSize = params['sample_size']
            # Make sampleSize a multiple of batchSize
            self.sampleSize = self.sampleSize - self.sampleSize % self.batchSize
        except KeyError:
            self.sampleSize = 0
            self.batchSize = 0

    def create_model(self, params):
        inputShape = (self.frameLength, self.frameHeight, self.frameWidth, 3)
        labelInput = Input(name='label_input', shape=[32])
        inputLen = Input(name='input_length', shape=[1], dtype='int64')
        labelLen = Input(name='label_length', shape=[1], dtype='int64')
        # Input layer
        imgInput = Input(shape=inputShape, name='input')
        # Layer 1: Convolution 3D, 32 filters
        padd1 = ZeroPadding3D(input_shape=inputShape, padding=(
            1, 2, 2), name='zero_pad1')(imgInput)
        conv1 = Conv3D(filters=32, kernel_size=(3, 5, 5), strides=(
            1, 2, 2), kernel_initializer='he_normal', name='conv3d_32')(padd1)
        norm1 = BatchNormalization(name='bnorm_1')(conv1)
        relu1 = Activation('relu', name='relu_1')(norm1)
        drop1 = SpatialDropout3D(0.5)(relu1)
        pool1 = MaxPooling3D(pool_size=(1, 2, 2), strides=(
            1, 2, 2), name='max_pool1')(drop1)
        # Layer 2: Covolution 3D, 64 filters
        padd2 = ZeroPadding3D(padding=(1, 2, 2), name='zero_pad2')(pool1)
        conv2 = Conv3D(filters=64, kernel_size=(3, 5, 5), strides=(
            1, 1, 1), kernel_initializer='he_normal', name='conv3d_64')(padd2)
        norm2 = BatchNormalization(name='bnorm2')(conv2)
        relu2 = Activation('relu', name='relu_2')(norm2)
        drop2 = SpatialDropout3D(0.5)(relu2)
        pool2 = MaxPooling3D(pool_size=(1, 2, 2), strides=(
            1, 2, 2), name='max_pool2')(drop2)
        # Layer 3: Convolution 3D, 96 filters
        padd3 = ZeroPadding3D(padding=(1, 1, 1), name='zero_pad3')(pool2)
        conv3 = Conv3D(filters=96, kernel_size=(3, 3, 3), strides=(
            1, 1, 1), kernel_initializer='he_normal', name='conv3d_96')(padd3)
        norm3 = BatchNormalization(name='bnorm3')(conv3)
        relu3 = Activation('relu', name='relu_3')(norm3)
        drop3 = SpatialDropout3D(0.5)(relu3)
        pool3 = MaxPooling3D(pool_size=(1, 2, 2), strides=(
            1, 2, 2), name='max_pool3')(drop3)
        # Layer 4: Time series distribution
        time1 = TimeDistributed(Flatten())(pool3)
        # Layer 5: GRU
        gru1 = Bidirectional(GRU(units=256, kernel_initializer='Orthogonal',
                                 return_sequences=True), merge_mode='concat', name='gru1')(time1)
        # Layer 6: GRU
        gru2 = Bidirectional(GRU(units=256, kernel_initializer='Orthogonal',
                                 return_sequences=True), merge_mode='concat', name='gru2')(gru1)
        # Output layer
        dense = Dense(28, kernel_initializer='he_normal', name='dense')(gru2)
        ypred = Activation('softmax', name='softmax')(dense)
        loss = CTC([ypred, labelInput, inputLen, labelLen], name=CTC_LOSS_STR)
        model = Model(inputs=[imgInput, labelInput,
                              inputLen, labelLen], outputs=loss)

        adam = Adam(
            lr=params['learning_rate'],
            beta_1=params['learning_beta1'],
            beta_2=params['learning_beta2'],
            epsilon=params['learning_decay'])
        model.compile(optimizer=adam,
                      loss=params['loss_func'], metrics=['accuracy'])
        self.model = model
        # capture_output is a Function that captures tensor output from ypred and converts it into numpy array
        self.capture_output = K.function([imgInput], [ypred])

    def train_model(self, xtrain, ytrain, trainParams, generator=None):
        history = None
        error = False
        tensorboard = TensorBoard(log_dir=os.path.join(
            trainParams['log_dir'], os.path.basename(trainParams['model_file'])))
        try:
            history = self.model.fit(
                xtrain, ytrain,
                batch_size=trainParams['batch_size'],
                epochs=trainParams['epochs'],
                validation_split=trainParams['validation_split'],
                callbacks=[tensorboard])
        except InterruptedError:
            self.model.save(
                f"checkpoint-{timeformat('%d-%m-%Y-%H-%M-%S', localtime())}-{trainParams['model_file']}")
            error = True
        if not error:
            self.model.save(trainParams['model_file'])
        return history

    def train_with_generator(self, trainParams, generator):
        history = None
        error = False
        tensorboard = TensorBoard(log_dir=os.path.join(
            trainParams['log_dir'], os.path.basename(trainParams['model_file'])))
        try:
            assert isinstance(generator, GeneratorInterface)
            history = self.model.fit_generator(
                generator.next_batch(self.batchSize),
                initial_epoch=trainParams['initial_epoch'],
                epochs=trainParams['epochs'],
                steps_per_epoch=int(
                    trainParams['sample_size']//trainParams['batch_size']),
                max_queue_size=trainParams['generator_queue_size'],
                verbose=1,
                callbacks=[tensorboard])
        except InterruptedError:
            self.model.save(
                f"checkpoint-{timeformat('%d-%m-%Y-%H-%M-%S', localtime())}-{trainParams['model_file']}")
            error = True
        if not error:
            self.model.save(trainParams['model_file'])
        return history

    def test_model(self, xtest, ytest, testParams):
        lossPercent, accuracy = self.model.evaluate(xtest, ytest)
        return lossPercent, accuracy

    def predict_raw(self, frames):
        assert frames.shape == (
            self.frameLength, self.frameHeight, self.frameWidth, 3)
        out = self.capture_output([np.array([frames])])[0]
        return out[0]

    def predict_word(self, frames):
        out = self.predict_raw(frames)
        codePoints = np.argmax(out, axis=1)
        expandedWord = ints2word(codePoints)
        word = wordCollapse(expandedWord)
        return word

    def __str__(self):
        return f"{self.name:\n}Model Summary:  {self.model.summary()}"


class SentenceReader(WordReader):

    def __init__(self, params):
        WordReader.__init__(self, params)
        try:
            self.frameLength = params['frame_length']
        except KeyError:
            self.frameLength = 75
        self.frameWidth = 100
        self.frameHeight = 50
        self.create_model(params)
        if params['resume']:
            self.model.load_weights(params['model_file_checkpoint'])

    def predict_sentence(self, frames):
        out = self.predict_raw(frames)
        codePoints = np.argmax(out, axis=1)
        ctcStr = ints2word(codePoints)
        sentence = sentenceCollapse(ctcStr)
        return sentence

    def __str__(self):
        return f"{self.name:\n}Model Summary:  {self.model.summary()}"

"""
    Eager execution must be enabled to use this class
"""
class VSPNet:
    """
        VSPNet constructor
        Required args
        - params: A dict-like object containing model parameters
    """

    def __init__(self, params):
        self.name = params['model_name']

        self.encoder = None
        self.decoder = None

        self.frameLength = params.get('frame_length', 75)
        self.frameWidth = params.get('frame_width', 100)
        self.frameHeight = params.get('frame_height', 50)

        self.tokeniser = load_tokeniser(params['tokeniser'])
        self.vocabSize = len(tokeniser.word_index)
        self.gruUnits = params.get('gru_units', 256)
        self.embeddingDim = params.get('embedding_dim', 50)

        self.batchSize = params.get('batch_size', 1)
        self.sampleSize = params.get('sample_size', 1)
        # Make sampleSize a multiple of batchSize
        self.sampleSize = self.sampleSize - self.sampleSize % self.batchSize

        self.create_model(params)
        if params['resume']:
            self.checkpointDir = os.path.join(params['checkpoint_dir'], self.name)
            self.checkpoint.restore(tf.train.latest_checkpoint(checkpointDir)).assert_consumed()

    def create_model(self, params):

        self.encoder = Encoder(self.vocabSize, self.batchSize)
        self.decoder = Decoder(
            self.vocabSize, self.embeddingDim, self.gruUnits, self.batchSize)
        self.optimizer = tf.train.AdamOptimizer()
        self.checkpoint = tf.train.Checkpoint(
            encoder=self.encoder, decoder=self.decoder, optimizer=self.optimizer)


    def _train(self, Xtrain, decoderInputData, trainParams):
        BUFFER_SIZE = trainParams['generator_queue_size']
        N_BATCH = BUFFER_SIZE/self.batchSize
        dataset = tf.data.Dataset.from_tensor_slices((Xtrain, decoderInputData)).shuffle(BUFFER_SIZE)
        dataset = dataset.batch(self.batchSize, drop_remainder=True)

        loss_plot = []
        train_accuracy_results = []

        for epoch in range(self.epochs):
            start = time.time()
            hidden = self.encoder.initialize_hidden_state()
            total_loss = 0
            epoch_accuracy = tf.contrib.eager.metrics.Accuracy() #accuracy #Change here for without Nightly

            for (batch, (img_tensor, target)) in enumerate(dataset):
                loss = 0

                pred_list=[]#accuracy
                with tf.GradientTape() as tape:
                    enc_output, enc_hidden = self.encoder(img_tensor, hidden)

                    dec_hidden = enc_hidden

                    dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']] * self.batchSize, 1)       

                    # Teacher forcing - feeding the target as the next input
                    for t in range(1, target.shape[1]):
                        # passing enc_output to the decoder
                        predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
                        pred_list.append(tf.argmax(predictions, axis=1, output_type=tf.int32))#accuracy

                        loss += loss_function(target[:, t], predictions)

                        # using teacher forcing
                        dec_input = tf.expand_dims(target[:, t], 1)

                total_loss += (loss / int(target.shape[1]))

                variables = self.encoder.variables + self.decoder.variables

                gradients = tape.gradient(loss, variables) 

                self.optimizer.apply_gradients(zip(gradients, variables))#, tf.train.get_or_create_global_step()

                epoch_accuracy(np.asarray(pred_list).T, target[:,1:]) #accuracy

                if batch % 100 == 0:
                    print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, 
                                                                  batch, 
                                                                  loss.numpy() / int(target.shape[1])))
            # storing the epoch end loss value to plot later
            loss_plot.append(total_loss / N_BATCH)
            train_accuracy_results.append(epoch_accuracy.result())

            # saving (checkpoint) the model every 2 epochs
            #if (epoch + 1) % 2 == 0:
              #checkpoint.save(file_prefix = checkpoint_prefix)

            print ('Epoch {} Loss {:.6f}, Accuracy: {:.3%}'.format(epoch + 1, 
                                                                   total_loss/N_BATCH,
                                                                   epoch_accuracy.result()))#accuracy
            print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))        

    def train_model(self, xtrain, ytrain, trainParams):
        
        try:
            self._train(xtrain, ytrain, trainParams)
        except InterruptedError:
            self.checkpoint.save(self.checkpointDir)
            Log.error("Training interrupted. Checkpoint saved")
            

    def train_with_generator(self, trainParams=None, generator=None):
       raise NotImplementedError("Training using generator not implemented yet")

    def test_model(self, xtest, ytest, testParams):
        lossPercent, accuracy = self.evaluate(xtest, ytest)
        return lossPercent, accuracy

    def predict_raw(self, img_tensor):
        hidden = [tf.zeros((1, units))] #?
        img_tensor=tf.convert_to_tensor(img_tensor)
        img_tensor=tf.expand_dims(img_tensor, 0) #start?
        enc_out, enc_hidden = self.encoder(img_tensor, hidden)

        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']], 0)
        dec_hidden = enc_hidden #?

        result = ''

        for i in range(max_length_targ):
            predictions, dec_hidden, attention_weights = self.decoder(dec_input, dec_hidden, enc_out)

            predicted_id = tf.argmax(predictions[0]).numpy()
            result += tokenizer.index_word[predicted_id] + ' '


            if tokenizer.index_word[predicted_id] == '<end>':
                return result

            dec_input = tf.expand_dims([predicted_id], 0)

        return result
