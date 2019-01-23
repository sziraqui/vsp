import os
from time import localtime as time
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
from .textprocessing import ints2word, wordCollapse, CODE_BLANK
from .generators import GeneratorInterface
from .metrics import CTC, CTC_LOSS_STR


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
        if params['resume']:
            self.model = load_model(params['model_file_checkpoint'])
        else:
            self.create_model(params)
        
        try:
            self.batchSize = params['batch_size']
            self.sampleSize = params['sample_size']
            # Make sampleSize a multiple of batchSize
            self.sampleSize = self.sampleSize - self.sampleSize%self.batchSize
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
        padd1 = ZeroPadding3D(input_shape=inputShape, padding=(1,2,2), name='zero_pad1')(imgInput)
        conv1 = Conv3D(filters=32, kernel_size=(3,5,5), strides=(1,2,2), kernel_initializer='he_normal', name='conv3d_32')(padd1)
        norm1 = BatchNormalization(name='bnorm_1')(conv1)
        relu1 = Activation('relu', name='relu_1')(norm1)
        drop1 = SpatialDropout3D(0.5)(relu1)
        pool1 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max_pool1')(drop1)
        # Layer 2: Covolution 3D, 64 filters
        padd2 = ZeroPadding3D(padding=(1,2,2), name='zero_pad2')(pool1)
        conv2 = Conv3D(filters=64, kernel_size=(3,5,5), strides=(1,1,1), kernel_initializer='he_normal', name='conv3d_64')(padd2)
        norm2 = BatchNormalization(name='bnorm2')(conv2)
        relu2 = Activation('relu', name='relu_2')(norm2)
        drop2 = SpatialDropout3D(0.5)(relu2)
        pool2 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max_pool2')(drop2)
        # Layer 3: Convolution 3D, 96 filters
        padd3 = ZeroPadding3D(padding=(1,1,1), name='zero_pad3')(pool2)
        conv3 = Conv3D(filters=96, kernel_size=(3,3,3), strides=(1,1,1), kernel_initializer='he_normal', name='conv3d_96')(padd3)
        norm3 = BatchNormalization(name='bnorm3')(conv3)
        relu3 = Activation('relu', name='relu_3')(norm3)
        drop3 = SpatialDropout3D(0.5)(relu3)
        pool3 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max_pool3')(drop3)
        # Layer 4: Time series distribution
        time1 = TimeDistributed(Flatten())(pool3)
        # Layer 5: GRU 
        gru1 = Bidirectional(GRU(units=256, kernel_initializer='Orthogonal', return_sequences=True), merge_mode='concat', name='gru1')(time1)
        # Layer 6: GRU
        gru2 = Bidirectional(GRU(units=256, kernel_initializer='Orthogonal', return_sequences=True), merge_mode='concat', name='gru2')(gru1)
        # Output layer
        dense = Dense(28, kernel_initializer='he_normal', name='dense')(gru2)
        ypred = Activation('softmax', name='softmax')(dense)
        loss = CTC([ypred, labelInput, inputLen, labelLen], name=CTC_LOSS_STR)
        model = Model(inputs=[imgInput, labelInput, inputLen, labelLen], outputs=loss)
        
        adam = Adam(
                lr=params['learning_rate'], 
                beta_1=params['learning_beta1'],
                beta_2=params['learning_beta2'],
                epsilon=params['learning_decay'])
        model.compile(optimizer=adam, loss=params['loss_func'], metrics=['accuracy'])
        self.model = model
        self.model_input = imgInput
        self.model_output = ypred

    
    def train_model(self, xtrain, ytrain, trainParams, generator=None):
        history = None
        error = False
        tensorboard = TensorBoard(log_dir=os.path.join(trainParams['log_dir'], os.path.basename(trainParams['model_file'])))
        try:
            history = self.model.fit(
                xtrain, ytrain, 
                batch_size=trainParams['batch_size'], 
                epochs=trainParams['epochs'], 
                validation_split=trainParams['validation_split'],
                callbacks=[tensorboard])
        except InterruptedError:
            self.model.save(f"checkpoint-{timeformat('%d-%m-%Y-%H-%M-%S', time())}-{trainParams['model_file']}")
            error = True
        if not error:
            self.model.save(trainParams['model_file'])
        return history
     
    def train_with_generator(self, trainParams, generator):
        history = None
        error = False
        tensorboard = TensorBoard(log_dir=os.path.join(trainParams['log_dir'], os.path.basename(trainParams['model_file'])))
        try:
            assert isinstance(generator, GeneratorInterface)
            history = self.model.fit_generator(
                generator.next_batch(self.batchSize), 
                initial_epoch=trainParams['initial_epoch'], 
                epochs=trainParams['epochs'], 
                steps_per_epoch=int(trainParams['sample_size']//trainParams['batch_size']),
                max_queue_size=trainParams['generator_queue_size'],
                verbose=1,
                callbacks=[tensorboard])
        except InterruptedError:
            self.model.save(f"checkpoint-{timeformat('%d-%m-%Y-%H-%M-%S', time())}-{trainParams['model_file']}")
            error = True
        if not error:
            self.model.save(trainParams['model_file'])
        return history

    def test_model(self, xtest, ytest, testParams):
        lossPercent, accuracy = self.model.evaluate(xtest, ytest)
        return lossPercent, accuracy

    @property
    def capture_output(self):
        # captures output of softmax so we can decode the output during visualization
        return K.function([self.input_data, K.learning_phase()], [self.y_pred, K.learning_phase()])


    def predict_raw(self, frames):
        assert frames.shape == (self.frameLength, self.frameHeight, self.frameWidth, 3)
        input_batch = np.array([frames]);
        codePoints = capture_output([input_batch, 0])[0]
        return codePoints


    def predict_word(self, frames):
        codePoints = self.predict_raw(frames)
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
    if params['resume']:
        self.model = load_model(params['model_file_checkpoint'])
    else:
        self.create_model(params)


    def __str__(self):
        return f"{self.name:\n}Model Summary:  {self.model.summary()}"
