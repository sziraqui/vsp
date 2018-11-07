import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Input
from keras.layers.convolutional import Conv3D, ZeroPadding3D
from keras.layers.pooling import MaxPooling3D
from keras.layers.core import SpatialDropout3D, Flatten, Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras import backend as K
from .textprocessing import binmat2word


class WordReader:

    def __init__(self, name, params):
        self.name = name or "WordReader"
        self.model = None
        self.frameLength = params['frameLength']
        self.frameWidth = 100
        self.frameHeight = 50
        if params['resume']:
            self.model = model.load_model(params['model_file_checkpoint'])
        else:
            self.create_model(params)


    def create_model(self,params):
        input_shape = (self.frameLength, self.frameHeight, self.frameWidth, 3)
        model = Sequential()
        # Layer 1: Convolution 3D, 32 filters
        model.add(ZeroPadding3D(input_shape=input_shape, padding=(1,2,2)))
        model.add(Conv3D(filters=32, kernel_size=(3,5,5), strides=(1,2,2), kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(SpatialDropout3D(0.5))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        # Layer 2: Covolution 3D, 64 filters
        model.add(ZeroPadding3D(padding=(1,2,2)))
        model.add(Conv3D(filters=64, kernel_size=(3,5,5), strides=(1,1,1), kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(SpatialDropout3D(0.5))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        # Layer 3: Convolution 3D, 96 filters
        model.add(ZeroPadding3D(padding=(1,1,1)))
        model.add(Conv3D(filters=96, kernel_size=(3,3,3), strides=(1,1,1), kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(SpatialDropout3D(0.5))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        # Layer 4: Time series distribution
        model.add(TimeDistributed(Flatten()))
        # Layer 5: GRU
        model.add(Bidirectional(GRU(units=256, kernel_initializer='Orthogonal', return_sequences=True), merge_mode='concat'))
        # Layer 6: GRU
        model.add(Bidirectional(GRU(units=256, kernel_initializer='Orthogonal', return_sequences=True), merge_mode='concat'))
        # Output layer
        model.add(Dense(28, kernel_initializer='he_normal'))
        model.add(Activation('softmax'))

        model.compile(optimizer=params['optimizer'], loss=params['loss_func'], metrics=['accuracy'])
        self.model = model

    
    def train_model(self, xtrain, ytrain, trainParams):
        history = None
        error = False
        try:
            history = self.model.fit(xtrain, ytrain, batch_size=trainParams['batch_size'], epochs=trainParams['epochs'], validation_split=trainParams['validation_split'])
        except InterruptedError:
            model.save(f"checkpoint-{time.time()}-{trainParams['model_file']}")
            error = True
        if not error:
            model.save(trainParams['model_file'])
        return history
    

    def test_model(self, xtest, ytest, testParams):
        lossPercent, accuracy = self.model.evaluate(xtest, ytest)
        return lossPercent, accuracy


    def predict_word(self, frames):
        if frames.shape == (self.frameLength, self.frameHeight, self.frameWidth, 3):
            binmat = self.mode.predict(frames)
            return binmat2word(binmat)
        else:
            print("Incorrect frame shape")


    def __str__(self):
        return f"{self.name:\n}Model Summary:  {self.model.summary()}"


class SentenceReader(WordReader):

   def __init__(self, name, params):

        WordReader.__init__(self)
        self.frameLength = 75
        self.frameWidth = 100
        self.frameHeight = 50
        if params['resume']:
            self.model = model.load_model(params['model_file_checkpoint'])
        else:
            self.create_model(params)

    def __str__(self):
        return f"{self.name:\n}Model Summary:  {self.model.summary()}"
