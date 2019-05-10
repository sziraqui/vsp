import tensorflow as tf

# NOTE: GRU outputs state = [h,c] But, LSTM doesn't give this list, instead separately h,c


def gru(units):
    # If you have a GPU, we recommend using the CuDNNGRU layer (it provides a
    # significant speedup).
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNGRU(units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
    else:
        return tf.keras.layers.GRU(units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_activation='sigmoid',
                                   recurrent_initializer='glorot_uniform')


def lstm(units):
    # If you have a GPU, we recommend using the CuDNNGRU layer (it provides a
    # significant speedup).
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNLSTM(units,
                                         return_sequences=True,
                                         return_state=True,
                                         recurrent_initializer='glorot_uniform')
    else:
        return tf.keras.layers.LSTM(units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_activation='sigmoid',
                                    recurrent_initializer='glorot_uniform')


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, 64, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


"""###Encoder-Decoder Architecture"""


class Encoder(tf.keras.Model):
    def __init__(self, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units

        # NOTE: Define as many times as you want to use. Thus, there will be REDUNDANCY
        self.zero1 = tf.keras.layers.ZeroPadding3D(padding=(1, 2, 2))
        self.conv1 = tf.keras.layers.Conv3D(filters=32, kernel_size=(
            3, 5, 5), strides=(1, 2, 2), kernel_initializer='he_normal')
        self.bn1 = tf.keras.layers.BatchNormalization()
        # activation defined later
        self.spd1 = tf.keras.layers.SpatialDropout3D(0.5)
        self.maxp1 = tf.keras.layers.MaxPooling3D(
            pool_size=(1, 2, 2), strides=(1, 2, 2))
        # activation defined later
        self.zero2 = tf.keras.layers.ZeroPadding3D(padding=(1, 2, 2))
        self.conv2 = tf.keras.layers.Conv3D(filters=64, kernel_size=(
            3, 5, 5), strides=(1, 1, 1), kernel_initializer='he_normal')
        self.bn2 = tf.keras.layers.BatchNormalization()
        # activation defined later
        self.spd2 = tf.keras.layers.SpatialDropout3D(0.5)
        self.maxp2 = tf.keras.layers.MaxPooling3D(
            pool_size=(1, 2, 2), strides=(1, 2, 2))
        # activation defined later
        self.zero3 = tf.keras.layers.ZeroPadding3D(padding=(1, 1, 1))
        self.conv3 = tf.keras.layers.Conv3D(filters=96, kernel_size=(
            3, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal')
        self.bn3 = tf.keras.layers.BatchNormalization()
        # activation defined later
        self.spd3 = tf.keras.layers.SpatialDropout3D(0.5)
        self.maxp3 = tf.keras.layers.MaxPooling3D(
            pool_size=(1, 2, 2), strides=(1, 2, 2))

        self.td = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())
        self.gru = gru(units=256)

    def call(self, x, hidden):
        zero1 = self.zero1(x)
        conv1 = self.conv1(zero1)
        batc1 = self.bn1(conv1)
        actv1 = tf.nn.relu(batc1)
        drop1 = self.spd1(actv1)
        maxp1 = self.maxp1(drop1)

        zero2 = self.zero2(maxp1)
        conv2 = self.conv2(zero2)
        batc2 = self.bn2(conv2)
        actv2 = tf.nn.relu(batc2)
        drop2 = self.spd2(actv2)
        maxp2 = self.maxp2(drop2)

        zero3 = self.zero3(maxp2)
        conv3 = self.conv3(zero3)
        batc3 = self.bn3(conv3)
        actv3 = tf.nn.relu(batc3)
        drop3 = self.spd3(actv3)
        maxp3 = self.maxp3(drop3)

        resh1 = self.td(maxp3)
        output, state = self.gru(resh1, initial_state=hidden)

        #output, state_h, state_c=self.lstm(resh1,initial_state = hidden)
        #state = [state_h, state_c]

        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru=gru(self.dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size)
        
        # used for attention
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying tanh(FC(EO) + FC(H)) to self.V
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))
        
        # attention_weights shape == (batch_size, max_length, 1) get alpha
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        # x shape after passing through embedding == (batch_size, 1, embedding_dim) coz 1st dec input is '<start>' i.e x of shape (bz,1)
        x = self.embedding(x)
        
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        
        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))
        
        # output shape == (batch_size * 1, vocab)
        x = self.fc(output)
        
        return x, state, attention_weights
        
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.dec_units))