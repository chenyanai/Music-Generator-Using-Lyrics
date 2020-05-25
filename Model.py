import tensorflow as tf
from tensorflow import keras
from keras.layers import Flatten, Embedding, Dense, GRU, TimeDistributed, LSTM, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras import Model
from keras import Input

def build_model(sequence_length, mid_data_len):

    melody_input = Input(shape=(sequence_length, mid_data_len, ), name='melody_input')
    lyrics_vectors_input = Input(shape=(300, ), name='lyrics_vectors_input')

    melody_lstm = LSTM(300, return_sequences=False, name='melody_LSTM')(melody_input)

    lyrics_melody_concat = Concatenate(axis=-1, name='lyrics_melody_concat')([lyrics_vectors_input, melody_lstm])

    dense = Dense(1024, activation='relu', name='first_dense')(lyrics_melody_concat)
    dropout = Dropout(0.1)(dense)
    dense = Dense(300, activation='relu', name='last_dense')(dropout) # Predicting the next locations of the defense players

    model = Model(inputs=[melody_input, lyrics_vectors_input], outputs=dense)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()


model = build_model(10, 10)
