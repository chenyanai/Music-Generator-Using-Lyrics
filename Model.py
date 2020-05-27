from tensorflow.keras.layers import Flatten, Embedding, Dense, GRU, TimeDistributed, LSTM, Dropout, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import Model, Input
import numpy as np
from Preprocessing import one_hot_by_max_value
from datetime import datetime
from sklearn.preprocessing import normalize

def build_model(sequence_length, mid_data_len, embedding_matrix, vocab_size):

    melody_input = Input(shape=(sequence_length, mid_data_len, ), name='melody_input')
    lyrics_vectors_input = Input(shape=(1, ), name='lyrics_vectors_input')

    melody_lstm = LSTM(50, return_sequences=False, name='melody_LSTM')(melody_input)
    lyrics_embedding = Embedding(vocab_size + 1, 300, weights=[embedding_matrix], input_length=1, trainable=False)\
        (lyrics_vectors_input)
    lyrics_flatten = Flatten()(lyrics_embedding)

    lyrics_melody_concat = Concatenate(axis=-1, name='lyrics_melody_concat')([lyrics_flatten, melody_lstm])

    dense = Dense(500, activation='relu', name='first_dense')(lyrics_melody_concat)
    dropout = Dropout(0.1)(dense)
    dense = Dense(vocab_size + 1, activation='softmax', name='last_dense')(dropout) # Predicting the next locations of the defense players

    model = Model(inputs=[melody_input, lyrics_vectors_input], outputs=dense)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    return model


def train_model(model, X, y, vocab_size):

    log_dir = "Logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    X_lyrics = np.vstack(X['lyrics'])
    y = one_hot_by_max_value(y, vocab_size)

    X_melody = []
    for array in X['melody_vectors']:
        X_melody.append(normalize(array))
    X_melody = np.array(X_melody)
    batch_size = 16
    model.fit(x=[X_melody, X_lyrics],
              y=y,
              batch_size=batch_size,
              epochs=3,
              verbose=1,
              steps_per_epoch=int(len(X_lyrics) / batch_size))
              # callbacks=[tensorboard_callback])
    model.save("model.h5")
    return model