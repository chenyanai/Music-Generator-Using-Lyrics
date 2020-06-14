from gensim.models import KeyedVectors
from tensorflow.keras.layers import Flatten, Embedding, Dense, GRU, TimeDistributed, LSTM, Dropout, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras import Model, Input
from tensorflow.keras.models import load_model
import numpy as np
from Preprocessing import one_hot_by_max_value
from datetime import datetime
from sklearn.preprocessing import normalize
import random
import os

def build_model(sequence_length, mid_data_len, embedding_matrix, vocab_size):

    melody_input = Input(shape=(sequence_length, mid_data_len, ), name='melody_input')
    lyrics_vectors_input = Input(shape=(1, ), name='lyrics_vectors_input')

    melody_lstm = LSTM(50, return_sequences=False, name='melody_LSTM')(melody_input)
    lyrics_embedding = Embedding(vocab_size + 1, 300, weights=[embedding_matrix], input_length=1, trainable=True)\
        (lyrics_vectors_input)
    lyrics_flatten = Flatten()(lyrics_embedding)

    lyrics_melody_concat = Concatenate(axis=-1, name='lyrics_melody_concat')([lyrics_flatten, melody_lstm])

    dense = Dense(500, activation='relu', name='first_dense')(lyrics_melody_concat)
    dropout = Dropout(0.1)(dense)
    dense = Dense(vocab_size + 1, activation='softmax', name='last_dense')(dropout) # Predicting the next locations of the defense players

    model = Model(inputs=[melody_input, lyrics_vectors_input], outputs=dense)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mse'])
    model.summary()
    return model

def new_model(sequence_length, mid_data_len, embedding_matrix, vocab_size):

    melody_input = Input(shape=(sequence_length, mid_data_len, ), name='melody_input')
    lyrics_vectors_input = Input(shape=(1, ), name='lyrics_vectors_input')
    tempo_input = Input(shape=(1, ), name='tempo_input')

    melody_lstm = LSTM(50, return_sequences=False, name='melody_LSTM')(melody_input)
    lyrics_embedding = Embedding(vocab_size + 1, 300, weights=[embedding_matrix], input_length=1, trainable=True)\
        (lyrics_vectors_input)
    lyrics_flatten = Flatten()(lyrics_embedding)

    lyrics_melody_concat = Concatenate(axis=-1, name='lyrics_melody_concat')([lyrics_flatten, melody_lstm, tempo_input])

    dense = Dense(500, activation='relu', name='first_dense')(lyrics_melody_concat)
    dropout = Dropout(0.1)(dense)
    dense = Dense(vocab_size + 1, activation='softmax', name='last_dense')(dropout) # Predicting the next locations of the defense players

    model = Model(inputs=[melody_input, lyrics_vectors_input, tempo_input], outputs=dense)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mse'])
    model.summary()
    return model


def train_model(model, X, y, vocab_size, melody_input_type=1):
    log_dir = os.path.join("Logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping = EarlyStopping(monitor='loss', patience=3, min_delta=1e-1)

    X_lyrics = np.vstack(X['lyrics'])
    y = one_hot_by_max_value(y, vocab_size + 1)

    X_melody = []
    for array in X['melody_vectors']:
        X_melody.append(normalize(array))
    X_melody = np.array(X_melody)

    x = [X_melody, X_lyrics]

    if melody_input_type == 2:
        X_tempo = np.array(X['tempo'])
        x.append(X_tempo)
        # x = x + X_tempo
    batch_size = 1024
    # batch_size = 1
    epochs = 5

    history = model.fit(x=x,
              y=y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              # steps_per_epoch=int(len(X_lyrics) / batch_size),
              callbacks=[tensorboard_callback, early_stopping],
              validation_split=0.2
              )

    model.save("model_eos_trainable_melody2_1024b_5e.h5")
    return history

def predict_word(model:Model, melody, word:str, vocab:dict, reverse_word_dict:dict, tempo=None):
    word_index = vocab[word]
    if tempo:
        x = [np.array([melody]), np.array([word_index]), np.array([tempo])]
    else:
        x = [np.array([melody]), np.array([word_index])]
    predicted_vec = model.predict(x=x)
    word_index = np.random.choice(np.arange(len(vocab) + 1), p=predicted_vec[0])
    one_hot_array = np.zeros(shape=(len(vocab) + 1))
    one_hot_array[word_index] = 1
    next_word = reverse_word_dict[word_index]
    return next_word, one_hot_array


def generate_song(model:Model, X, words_dict, reverse_word_dict, song_length=50, melody_type=1):
    first_word = random.choice(list(words_dict.keys()))
    song = [first_word]
    melody = X['melody_vectors'][:song_length]
    X_melody = []
    for array in melody:
        X_melody.append(normalize(array))
    X_melody = np.array(X_melody)
    if melody_type == 1:
        next_word, one_hot_array = predict_word(model, X_melody[0], first_word, words_dict, reverse_word_dict)
    else:
        next_word, one_hot_array = predict_word(model, X_melody[0], first_word, words_dict, reverse_word_dict,
                                                X['tempo'][0])
    song.append(next_word)
    i = 0
    while(next_word != 'EOS' and i < len(X_melody)):
        next_word, one_hot_array = predict_word(model, X_melody[0], first_word, words_dict, reverse_word_dict,
                                                X['tempo'][i])
        song.append(next_word)
        i += 1
    return song

def load(path:str):
    return load_model(path)






