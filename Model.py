from gensim.models import KeyedVectors
from tensorflow.keras.layers import Flatten, Embedding, Dense, GRU, TimeDistributed, LSTM, Dropout, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import Model, Input
from tensorflow.keras.models import load_model
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
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mse'])
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
    batch_size = 1024
    epochs = 10
    x = [X_melody, X_lyrics]
    history = model.fit(x=x,
              y=y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              steps_per_epoch=int(len(X_lyrics) / batch_size),
              callbacks=[tensorboard_callback],
              )

    model.save("model.h5")
    return history

def predict_word(model:Model, melody, word:str, vocab:dict, index2word):
    word_index = vocab[word].index
    X = [np.array([melody]), np.vstack([word_index])]
    predicted_vec = model.predict(X)
    max_index = np.argmax(predicted_vec)
    one_hot_array = np.zeros((1, len(vocab) + 1))
    one_hot_array[max_index] = 1
    next_word = index2word[max_index]
    return next_word, one_hot_array


def generate_song(model:Model, X, we:KeyedVectors, song_lenth=50):
    first_word = 'hi' # TODO change to random pick from vocab
    song = [first_word]
    vocab = we.wv.vocab
    index2word = we.index2word
    melody = X['melody_vectors'][:song_lenth]
    X_melody = []
    for array in melody:
        X_melody.append(normalize(array))
    X_melody = np.array(X_melody)
    next_word, one_hot_array = predict_word(model, X_melody[0], first_word, vocab, index2word)
    song.append(next_word)
    for i in range(1, song_lenth):
        next_word, one_hot_array = predict_word(model, X_melody[i], next_word, vocab, index2word)
        song.append(next_word)

    return song

def load(path:str):
    return load_model(path)






