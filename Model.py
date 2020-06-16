from tensorflow.keras.layers import Flatten, Embedding, Dense, GRU, TimeDistributed, LSTM, Dropout, Concatenate
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras import Model, Input
from tensorflow.keras.models import load_model
import numpy as np
from Preprocessing import one_hot_by_max_value, load_lyrics
from datetime import datetime
from sklearn.preprocessing import normalize
import random
import os

def first_melody_model(sequence_length, mid_data_len, embedding_matrix, vocab_size):
    """
    Create the model for the first method for incorporating the melody
    :param sequence_length: melody seq length
    :param mid_data_len: num of features in melody
    :param embedding_matrix: words embedding data
    :param vocab_size: num of words
    :return: keras model
    """
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

def second_melody_model(sequence_length, mid_data_len, embedding_matrix, vocab_size):
    """
    Create the model for the second method for incorporating the melody
    :param sequence_length: melody seq length
    :param mid_data_len: num of features in melody
    :param embedding_matrix: words embedding data
    :param vocab_size: num of words
    :return: keras model
    """
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
    """
    Train the given model on the data
    :param model: keras model
    :param X: x train data
    :param y: target data
    :param vocab_size: num of words
    :param melody_input_type: melody type
    :return: history of the trained model
    """
    log_dir = os.path.join("Logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, min_delta=1e-1)

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
    batch_size = 1024
    epochs = 100

    history = model.fit(x=x,
              y=y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              callbacks=[tensorboard_callback, early_stopping],
              validation_split=0.2
              )

    model.save(f"model_eos_trainable_melody{melody_input_type}.h5")
    return history

def predict_word(model:Model, melody, word:str, word_dict:dict, reverse_word_dict:dict, tempo=None):
    """
    Predicts the next word using the model and according to the given data
    :param model: keras model
    :param melody: melody data
    :param word: current word
    :param word_dict: dictionary that contains words and their index
    :param reverse_word_dict: index to words dictionary
    :param tempo: tempo data if needed
    :return: the next word
    """
    word_index = word_dict[word]
    if tempo:
        x = [np.array([melody]), np.array([word_index]), np.array([tempo])]
    else:
        x = [np.array([melody]), np.array([word_index])]

    predicted_vec = model.predict(x=x)
    word_index = np.random.choice(np.arange(len(word_dict) + 1), p=predicted_vec[0])

    one_hot_array = np.zeros(shape=(len(word_dict) + 1))
    one_hot_array[word_index] = 1
    next_word = reverse_word_dict[word_index]
    return next_word, one_hot_array


def generate_song(model:Model, X, word_dict, reverse_word_dict, song_length=50, melody_type=1, first_word=None):
    """
    Generate song using the model
    :param model: keras model
    :param X: test data used for generating
    :param word_dict: dictionary that contains words and their index
    :param reverse_word_dict: index to words dictionary
    :param song_length: max song length
    :param melody_type: melody integration type
    :param first_word: first word of the song, if None a random word will be chosen
    :return: generated song
    """
    if first_word is None:
        current_word = random.choice(list(word_dict.keys()))
    else:
        current_word = first_word

    song = [current_word]
    melody = X['melody_vectors'][:song_length]
    X_melody = []

    for array in melody:
        X_melody.append(normalize(array))
    X_melody = np.array(X_melody)

    i = 0
    while(current_word != 'eos' and i < len(X_melody)):
        if melody_type == 1:
            tempo = None
        else:
            tempo = X['tempo'][i]
        current_word, one_hot_array = predict_word(model, X_melody[i], current_word, word_dict,
                                                   reverse_word_dict, tempo=tempo)
        song.append(current_word)
        i += 1
    return song

def load(path:str):
    return load_model(path)

def generate_song_with_same_words(model_1, model_2, x_test_1, x_test_2, word_dict, reverse_word_dict, song_length=50,
                                  iter_num=3):
    """
    Generate songs with the same fist word
    :param model_1: keras model of melody type 1
    :param model_2: keras model of melody type 2
    :param x_test_1: test data that first to model 1
    :param x_test_2: test data that first to model 2
    :param word_dict: dictionary that contains words and their index
    :param reverse_word_dict: index to words dictionary
    :param song_length: max song length
    :param iter_num: number of iterations of songs generations
    :return: None, prints the songs
    """

    frames_columns = ['artist', 'song_name', 'lyrics']
    test_path = os.path.join(r'Data', 'lyrics_test_set.csv')
    test_lyrics = load_lyrics(test_path, frames_columns)

    for i in range(iter_num):
        word = random.choice(list(word_dict.keys()))

        for j, Xi_test in enumerate(x_test_1):
            print('song:' + str(test_lyrics['song_name'].iloc[j]))

            song = generate_song(model_1, Xi_test, word_dict=word_dict, reverse_word_dict=reverse_word_dict,
                                 song_length=song_length, melody_type=1, first_word=word)
            print('model 1:')
            print(song)

            song = generate_song(model_2, x_test_2[j], word_dict=word_dict, reverse_word_dict=reverse_word_dict,
                                 song_length=song_length, melody_type=2, first_word=word)
            print('model 2:')
            print(song)

def generate_songs_using_first_word(model_1, model_2, x_test_1, x_test_2, word_dict, reverse_word_dict, song_length=50):
    """
    Generating songs using the first word of the original lyrics
    :param model_1: keras model of melody type 1
    :param model_2: keras model of melody type 2
    :param x_test_1: test data that first to model 1
    :param x_test_2: test data that first to model 2
    :param word_dict: dictionary that contains words and their index
    :param reverse_word_dict: index to words dictionary
    :param song_length: max song length
    :return: None, prints the songs
    """
    frames_columns = ['artist', 'song_name', 'lyrics']
    test_path = os.path.join(r'Data', 'lyrics_test_set.csv')
    test_lyrics = load_lyrics(test_path, frames_columns)

    for j, Xi_test in enumerate(x_test_1):
        word = test_lyrics['lyrics'].iloc[j].split()[0]
        print('song:' + str(test_lyrics['song_name'].iloc[j]))

        song = generate_song(model_1, Xi_test, word_dict=word_dict, reverse_word_dict=reverse_word_dict,
                             song_length=song_length, melody_type=1, first_word=word)
        print('model 1:')
        print(song)

        song = generate_song(model_2, x_test_2[j], word_dict=word_dict, reverse_word_dict=reverse_word_dict,
                             song_length=song_length, melody_type=2, first_word=word)
        print('model 2:')
        print(song)