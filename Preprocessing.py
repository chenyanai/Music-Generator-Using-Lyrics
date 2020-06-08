import pandas as pd
import pretty_midi
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import os
import re
import numpy as np
import pickle

from WordEmbedding import WordEmbedding


DATA_PATH = r'\Data'

def load_data(data_path):
    frames_columns = ['artist', 'song_name', 'lyrics']

    train_path = os.path.join(data_path, 'lyrics_train_set.csv')
    train_index = load_lyrics(train_path, frames_columns)

    test_path = os.path.join(data_path, 'lyrics_test_set.csv')
    test_index = load_lyrics(test_path, frames_columns)
    test_index['song_name'] = [song_name[1:] for song_name in test_index['song_name'].values]

    we_model = WordEmbedding()
    # we_model = None

    tokenizer, embedding_matrix, vocab_size = prepare_lyrics(test_index, train_index, we_model)

    #saving
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('embedding_matrix.pickle', 'wb') as handle:
        pickle.dump(embedding_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('Data/tokenizer.pickle', 'rb') as handle:
    #     tokenizer = pickle.load(handle)
    #
    # with open('Data/embedding_matrix.pickle', 'rb') as handle:
    #     embedding_matrix = pickle.load(handle)

    vocab_size = len(tokenizer.word_index)

    midi_path = os.path.join(data_path, 'midi_files')
    train_index['lyrics_sequence'] = tokenizer.texts_to_sequences(train_index['lyrics'])
    test_index['lyrics_sequence'] = tokenizer.texts_to_sequences(test_index['lyrics'])
    train_midis, test_midis = read_midi_files(midi_path, train_index, test_index, vocab_size)

    return train_midis, train_index, test_midis, test_index, embedding_matrix, vocab_size, we_model


def load_lyrics(data_path, frames_columns):
    df = pd.read_csv(data_path)
    df = df.iloc[:, : 3]
    df.columns = frames_columns
    df['lyrics'] = df.lyrics.apply(clean_text)
    df['lyrics'] = [lyric + ' EOS' for lyric in df['lyrics']]
    return df


def prepare_lyrics(test_index, train_index, we_model):

    tokenizer = Tokenizer()
    lyrics = list(train_index['lyrics'].values) + list(test_index['lyrics'].values)
    tokenizer.fit_on_texts(lyrics)
    vocab_size = len(tokenizer.word_index)
    embedding_matrix = create_vectors_matrix(tokenizer, vocab_size, we_model)

    return tokenizer, embedding_matrix, vocab_size

def create_vectors_matrix(tokenizer, vocab_size, we_model):
    # create a weight matrix for words in training docs

    embedding_matrix = np.zeros((vocab_size+1, 300))
    for word, i in tokenizer.word_index.items():
        embedding_vector = we_model.get_word_vec(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            # TODO: try give zeros for eos only and change trainable to false
            embedding_matrix[i] = np.random.rand(300)

    return embedding_matrix

def clean_text(text:str)->str:
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = text.replace('&', '')
    text = text.replace('--', '')
    text = text.replace(':', '')
    return text

def read_midi_files(path, train_index, test_index, vocab_size):
    train_midi_dict = {}
    test_midi_dict = {}
    # i = 0
    for file in tqdm(os.listdir(path)):
        # i += 1
        # if i < 20:

        file_path = os.path.join(path, file)
        song_name = get_song_name_from_file_name(file)

        if song_name[0] == ' ':
            song_name = song_name[1:]

        if song_name in train_index['song_name'].values:
            try:
                pm = pretty_midi.PrettyMIDI(file_path)
                song_lyrics = train_index[train_index['song_name'] == song_name]['lyrics_sequence'].values[0]
                train_midi_dict[file[:-4]] = [pm, song_lyrics]
            except:
                print(file)
        elif song_name in test_index['song_name'].values:
            try:
                pm = pretty_midi.PrettyMIDI(file_path)
                song_lyrics = test_index[test_index['song_name'] == song_name]['lyrics_sequence'].values[0]

                one_hot_by_max_value(song_lyrics, vocab_size)

                test_midi_dict[file[:-4]] = [pm, song_lyrics]
            except:
                print(file)

    return train_midi_dict, test_midi_dict


def one_hot_by_max_value(data, vocab_size):
    one_hot_array = to_categorical(data, num_classes=vocab_size + 1)
    # one_hot_array = np.zeros((len(data), vocab_size + 1))
    # one_hot_array[np.arange(len(data)), data] = 1
    return one_hot_array

# def convert_lyrics_to_one_hot_array(lyrics, tokenizer):


def get_song_name_from_file_name(file_name):
    name = file_name[:-4] # remove file type
    name = name[name.find('-') + 2:]
    name = name.replace('_', ' ')
    return name.lower()

def convert_data_to_model_input(df, sequence_length):
    """

    :param df:
                df columns:
                    md - prettyMIDI object
                    lyrics - string with the song lyrics
                    lyrics_vectors - list of numpy arrays, each array is 300 long and represents the embedding of a
                    word in the lyrics
                    chroma_vectors - numpy array that contains data on the melody, the length of the array equals to
                    the number of the words in the lyrics
    :return: data fitted to be used for training the model
    """

    X = {
        'lyrics': list(),
        'melody_vectors': list()
    }
    y = list()

    for i, song_data in df.iterrows():

        melody_data = song_data['chroma_vectors']

        if len(melody_data) >= sequence_length + 1:
            for j in range(len(melody_data)):
                if len(melody_data) > j + sequence_length + 1:
                    if len(song_data['lyrics']) > j + sequence_length + 1:
                        try:
                            X['lyrics'].append(song_data['lyrics'][j + sequence_length])
                            X['melody_vectors'].append(song_data['chroma_vectors'][j:j + sequence_length])
                            y.append(song_data['lyrics'][j + sequence_length + 1])
                        except:
                            print('hi')
    y = np.vstack(y)
    return X, y