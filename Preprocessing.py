import pandas as pd
import pretty_midi
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import os
import re
import numpy as np
import pickle

from WordEmbedding import WordEmbedding


DATA_PATH = r'\Data'


def pp_pipeline(data_folder, seq_len=20, melody_method=1, save_data=True, embedding_model_path='Data/GoogleNews-vectors-negative300.bin'):

    if save_data:

        test_index, train_index = load_lyrics_data(data_folder)

        we_model = WordEmbedding(path=embedding_model_path)
        tokenizer, embedding_matrix, vocab_size = prepare_lyrics(test_index, train_index, we_model)
        save_processed_data(embedding_matrix, tokenizer)

        # with open(os.path.join(data_folder, 'embedding_matrix.pickle'), 'rb') as handle:
        #     embedding_matrix = pickle.load(handle)
        # with open(os.path.join(data_folder, 'tokenizer.pickle'), 'rb') as handle:
        #     tokenizer = pickle.load(handle)

        train_index['lyrics_sequence'] = tokenizer.texts_to_sequences(train_index['lyrics'])
        test_index['lyrics_sequence'] = tokenizer.texts_to_sequences(test_index['lyrics'])

        test_df, test_midis, train_df, train_midis = load_midi_files(data_folder, test_index, train_index, 300)

        if melody_method == 1:
            melody_method = add_chroma
        else:
            melody_method = add_piano_roll
            add_tempo_feature(test_df, train_df)

        train_df['chroma_vectors'] = train_df.apply(melody_method, axis=1)
        test_df['chroma_vectors'] = test_df.apply(melody_method, axis=1)


        X_train, y_train = convert_data_to_model_input(train_df, seq_len)
        x_test, y_test = convert_data_to_model_input(test_df, seq_len)

        save_data_as_pickles(X_train, data_folder, x_test, y_test, y_train)
    else:
        X_train, y_train, x_test, y_test, tokenizer, embedding_matrix, vocab_size = load_preprocessed_data(data_folder)

    return X_train, y_train, x_test, y_test, tokenizer, embedding_matrix, vocab_size


def add_tempo_feature(test_df, train_df):
    train_tempo = train_df.apply(get_tempo, axis=1)
    test_tempo = test_df.apply(get_tempo, axis=1)
    scaler = MinMaxScaler()
    scaler.fit(pd.concat([train_tempo, test_tempo]).values.reshape(-1, 1))
    train_df['tempo'] = scaler.transform(train_tempo.values.reshape(-1, 1))
    test_df['tempo'] = scaler.transform(test_tempo.values.reshape(-1, 1))


def load_preprocessed_data(data_folder):

    with open(os.path.join(data_folder, 'train_data.pickle'), 'rb') as f:
        X_train, y_train = pickle.load(f)
    with open(os.path.join(data_folder, 'test_data.pickle'), 'rb') as f:
        x_test, y_test = pickle.load(f)
    with open(os.path.join(data_folder, 'embedding_matrix.pickle'), 'rb') as handle:
        embedding_matrix = pickle.load(handle)
    with open(os.path.join(data_folder, 'tokenizer.pickle'), 'rb') as handle:
        tokenizer = pickle.load(handle)

    vocab_size = embedding_matrix.shape[0] - 1
    return X_train, y_train, x_test, y_test, tokenizer, embedding_matrix, vocab_size


def load_midi_files(data_folder, test_index, train_index, vocab_size):
    midi_path = os.path.join(data_folder, 'midi_files')
    train_midis, test_midis = read_midi_files(midi_path, train_index, test_index, vocab_size)
    train_df = pd.DataFrame.from_records(list(train_midis.values()), columns=['md', 'lyrics'])
    test_df = pd.DataFrame.from_records(list(test_midis.values()), columns=['md', 'lyrics'])
    return test_df, test_midis, train_df, train_midis


def save_data_as_pickles(X_train, data_folder, x_test, y_test, y_train):
    with open(os.path.join(data_folder, 'train_data.pickle'), 'wb') as f:
        pickle.dump([X_train, y_train], f)
    with open(os.path.join(data_folder, 'test_data.pickle'), 'wb') as f:
        pickle.dump([x_test, y_test], f)


# def load_data(data_path):
#     test_index, train_index = load_lyrics_data(data_path)
#
#     we_model = WordEmbedding()
#     # we_model = None
#
#     tokenizer, embedding_matrix, vocab_size = prepare_lyrics(test_index, train_index, we_model)
#
#     save_processed_data(embedding_matrix, tokenizer)
#
#     embedding_matrix, tokenizer = load_processed_data(embedding_matrix, tokenizer)
#
#     vocab_size = len(tokenizer.word_index)
#
#     midi_path = os.path.join(data_path, 'midi_files')
#     train_index['lyrics_sequence'] = tokenizer.texts_to_sequences(train_index['lyrics'])
#     test_index['lyrics_sequence'] = tokenizer.texts_to_sequences(test_index['lyrics'])
#     train_midis, test_midis = read_midi_files(midi_path, train_index, test_index, vocab_size)
#
#     return train_midis, train_index, test_midis, test_index, embedding_matrix, vocab_size, we_model

# First melody representation
def add_chroma(row):
    md = row['md']
    lyrics_num = len(row['lyrics'])
    fs_value = md.get_end_time() / lyrics_num
    times_value = np.arange(0, md.get_end_time(), fs_value)
    return md.get_chroma(fs=fs_value, times=times_value).T

# Second melody representation
# TODO: check that it works
def add_piano_roll(row):
    md = row['md']
    lyrics_num = len(row['lyrics'])
    fs_value = md.get_end_time() / lyrics_num
    times_value = np.arange(0, md.get_end_time(), fs_value)
    return md.get_piano_roll(times=times_value)

    # instruments_data = {}
    # for instrument in md.instruments:
    #     if instrument.name in instrument_list:
    #         instruments_data[instrument.name] = instrument.get_chroma(fs=fs_value, times=times_value).T

    # return np.hstack(instruments_data.values())

def get_tempo(row):
    return row['md'].estimate_tempo()

def load_processed_data():
    with open('Data/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('Data/embedding_matrix.pickle', 'rb') as handle:
        embedding_matrix = pickle.load(handle)
    return embedding_matrix, tokenizer


def save_processed_data(embedding_matrix, tokenizer):
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('embedding_matrix.pickle', 'wb') as handle:
        pickle.dump(embedding_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_lyrics_data(data_path):
    frames_columns = ['artist', 'song_name', 'lyrics']
    train_path = os.path.join(data_path, 'lyrics_train_set.csv')
    train_index = load_lyrics(train_path, frames_columns)
    test_path = os.path.join(data_path, 'lyrics_test_set.csv')
    test_index = load_lyrics(test_path, frames_columns)
    test_index['song_name'] = [song_name[1:] for song_name in test_index['song_name'].values]
    return test_index, train_index


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
    for file in tqdm(os.listdir(path)):
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
                continue

        elif song_name in test_index['song_name'].values:
            try:
                pm = pretty_midi.PrettyMIDI(file_path)
                song_lyrics = test_index[test_index['song_name'] == song_name]['lyrics_sequence'].values[0]
                # one_hot_by_max_value(song_lyrics, vocab_size)

                test_midi_dict[file[:-4]] = [pm, song_lyrics]
            except:
                continue

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
    # TODO: add an option to adjust it to the new model structure
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