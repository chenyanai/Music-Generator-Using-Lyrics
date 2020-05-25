import pandas as pd
import pretty_midi
from tqdm import tqdm
import os
import re
import numpy as np

DATA_PATH = r'\Data'

def load_data(data_path):
    frames_columns = ['artist', 'song_name', 'lyrics']

    train_path = os.path.join(data_path, 'lyrics_train_set.csv')
    train_index = pd.read_csv(train_path)
    train_index = train_index.iloc[:, : 3]
    train_index.columns = frames_columns
    train_index['lyrics'] = train_index.lyrics.apply(clean_text)

    test_path = os.path.join(data_path, 'lyrics_test_set.csv')
    test_index = pd.read_csv(test_path)
    test_index = test_index.iloc[:, : 3]
    test_index.columns = frames_columns
    test_index['song_name'] = [name[1:] for name in test_index['song_name']]
    test_index['lyrics'] = test_index.lyrics.apply(clean_text)

    midi_path = os.path.join(data_path, 'midi_files')
    train_midis, test_midis = read_midi_files(midi_path, train_index, test_index)

    return train_midis, train_index, test_midis, test_index


def clean_text(text:str)->str:
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = text.replace('&', '')
    text = text.replace('--', '')
    text = text.replace(':', '')
    # text = text.replace('[chorus:]', '')
    # text = text.replace('[chorus: ]', '')
    return text

def read_midi_files(path, train_index, test_index):
    train_midi_dict = {}
    test_midi_dict = {}

    for file in tqdm(os.listdir(path)):
        file_path = os.path.join(path, file)
        song_name = get_song_name_from_file_name(file)
        if song_name in train_index['song_name'].values:
            try:
                pm = pretty_midi.PrettyMIDI(file_path)
                # pm.lyrics = str(train_index[train_index['song_name'] == song_name]['lyrics'].values[0])
                train_midi_dict[file[:-4]] = [pm, train_index[train_index['song_name'] == song_name]['lyrics'].values[0]]
            except:
                print(file)
        elif song_name in test_index['song_name'].values:
            try:
                pm = pretty_midi.PrettyMIDI(file_path)
                # pm.lyrics = str(test_index[test_index['song_name'] == song_name]['lyrics'].values[0])
                test_midi_dict[file[:-4]] = [pm, test_index[test_index['song_name'] == song_name]['lyrics'].values[0]]
            except:
                print(file)

    return train_midi_dict, test_midi_dict

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
        'lyric_vectors': list(),
        'melody_vectors': list()
    }
    y = list()

    for i, song_data in df.iterrows():

        melody_data = song_data['chroma_vectors']

        if len(melody_data) >= sequence_length + 1:
            for j in range(len(melody_data)):
                if len(melody_data) - 1 > j + sequence_length :
                    X['lyric_vectors'].append(song_data['lyrics_vectors'][j])
                    X['melody_vectors'].append(song_data['chroma_vectors'][j:j + sequence_length])
                    y.append(song_data['lyrics_vectors'][j + sequence_length])

    return X, y
