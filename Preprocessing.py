import pandas as pd
import pretty_midi
import os
import re

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

    for file in os.listdir(path):
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
                pm.lyrics = str(test_index[test_index['song_name'] == song_name]['lyrics'].values[0])
                train_midi_dict[file[:-4]] = [pm, test_index[test_index['song_name'] == song_name]['lyrics'].values[0]]
            except:
                print(file)

    return train_midi_dict, test_midi_dict

def get_song_name_from_file_name(file_name):
    name = file_name[:-4] # remove file type
    name = name[name.find('-') + 2:]
    name = name.replace('_', ' ')
    return name.lower()

# train_midis, train, test_midis, test = load_data(r'Data')
# print('hi')