import Preprocessing
import Model
from WordEmbedding import WordEmbedding
import numpy as np
import pandas as pd
import os
import pickle


SEQ_LEN = 20
TRAIN_PATH = 'Data/train_data.pickle'
TEST_PATH = 'Data/test_data.pickle'

start_fresh = False

def add_chroma(row):
    md = row['md']
    lyric_vectors = row['lyrics']
    return md.get_chroma(fs=md.get_end_time()/len(lyric_vectors), times=np.arange(0, md.get_end_time(),md.get_end_time()/len(lyric_vectors) )).T

if not os.path.exists(TRAIN_PATH) or start_fresh:

    # train_midis, train, test_midis, test = Preprocessing.load_data(r'Data')

    train_midis, train, test_midis, test, embedding_matrix, vocab_size, we_model = Preprocessing.load_data(r'Data')

    train_df = pd.DataFrame.from_records(list(train_midis.values()), columns=['md','lyrics'])
    test_df = pd.DataFrame.from_records(list(test_midis.values()), columns=['md','lyrics'])

    # train_df['lyrics_vectors'] = train_df['lyrics'].apply(we.get_all_vectors)
    # test_df['lyrics_vectors'] = test_df['lyrics'].apply(we.get_all_vectors)

    train_df['chroma_vectors'] = train_df.apply(add_chroma, axis=1)
    test_df['chroma_vectors'] = test_df.apply(add_chroma, axis=1)

    X_train, y_train = Preprocessing.convert_data_to_model_input(train_df, SEQ_LEN)
    x_test, y_test = Preprocessing.convert_data_to_model_input(test_df, SEQ_LEN)

    with open(TRAIN_PATH, 'wb') as f:
        pickle.dump([X_train, y_train], f)
    with open(TEST_PATH, 'wb') as f:
        pickle.dump([x_test, y_test], f)
else:
    # with open(TRAIN_PATH, 'rb') as f:
    #     X_train, y_train = pickle.load(f)
    with open(TEST_PATH, 'rb') as f:
        x_test, y_test = pickle.load(f)
    with open('embedding_matrix.pickle', 'rb') as handle:
        embedding_matrix = pickle.load(handle)
    vocab_size = embedding_matrix.shape[0]-1
    # we_model = WordEmbedding()

print(f'vocab_size: {vocab_size}')

mid_vector_size = len(x_test['melody_vectors'][0][0:5][0])

# model = Model.build_model(sequence_length=SEQ_LEN, mid_data_len=mid_vector_size,
#                           embedding_matrix=embedding_matrix,vocab_size=vocab_size)
# history= Model.train_model(model, X_train, y_train, vocab_size)
# Model.train_model(model, x_test, y_test)

# tokenizer = None
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    word_map = tokenizer.word_index

model = Model.load("model.h5")
model.summary()
song = Model.generate_song(model, x_test, words_dict=word_map, reverse_word_dict=reverse_word_map)
print(song)