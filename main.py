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

start_fresh = True

def add_chroma(row):
    md = row['md']
    lyric_vectors = row['lyrics']
    return md.get_chroma(fs=md.get_end_time()/len(lyric_vectors), times=np.arange(0, md.get_end_time(),md.get_end_time()/len(lyric_vectors) )).T

if not os.path.exists(TRAIN_PATH) or start_fresh:

    # we = WordEmbedding()
    # train_midis, train, test_midis, test = Preprocessing.load_data(r'Data')

    we = None
    train_midis, train, test_midis, test, embedding_matrix, vocab_size = Preprocessing.load_data(r'Data', we)

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
    with open(TRAIN_PATH, 'rb') as f:
        X_train, y_train = pickle.load(f)
    with open(TEST_PATH, 'rb') as f:
        x_test, y_test = pickle.load(f)


mid_vector_size = len(x_test['melody_vectors'][0][0:5][0])

model = Model.build_model(sequence_length=SEQ_LEN, mid_data_len=mid_vector_size,
                          embedding_matrix=embedding_matrix,vocab_size=vocab_size)

Model.train_model(model, X_train, y_train, vocab_size)
# Model.train_model(model, x_test, y_test)

# train_df['lyrics_shape'] = train_df['lyrics_vectors'].apply(len)
# train_df['chroma_shape'] = train_df['chroma_vectors'].apply(lambda x: x.shape)