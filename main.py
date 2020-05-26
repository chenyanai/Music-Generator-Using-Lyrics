import Preprocessing
import Model
from WordEmbedding import WordEmbedding
import numpy as np
import pandas as pd

SEQ_LEN = 5

def add_chroma(row):
    md = row['md']
    lyric_vectors = row['lyrics_vectors']
    return md.get_chroma(fs=md.get_end_time()/len(lyric_vectors), times=np.arange(0, md.get_end_time(),md.get_end_time()/len(lyric_vectors) )).T


we = WordEmbedding()
train_midis, train, test_midis, test = Preprocessing.load_data(r'Data')

train_df = pd.DataFrame.from_records(list(train_midis.values()), columns=['md','lyrics'])
test_df = pd.DataFrame.from_records(list(test_midis.values()), columns=['md','lyrics'])

train_df['lyrics_vectors'] = train_df['lyrics'].apply(we.get_all_vectors)
test_df['lyrics_vectors'] = test_df['lyrics'].apply(we.get_all_vectors)

train_df['chroma_vectors'] = train_df.apply(add_chroma, axis=1)
test_df['chroma_vectors'] = test_df.apply(add_chroma, axis=1)
# chrome = test_df.apply(add_chroma, axis=1)

X_train, y_train = Preprocessing.convert_data_to_model_input(train_df, SEQ_LEN)
x_test, y_test = Preprocessing.convert_data_to_model_input(test_df, SEQ_LEN)

mid_vector_size = len(x_test['melody_vectors'][0][0:5][0])

model = Model.build_model(sequence_length=SEQ_LEN, mid_data_len=mid_vector_size)

Model.train_model(model, X_train, y_train)
# Model.train_model(model, x_test, y_test)

# train_df['lyrics_shape'] = train_df['lyrics_vectors'].apply(len)
# train_df['chroma_shape'] = train_df['chroma_vectors'].apply(lambda x: x.shape)