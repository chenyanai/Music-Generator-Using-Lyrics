import Preprocessing
from WordEmbedding import WordEmbedding
import numpy as np
import pandas as pd
import os

def add_chroma(row):
    md = row['md']
    lyric_vectors = row['lyrics_vectors']
    return md.get_chroma(fs=md.get_end_time()/len(lyric_vectors), times=np.arange(0, md.get_end_time(),md.get_end_time()/len(lyric_vectors) )).T

# chroma = md.get_chroma(fs=len(lyrics_vec), times=np.arange(0, md.get_end_time(),md.get_end_time()/len(lyrics_vec) ))
# chroma = md.get_chroma(fs=md.get_end_time()/len(lyrics_vec), times=np.arange(0, md.get_end_time(),md.get_end_time()/len(lyrics_vec) ))

# print(os.curdir)
print(os.listdir())
print(os.getcwd())
# os.chdir('DL-ass3')

we = WordEmbedding()
train_midis, train, test_midis, test = Preprocessing.load_data(r'Data')

train_df = pd.DataFrame.from_records(list(train_midis.values()), columns=['md','lyrics'])
test_df = pd.DataFrame.from_records(list(test_midis.values()), columns=['md','lyrics'])

train_df['lyrics_vectors'] = train_df['lyrics'].apply(we.get_all_vectors)
test_df['lyrics_vectors'] = test_df['lyrics'].apply(we.get_all_vectors)

train_df['chroma_vectors'] = train_df.apply(add_chroma, axis=1)
test_df['chroma_vectors'] = test_df.apply(add_chroma, axis=1)

# train_df['lyrics_shape'] = train_df['lyrics_vectors'].apply(len)
# train_df['chroma_shape'] = train_df['chroma_vectors'].apply(lambda x: x.shape)