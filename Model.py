from tensorflow.keras.layers import Flatten, Embedding, Dense, GRU, TimeDistributed, LSTM, Dropout, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.layers import TensorBoard
from tensorflow.keras import Model, Input

def build_model(sequence_length, mid_data_len, embedding_matrix, vocab_size):

    melody_input = Input(shape=(sequence_length, mid_data_len, ), name='melody_input')
    # lyrics_vectors_input = Input(shape=(300, ), name='lyrics_vectors_input')

    lyrics_embedding = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=4, trainable=False)

    melody_lstm = LSTM(300, return_sequences=False, name='melody_LSTM')(melody_input)

    lyrics_melody_concat = Concatenate(axis=-1, name='lyrics_melody_concat')([lyrics_embedding, melody_lstm])

    dense = Dense(1024, activation='relu', name='first_dense')(lyrics_melody_concat)
    dropout = Dropout(0.1)(dense)
    dense = Dense(vocab_size, activation='softmax', name='last_dense')(dropout) # Predicting the next locations of the defense players

    model = Model(inputs=[melody_input, lyrics_embedding], outputs=dense)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    return model


def train_model(model, X, y):
    model.fit(x=[X['melody_vectors'], X['lyric_vectors']], y=y, batch_size=256, epochs=3, verbose=1)
    return model