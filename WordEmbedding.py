from gensim.models import KeyedVectors
class WordEmbedding:

    def __init__(self, path='Data/GoogleNews-vectors-negative300.bin'):
        # Load vectors directly from the file
        self.model = KeyedVectors.load_word2vec_format(path, binary=True)

    def get_word_vec(self, word:str):
        if word == '':
            return None
        try:
            return self.model[word]
        except Exception as e:
            return None

    def get_all_vectors(self, sentence:str):
        vectors = [self.get_word_vec(x) for x in sentence.split(' ')]
        return list(filter(lambda x: x is not None , vectors))

    def get_closest_word(self, vector):
        self.model.similar_by_vector(vector, topn=1)