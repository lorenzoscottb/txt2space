
__author__ = 'lb540'

import numpy as np
import nltk
import math
from scipy.stats.stats import spearmanr
import pandas as pd
from sklearn.neighbors import NearestNeighbors as NN

class Space():

    """"
    WARNING: many txt spaces main interest of this class are in language_word format
            there for some methods have the plus_en parameter, you can set if to False
            if space's words are in en_word format but stimuli are not.
    """

    def __init__(self):
        self.vocabulary = {}

    def clear_space(self):
        self.vocabulary = {}
        self.matrix_space = None

    def key_by_value(self, value):
        return list(self.vocabulary.keys())[list(self.vocabulary.values()).index(value)]

    def vulic_lear_txt2space(self):

        file = open('/Users/lb540/Documents/corpora/vectors/lear_hyperlex.txt', 'r')

        self.matrix_space = np.ndarray(shape=(183870, 300))  # (183870, 300)

        line = file.readline()
        for index, line in enumerate(file):
            self.vocabulary[line.split(' ', 1)[0]] = index
            self.matrix_space[index] = np.fromstring(line.split(' ', 1)[1], dtype="float32", sep=" ")

    def txt2space(self, dir, token=None, dimension=None, en_remove=True, dim_in_file=False):

        file = open(dir, 'r')
        line = file.readline()
        if dim_in_file:
            info = line.split()
            token = int(info[0])
            dimesion = int(info[1])

        self.matrix_space = np.ndarray(shape=(token, dimension))

        for index, line in enumerate(file):
            if en_remove:
                self.vocabulary[line.split(' ', 1)[0].replace('en_', '')] = index
            else:
                self.vocabulary[line.split(' ', 1)[0]] = index
            self.matrix_space[index] = np.fromstring(line.split(' ', 1)[1], dtype="float32", sep=" ")

    def to_csr_matrix(self):
        
        import scipy.sparse.to_csr_matrix as csr
        return csr(self.matrix_space)

    def vector(self, word, plus_en=False):

        """
        extract vector for a given 'word'
        """

        if plus_en:
            word = 'en_'+word

        return self.matrix_space[self.vocabulary[word]]

    def multi_vec(self, words, plus_en=False):

        """
        extract and summ mutiple vectors from a set of words
        words: set of target words, can be a string of list of token
        """

        en_stop = nltk.corpus.stopwords.words('english')

        if type(words) == str:
            words = nltk.word_tokenize(words)
        v = np.zeros(self.matrix_space.shape[1], dtype=np.float32)

        if plus_en:
            words = ['en_'+word for word in words]

        for word in words:
            if word not in en_stop:
                v += self.vector(word.strip('.1'))  # pandas does not like repetitions...

        return v

    def norm(self, vector):

        from numpy.linalg import norm as nm

        if type(vector) == str:
            return nm(self.vector(vector))

        else:
            return nm(vector)

    def cosine_similarity(self, v1, v2):  # no difference for the order

        "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"

        if type(v1) == str:
            v1 = self.vector(v1)
        if type(v2) == str:
            v2 = self.vector(v2)

        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(v1)):
            x = v1[i]; y = v2[i]
            sumxx += x*x
            sumyy += y*y
            sumxy += x*y
        return sumxy/math.sqrt(sumxx*sumyy)    

    def extract_knn(self, word, algorithm='brute', n_nbrs=16,
                    metric='cosine', return_distance=False, plus_en=False):

        """
        word: single word string or array vector (for composed vectors)
        space: (x,y) matrix, the semantic space
        """

        space = self.matrix_space
        mydict = self.vocabulary

        nn = NN(algorithm=algorithm, n_neighbors=n_nbrs, metric=metric)
        nn.fit(space)

        if plus_en:
            word = 'en_'+word

        if type(word) == str:
            index = mydict[word]
            vector = [space[index]]
            v1 = space[mydict[word]]

        else:
            vector = [word]
            v1 = word

        word_nn = nn.kneighbors(vector, return_distance=return_distance)

        final_knn = []
        for e in word_nn[0]:
            if self.key_by_value(e) != word:

                v2 = self.matrix_space[e]

                final_knn.append((self.key_by_value(e), self.cosine_similarity(v1, v2)))

        return final_knn

    def generate_tsne(self, path=None, size=(10, 7), word_count=1000,
                      embeddings=None, plot=False):

        """
        adapted from github repo GradySimon/tensorflow-glove,
        generates semantic space graph
        """

        if embeddings is None:
            embeddings = self.matrix_space
        from sklearn.manifold import TSNE
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_dim_embs = tsne.fit_transform(embeddings[:word_count, :])
        labels = list(self.vocabulary.keys())[:word_count]
        return _plot_with_labels(low_dim_embs, labels, path, size)

    def ml_10_evaluation(self, test_phrase='adjectivenouns', plus_en=False, plot=False,
    					 ml_10='/tests/ml_10.csv'):

        df = pd.read_csv(ml_10)
        ml_values = []
        cs_values = []
        c = 0
        test_values = int(len(df.values)/3)
        if test_phrase == 'all':
            test_phrase = list(set(df['type']))
            test_values = int(len(df.values))
        for index, e in enumerate(df.values):
            if  e[1] not in test_phrase:
                continue
            try:
                c_1  = self.vector(e[3], plus_en=plus_en) + self.vector(e[4], plus_en=plus_en)
                c_2  = self.vector(e[5], plus_en=plus_en) + self.vector(e[6], plus_en=plus_en)
                cs_values.append(self.cosine_similarity(c_1, c_2))
                ml_values.append(int(e[-1]))
                # print('%s:collected' % e[3:7])
                c += 1
            except Exception as e: 
                print(e)
                
        print('testing {}, coverage {}/{}: {}'.format(test_phrase, c, test_values, 
                                             spearmanr(ml_values, cs_values)))
        if plot:

            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set(style="whitegrid")
            plt.scatter(ml_values,cs_values)

    def simlex_evaluation(self,  en=False, plot=False,
    					  simlex='tests/SimLex-en.csv'):

        df = pd.read_csv(simlex)
        sim_values = []
        cs_values = []

        for index, e in enumerate(df.values):
            sim_values.append(int(e[-1]))
            cs_values.append(self.cosine_similarity(e[0], e[1]))
#             print('%s:evaluated' % e[0:2])

        print(' vectors ', spearmanr(sim_values, cs_values))

        if plot:

            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set(style="whitegrid")
            plt.scatter(sim_values,cs_values)
            
    def MEN_evaluation(self, n=False, plot=False, 
                       men='tests/MEN_dataset_natural_form_full.csv'):
    
        df = pd.read_csv(men)
        sim_values = []
        cs_values = []

        for index, e in enumerate(df.values):
            try:
                cs_values.append(self.cosine_similarity(self.vector(e[0]), self.vector(e[1])))
                sim_values.append(int(e[-1]))

        #         print('%s:evaluated' % e[0:2])
            except Exception as ex:
                print(ex)

        print('MEN(sim) test, coverage:',len(sim_values),'/',len(df.values),spearmanr(sim_values, cs_values))    
        
    def ws353_evaluation(self, datasets='agrred'):
        
        ws353_ag = '/tests/'\
                    'wordsim353_sim_rel/wordsim353_agreed.csv'
        ws353_gs = 'tests/wordsim353_sim_rel/'\
                    'wordsim_similarity_goldstandard.csv'

        if datasets != 'gold_standars':
            ws353 = ws353_ag
        else:    
            ws353 = ws353_gs
        cs_values = []
        ws_values = []
        ws_df = pd.read_csv(ws353)
        for index, ws_e in enumerate(ws_df.values):
            try:
                cs_values.append(self.cosine_similarity(self.vector(ws_e[1]), 
                                                        self.vector(ws_e[2])))
                ws_values.append(int(ws_e[-1]))
        #         print('%s:evaluated' % e[0:2])
            except Exception as ex:
                print(ex)

        print('ws353', str(datasets),'test, coverage:',len(ws_values),'/',len(ws_df.values),spearmanr(ws_values, cs_values))
        
    
def _plot_with_labels(low_dim_embs, labels, path, size):

    """
    from github repo GradySimon/tensorflow-glove
    """

    import matplotlib.pyplot as plt
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    figure = plt.figure(figsize=size)  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right',
                     va='bottom')
    plt.show()

    if path is not None:
        figure.savefig(path)
        plt.close(figure)
