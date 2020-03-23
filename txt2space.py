
__author__ = 'lb540'

import numpy as np
import nltk
import math
from scipy.stats.stats import spearmanr
import pandas as pd
from numpy import linalg as LA
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

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

    def txt2space(self, dir, token=None, dimension=None, en_remove=True, dim_in_file=False, dtype="float64"):

        file = open(dir, 'r')
        if dim_in_file:
            line = file.readline()
            info = line.split()
            token = int(info[0])
            dimension = int(info[1])
        else:
            token = len(file.readlines())
            file = open(dir, 'r')
            line = file.readline()
            dimension = len(np.fromstring(line.split(' ', 1)[1], dtype=dtype, sep=" "))

        self.matrix_space = np.ndarray(shape=(token, dimension))

        for index, line in enumerate(file):
            if en_remove:
                self.vocabulary[line.split(' ', 1)[0].replace('en_', '')] = index
            else:
                self.vocabulary[line.split(' ', 1)[0]] = index
            try:
                self.matrix_space[index] = np.fromstring(line.split(' ', 1)[1], dtype=dtype, sep=" ")
            except Exception as exc:
                print('problem with', line)

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

    def poincare_glove_dist(self, v1, v2):

        if type(v1) == str:
            v1 = self.vector(v1)
        if type(v2) == str:
            v2 = self.vector(v2)

        return hyper_glv_dist(v1, v2)

    def extract_knn(self, word, algorithm='brute', n_nbrs=16, given_word=None,
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
            vector = space[index].reshape(1,self.matrix_space.shape[1])
            v1 = space[mydict[word]]

        else:
            vector = word.reshape(1,self.matrix_space.shape[1])
            v1 = word

        word_nn = nn.kneighbors(vector, return_distance=return_distance)

        final_knn = []
        for e in word_nn[0]:
            if self.key_by_value(e) != word:

                v2 = self.matrix_space[e]

                final_knn.append((self.key_by_value(e), self.cosine_similarity(v1, v2)))

        return final_knn

    def plot_space(self, method='tsne', word_count=1000, pick_random=True,
                      size=(10, 10), embeddings=None, path=None):

        """
        adapted from github repo GradySimon/tensorflow-glove,
        generates semantic space graph
        """

        if embeddings is None:
            embeddings = self.matrix_space
        
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        pca = PCA(n_components=2)

        if method == 'tsne':
            reduction_m = tsne
        elif method == 'pca':
            reduction_m = pca
        else: 
            print("No such method:", method)
            exit(1)

        if pick_random:
            mx = len(self.matrix_space)
            indx = np.random.randint(0, high=mx, size=word_count)
            low_dim_embs = reduction_m.fit_transform(np.take(self.matrix_space, indx, axis=0))
            labels = list(self.vocabulary.keys())[:word_count]
        else:
            low_dim_embs = reduction_m.fit_transform(embeddings[:word_count, :])
            labels = list(self.vocabulary.keys())[:word_count]

        return _plot_with_labels(low_dim_embs, labels, path, size)

    def ml_10_evaluation(self, test_phrase='adjectivenouns', plus_en=False, plot=False, retunr_result=False, 
        print_ex=False, ml_10='tests/ml_10.csv'):

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
                if print_ex:
                    print(e)

        corr = spearmanr(ml_values, cs_values)
        if retunr_result:
            return corr 

        print('testing {}, coverage:{}/{}, spearmanr:{:.3f}, p:{}'.format(test_phrase, c, test_values, 
                                                                          corr[0], corr[1]))
        if plot:
            plt.scatter(ml_values,cs_values)

    def ml_10_mx_trasnform(self, dep_sp, src_sp, test_phrase='adjectivenouns', retunr_result=False, plus_en=False, 
                        print_ex=False, ml_10='tests/ml_10.csv',
                        transpose=False, plot=False,):

        df = pd.read_csv(ml_10)
        ml_values = []
        cs_values = []
        c = 0
        dim = self.matrix_space.shape[1]
        dep_ph_dict = {'adjectivenouns':'_amod', 'verbobjects':'_dobj', 'compoundnouns':'_nmod'}
        test_values = int(len(df.values)/3)
        if test_phrase == 'all':
            test_phrase = list(set(df['type']))
            test_values = int(len(df.values))
        for index, e in enumerate(df.values):
            if  e[1] not in test_phrase:
                continue

            try:
                dep_lb = dep_ph_dict[e[1]]
                if transpose:
                    c_1  = dep_sp.vector(dep_lb).reshape(dim, dim).transpose().dot(src_sp.vector(e[3], plus_en=plus_en)) + self.vector(e[4], plus_en=plus_en) 
                    c_2  = dep_sp.vector(dep_lb).reshape(dim, dim).transpose().dot(src_sp.vector(e[5], plus_en=plus_en)) + self.vector(e[6], plus_en=plus_en)
                else:
                    c_1  = dep_sp.vector(dep_lb).reshape(dim, dim).dot(src_sp.vector(e[3], plus_en=plus_en)) + self.vector(e[4], plus_en=plus_en) 
                    c_2  = dep_sp.vector(dep_lb).reshape(dim, dim).dot(src_sp.vector(e[5], plus_en=plus_en)) + self.vector(e[6], plus_en=plus_en)
                cs_values.append(self.cosine_similarity(c_1, c_2))
                ml_values.append(int(e[-1]))
                # print('%s:collected' % e[3:7])
                c += 1
            except Exception as e:
                if print_ex:
                    print(e)
        corr = spearmanr(ml_values, cs_values)
        if retunr_result:
            return corr 

        print('testing {}, coverage:{}/{}, spearmanr:{:.3f} p:{}'.format(test_phrase+' trsfrm', c, test_values, 
                                                                          corr[0], corr[1]))
        if plot:
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.scatter(ml_values,cs_values)

    def simlex_evaluation(self, distance='cosine', en=False, plot=False, retunr_result=False ,
        print_ex=False, simlex='tests/SimLex-en.csv'):

        df = pd.read_csv(simlex)
        sim_values = []
        cs_values = []

        for index, e in enumerate(df.values):
            try:
                if distance=='cosine':
                    cs_values.append(self.cosine_similarity(e[0], e[1]))
                else:
                    cs_values.append(self.poincare_glove_dist(e[0], e[1]))
                sim_values.append(int(e[-1]))
#             print('%s:evaluated' % e[0:2])
            except Exception as ex:
               if print_ex:
                   print(ex)
        corr = spearmanr(sim_values, cs_values)
        if retunr_result:
            return corr 

        print('Simlex, coverage:{}/{}, spearmanr:{:.3f}, p:{}'.format(len(sim_values), len(df.values),
                                                                     corr[0], corr[1]))
        if plot:

            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set(style="whitegrid")
            plt.scatter(sim_values,cs_values)

    def MEN_evaluation(self, distance='cosine', n=False, plot=False, retunr_result=False, 
        print_ex=False, men='tests/MEN_dataset_natural_form_full.csv'):

        df = pd.read_csv(men)
        sim_values = []
        cs_values = []

        for index, e in enumerate(df.values):
            try:
                if distance=='cosine':
                    cs_values.append(self.cosine_similarity(e[0], e[1]))
                else:
                    cs_values.append(self.poincare_glove_dist(e[0], e[1]))
                sim_values.append(int(e[-1]))

        #         print('%s:evaluated' % e[0:2])
            except Exception as ex:
                if print_ex:
                    print(ex)
        corr = spearmanr(sim_values, cs_values)
        if retunr_result:
            return corr 

        print('MEN (sim), coverage:{}/{}, spearmanr:{:.3f}, p:{}'.format(len(sim_values),len(df.values),
                                                                     corr[0], corr[1]))

    def ws353_evaluation(self, datasets='sim', distance='cosine', retunr_result=False, print_ex=False):

        ws353_ag = 'testswordsim353_sim_rel/wordsim353_agreed.csv'
        ws353_gs_sim = 'tests/wordsim_similarity_goldstandard.csv'
        ws353_gs_rel = 'tests/wordsim_relatedness_goldstandard.csv'

        if datasets == 'sim':
            ws353 = ws353_gs_sim
        elif datasets == 'rel':
            ws353 = ws353_gs_rel
        else:
            ws353 = ws353_ag

        cs_values = []
        ws_values = []
        ws_df = pd.read_csv(ws353)
        for index, ws_e in enumerate(ws_df.values):
            try:
                if distance=='cosine':
                    cs_values.append(self.cosine_similarity(ws_e[1], ws_e[2]))
                else:
                    cs_values.append(self.poincare_glove_dist(ws_e[1], ws_e[2]))

                ws_values.append(int(ws_e[-1]))
        #         print('%s:evaluated' % e[0:2])
            except Exception as ex:
                if print_ex:
                    print(ex)
        corr = spearmanr(ws_values, cs_values)
        if retunr_result:
            return corr 

        print('WS353 {}, coverage:{}/{}, spearmanr:{:.3f}, p:{}'.format(str(datasets), len(ws_values), len(ws_df.values),
                                                                     corr[0], corr[1]))

    def ml_eval(self,retunr_results=False):
        a = self.ml_10_evaluation('adjectivenouns', retunr_result=retunr_results)
        v = self.ml_10_evaluation('verbobjects', retunr_result=retunr_results)
        c = self.ml_10_evaluation('compoundnouns', retunr_result=retunr_results)
        l = self.ml_10_evaluation('all', retunr_result=retunr_results)    

        if retunr_results:
            return a,v,c,l

    def ml_full_eval(self, dep_sp, src_sp, retunr_results=False, transpose=False):
        self.ml_10_evaluation(test_phrase='all', retunr_result=retunr_results)
        print('\n')
        self.ml_10_evaluation(test_phrase='adjectivenouns')
        self.ml_10_mx_trasnform(dep_sp, src_sp, test_phrase='adjectivenouns',transpose=transpose, retunr_result=retunr_results)
        print('\n')
        self.ml_10_evaluation(test_phrase='verbobjects', retunr_result=retunr_results)
        self.ml_10_mx_trasnform(dep_sp, src_sp, test_phrase='verbobjects', transpose=transpose, retunr_result=retunr_results)
        print('\n')
        self.ml_10_evaluation(test_phrase='compoundnouns', retunr_result=retunr_results)
        self.ml_10_mx_trasnform(dep_sp, src_sp, test_phrase='compoundnouns', transpose=transpose, retunr_result=retunr_results)

    def wordsim_evaluations(self, retunr_results=False):

        s = self.simlex_evaluation(retunr_result=retunr_results)
        m = self.MEN_evaluation(retunr_result=retunr_results)
        ws = self.ws353_evaluation(datasets='sim', retunr_result=retunr_results)
        wr = self.ws353_evaluation(datasets='rel', retunr_result=retunr_results)

        if retunr_results:
            return s,m,ws,wr
    
    def run_tests(self, retunr_results=False):
        
        if retunr_results:
            s,m,ws,wr = self.wordsim_evaluations(retunr_results=retunr_results)
            a,v,c,l = self.ml_eval(retunr_results=retunr_results)
            return s,m,ws,wr,a,v,c,l
        else:
            self.wordsim_evaluations()
            self.ml_eval()            

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

def lambda_vec(x):
    return 2/(1 - np.power(LA.norm(x, 2),2))


def hyper_glv_dist(x,y):

    return np.power(np.cosh(1 + lambda_vec(y)*lambda_vec(x)*np.power(LA.norm(x-y, 2),2)/2), -1)



