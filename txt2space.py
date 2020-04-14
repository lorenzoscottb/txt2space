
__author__ = 'lb540'

import numpy as np
import nltk
import math
from scipy.stats.stats import spearmanr
import pandas as pd
from tqdm import tqdm
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
        self.vec_dim = None

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

    def txt2space(self, dir, token=None, dimension=None, en_remove=False, dim_in_file=False, dtype="float64"):

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
        self.vec_dim = dimension

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

    def relpron_evaluation(self, dataset='test', mix_gr_rel=True, retunr_result=False, collectretunr_result=False):
        if mix_gr_rel:
            self.relpron_mix_gr_evaluation(dataset=dataset, retunr_result=retunr_result)
        else:
            self.relpron_sep_gr_evaluation(dataset=dataset, retunr_result=retunr_result)

    def relpron_sep_gr_evaluation(self, dataset='test', collect_es=False, retunr_result=False): #test, dev, all 

        dev_or_test = dataset
        goldfile = 'tests/relpron.'+dev_or_test+'.txt'
        dataset = pd.read_csv(goldfile, header=None, sep=':', names=['terms', 'props'], engine='python')
        termxprp = {}
        flc = 0
        llc = 0

        defs = []   # definitions
        for line in open(goldfile):
            if line == 'SBJ friction: phenomenon that prevent slipping': 
                line = 'SBJ friction: phenomenon that prevent slip'
            defs.append(line.replace('\n', ''))
            
        for tr in dataset.terms:
            trm_v = {}
            trm_c = 0
            for pr in dataset.props:
                flc += 1
                try: #tg = 1 if tr+':'+pr in defs else 0
                    p = ' '.join([po.split('_')[0] for po in pr.split()])
                    vp = self.vector(p.split()[0])+self.vector(p.split()[2])+self.vector(p.split()[3])
                    sim = self.cosine_similarity(tr.split()[1].split('_')[0], vp)
                    trm_v.update({pr:sim})
                    trm_c += 1 if tr+':'+pr in defs else 0
                except Exception as e:
                    # print(e)
                    None
            if trm_c == 0:
                continue            
            termxprp[tr] = {}                
            trm_v = sorted(trm_v.items(), key=lambda x: x[1], reverse=True)
            trm_v = [i[0] for i in trm_v] # sort by c.s.
            termxprp[tr]['cnt'] = trm_c
            termxprp[tr]['tag'] = [1 if tr+':'+pr in defs else 0 for pr in trm_v]
            # termxprp[tr]['props'] = trm_v
        # rlpr_test_df = pd.DataFrame(rlpr_test)
        ap = []
        for t in termxprp.keys():
            tc = termxprp[t]['cnt']
            apk = 0
            for c, e in enumerate(termxprp[t]['tag']):
                llc += 1
                apk += (sum(termxprp[t]['tag'][:c])/(c+1))*e
            ap.append(apk/tc)
        r_map = sum(ap)/len(ap)

        if retunr_result:
            return r_map        
        print('relpron {:15} coverage:{}/{}, MAP:{:.3f}'.format(dev_or_test+',', llc, flc, r_map))

    def relpron_mix_gr_evaluation(self, dataset='test', retunr_result=False): #test, dev, all 

        dev_or_test = 'test'
        goldfile = 'tests/relpron.'+dev_or_test+'.txt'
        dataset = pd.read_csv(goldfile, header=None, sep=':', names=['terms', 'props'], engine='python')
        termxprp = {}
        flc = 0 #full lenght counf
        llc = 0 #local lenght counf

        defs = []   # definitions: term+proposition
        for line in open(goldfile):
            if line == 'SBJ friction: phenomenon that prevent slipping': 
                line = 'friction: phenomenon that prevent slip'
            defs.append(line.split(' ', 1)[1].replace('\n', ''))

        for tr in dataset.terms:
            trm_v = {}
            trm_c = 0
            ttt = tr.split()[1]
            for pr in dataset.props:
                flc += 1
                try: #tg = 1 if tr+':'+pr in defs else 0
                    p = ' '.join([po.split('_')[0] for po in pr.split()])
                    vp = self.vector(p.split()[0])+self.vector(p.split()[2])+self.vector(p.split()[3])
                    sim = self.cosine_similarity(ttt.split('_')[0], vp)
                    trm_v.update({pr:sim})
                    trm_c += 1 if ttt+':'+pr in defs else 0
                except Exception as e:
                    # print(e)
                    None
            if trm_c == 0:
                continue
            termxprp[ttt] = {}
            trm_v = sorted(trm_v.items(), key=lambda x: x[1], reverse=True)
            trm_v = [i[0] for i in trm_v] # sort by c.s.
            termxprp[ttt]['cnt'] = trm_c
            termxprp[ttt]['tag'] = [1 if ttt+':'+pr in defs else 0 for pr in trm_v]
            termxprp[ttt]['props'] = trm_v
        # rlpr_test_df = pd.DataFrame(rlpr_test)
        ap = []
        for t in termxprp.keys():
            tc = termxprp[t]['cnt']
            apk = 0
            for c, e in enumerate(termxprp[t]['tag']):
                llc += 1
                apk += (sum(termxprp[t]['tag'][:c])/(c+1))*e
            ap.append(apk/tc)
        r_map = sum(ap)/len(ap)

        if retunr_result:
            return r_map        
        print('relpron {:15} coverage:{}/{}, MAP:{:.3f}'.format(dev_or_test+',', llc, flc, r_map))

    def relpron_ES_mix_gr_evaluation(self, e_sp, o_sp, dep_sp, dataset='test', retunr_result=False): #test, dev, all 

        dev_or_test = 'test'
        goldfile = 'tests/relpron.'+dev_or_test+'.txt'
        dataset = pd.read_csv(goldfile, header=None, sep=':', names=['terms', 'props'], engine='python')
        termxprp = {}
        flc = 0 #full lenght counf
        llc = 0 #local lenght counf

        defs = []   # definitions: term+proposition
        for line in open(goldfile):
            if line == 'SBJ friction: phenomenon that prevent slipping': 
                line = 'friction: phenomenon that prevent slip'
            defs.append(line.split(' ', 1)[1].replace('\n', ''))

        for tr in dataset.terms:
            trm_s = {}
            trm_e ={}
            trm_c = 0
            ttt = tr.split()[1]
            for pr in dataset.props:
                flc += 1
                try: #tg = 1 if tr+':'+pr in defs else 0
                    p = ' '.join([po.split('_')[0] for po in pr.split()])
                    vp = self.vector(p.split()[0])+self.vector(p.split()[2])+self.vector(p.split()[3])
                    ss = self.cosine_similarity(ttt.split('_')[0], vp)
                    es = relpron_es(ttt, pr, e_sp, o_sp, dep_sp)
                    trm_s.update({pr:ss})
                    trm_e.update({pr:es})
                    trm_c += 1 if ttt+':'+pr in defs else 0
                except Exception as e:
                    # print(e)
                    None
            if trm_c == 0:
                continue
            termxprp[ttt] = {}
            trm_s = [i[0] for i in sorted(trm_s.items(), key=lambda x: x[1], reverse=True)]
            trm_e = [i[0] for i in sorted(trm_e.items(), key=lambda x: x[1], reverse=True)]
            termxprp[ttt]['cnt'] = trm_c
            termxprp[ttt]['tag_s'] = [1 if ttt+':'+pr in defs else 0 for pr in trm_s]
            termxprp[ttt]['tag_e'] = [1 if ttt+':'+pr in defs else 0 for pr in trm_e]
            # termxprp[ttt]['propss'] = trm_ss
        # rlpr_test_df = pd.DataFrame(rlpr_test)
        ap_s = []
        ap_e = []
        for t in termxprp.keys():
            tc = termxprp[t]['cnt']
            apk_s = 0
            apk_e = 0
            for c, e in enumerate(termxprp[t]['tag_s']):
                llc += 1
                apk_s += (sum(termxprp[t]['tag_s'][:c])/(c+1))*e
                apk_e += (sum(termxprp[t]['tag_e'][:c])/(c+1))*termxprp[t]['tag_e'][c]
            ap_s.append(apk_s/tc)
            ap_e.append(apk_e/tc)

        r_map_s = sum(ap_s)/len(ap_s)
        r_map_e = sum(ap_e)/len(ap_e)

        if retunr_result:
            return r_map_s, r_map_e        
        print('relpron {:15} coverage:{}/{}, MAP:{:.3f}'.format(dev_or_test+',', llc, flc, r_map_s))
        print('relpron {:15} coverage:{}/{}, MAP ES:{:.3f}'.format(dev_or_test+',', llc, flc, r_map_e))

    def ml_10_evaluation(self, test_phrase='adjectivenouns', plus_en=False, plot=False, retunr_result=False, 
        print_ex=False, ml_10='tests/mitchell-lapata/ml_10.csv'):

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
        if type(test_phrase) == list: test_phrase = 'all'
        print('testing {:15} coverage:{}/{}, spearmanr:{:.3f}, p:{:.3f}'.format(test_phrase+',', c, test_values, 
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
        if type(test_phrase) == list: test_phrase = 'all'
        else: test_phrase = dep_ph_dict[test_phrase]
        print('testing {:15} coverage:{}/{}, spearmanr:{:.3f} p:{:.3f}'.format(test_phrase+' trsfrm,', c, test_values, 
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

        print('Simlex,    coverage:{}/{},   spearmanr:{:.3f}, p:{:.3f}'.format(len(sim_values), len(df.values),
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

        print('MEN (sim), coverage:{}/{}, spearmanr:{:.3f}, p:{:.3f}'.format(len(sim_values),len(df.values),
                                                                             corr[0], corr[1]))

    def ws353_evaluation(self, datasets='sim', distance='cosine', retunr_result=False, print_ex=False):

        ws353_ag = 'tests/wordsim353_sim_rel/wordsim353_agreed.csv'
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

        print('WS353 {}, coverage:{}/{},   spearmanr:{:.3f}, p:{:.3f}'.format(str(datasets), len(ws_values), 
                                                                              len(ws_df.values), corr[0], corr[1]))

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
    
    def run_tests(self, relpron_ds='test', mix_gr_rel=True, collect_es=False, retunr_results=False):
        if retunr_results:
            s,m,ws,wr = self.wordsim_evaluations(retunr_results=retunr_results)
            a,v,c,l = self.ml_eval(retunr_results=retunr_results)
            r = self.relpron_evaluation(dataset=relpron_ds, mix_gr_rel=mix_gr_rel, collect_es=collect_es,
                                        retunr_results=retunr_results)
            return s,m,ws,wr,a,v,c,l
        else:
            self.wordsim_evaluations()
            self.relpron_evaluation(dataset=relpron_ds, mix_gr_rel=mix_gr_rel)
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


def relpron_es(trgt, pr, e_sp, o_sp, d_sp):

    """
    DM paper eneched sum.; E: embeddings space; O: output space; T dep space; 
    target, verb, agent, patient
    cos(E$_{t}$,  E$_{a}$ + T$_{subj−1}$ O$_{v}$ + T$_{nsubj−1}$T$^{T}_{dobj−1}$E$_{p}$) +
    cos(T$_{nsubj}$O$_{t}$, T$_{nsubj}$O$_{a}$ + E$_{v}$ + T$_{dobj}$O$_{p}$)

    parameters
    trgt: target (e.g. expert_N)
    pr: preposition (e.g. 'quality_N' that tail_N aid_V')
    E/e_sp: input space; O/o_sp: output space; T/d_sp dep space
    """

    Et = e_sp.vector(trgt.split('_')[0]) 
    ppp = pr.split()
    d = e_sp.vec_dim
    Ea = e_sp.vector(ppp[0].split('_')[0])

    if "_V" in ppp[2]:
        TOv = d_sp.vector('_nsubj').reshape(d,d).dot(o_sp.vector(ppp[2].split('_')[0]))
        Ep = e_sp.vector(ppp[3].split('_')[0])
        TTtEp = (d_sp.vector('_nsubj').reshape(d,d)*d_sp.vector('_dobj').reshape(d,d).transpose()).dot(Ep)
    else:
        TOv = d_sp.vector('_nsubj').reshape(d,d).dot(o_sp.vector(ppp[3].split('_')[0]))
        Ep = e_sp.vector(ppp[3].split('_')[0])
        TTtEp = (d_sp.vector('_nsubj').reshape(d,d)*d_sp.vector('_dobj').reshape(d,d).transpose()).dot(Ep)
    prp_1 = Ea+TOv+TTtEp
    cs_1 = e_sp.cosine_similarity(Et, prp_1)

    TOt = d_sp.vector('nsubj').reshape(d,d).dot(o_sp.vector(trgt.split('_')[0]))
    TOa = d_sp.vector('nsubj').reshape(d,d).dot(o_sp.vector(ppp[0].split('_')[0]))
    if "_V" in ppp[2]:
        Ev = e_sp.vector(ppp[2].split('_')[0])
        TOp = d_sp.vector('dobj').reshape(d,d).dot(o_sp.vector(ppp[3].split('_')[0]))
    else:
        Ev = e_sp.vector(ppp[3].split('_')[0])
        TOp = d_sp.vector('dobj').reshape(d,d).dot(o_sp.vector(ppp[3].split('_')[0]))
    prp_2 = TOa+Ev+TOp
    cs_2= e_sp.cosine_similarity(TOt, prp_2)
    ES = cs_1+cs_2

    return ES
