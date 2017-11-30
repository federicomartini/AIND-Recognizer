import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose
        self.range_n_components = range(self.min_n_components, self.max_n_components + 1)

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        best_bic = float('inf')
        best_model = None
        
        for n in self.range_n_components:
            try:
                n_params = len(self.lengths)**2 + 2*n*len(self.lengths) - 1
                hmm_model = self.base_model(n) 
                bic = -2*(hmm_model.score(self.X, self.lengths)) + n_params*np.log(len(self.lengths))
                
                #Selection
                if bic < best_bic:
                    best_bic = bic
                    best_model = hmm_model
            except:
                pass
            
        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_dic = float('-inf')
        best_model = None
        
        for n in self.range_n_components:
            try:
                hmm_model = self.base_model(n)
                log_l_others = []
                
                for word in self.words.keys():
                    #Compare the score with other words
                    if word == self.this_word:
                        continue
                    else:
                        log_l_others.append(hmm_model.score(self.X, self.lengths))
                
                #Approximated version of the criterion
                dic = hmm_model.score(self.X, self.lengths) - np.mean(log_l_others)
                
                #Selection
                if dic > best_dic:
                    best_dic = dic
                    best_model = model
                    
            except:
                pass
        
        if best_model:
            return best_model
        
        return self.base_model(self.n_constant)
        


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        best_CV = float('-inf')
        best_model = None
        
        for n in self.range_n_components:
            try:
                #Split dataset into k consecutive folds
                method = KFold()
                log_Ls = []
                hmm_model = self.base_model(n)
                
                #For each sequence
                for _, test_idx in method.split(self.sequences):
                    #Retrieve the tests
                    test_x, test_lengths = combine_sequences(test_idx, self.sequences)
                    #Calculate the score and add it to the list of the scores
                    log_Ls.append(hmm_model.score(test_x, test_lengths))
                
                #Calculate the mean values of the scores
                mean_score = np.mean(log_Ls)
                
                #Select the best score for the CV
                if mean_score > best_CV:
                    best_CV = mean_score
                    best_model = hmm_model
            except:
                pass
        
        if best_model:
            return best_model
        
        return self.base_model(self.n_constant)
    