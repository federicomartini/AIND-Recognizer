import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    
    # TODO implement the recognizer
    # return probabilities, guesses
    x_lengths = test_set.get_all_Xlengths()
    n_sequences = len(test_set.get_all_sequences())

    for single_data in range(n_sequences):
        best_score = float('-inf')
        best_guess = None
        prob = {}
        X, lengths = x_lengths[single_data]
        
        for g_word, model in models.items():
            try:
                score = model.score(X, lengths)
                prob[g_word] = score
                
                #Selection
                if score > best_score:
                    best_score = score
                    guess_word = g_word
                    
            except:
                #failed process
                prob[g_word] = float('-inf')
                
        probabilities.append(prob)
        guesses.append(guess_word)
        
    return probabilities, guesses