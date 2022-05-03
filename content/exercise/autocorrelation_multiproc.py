import sys
import numpy as np
from wordcount import load_word_counts, load_text, DELIMITERS
import time
from multiprocessing import Pool
from functools import partial

def preprocess_text(text):
    """
    Remove delimiters, split lines into words and remove whitespaces, 
    and make lowercase. Return list of all words in the text.
    """
    clean_text = []
    for line in text:
        for purge in DELIMITERS:
            line = line.replace(purge, " ")    
        words = line.split()
        for word in words:
            word = word.lower().strip()
            clean_text.append(word)
    return clean_text

def word_acf(word, text, timesteps):
    """
    Calculate word-autocorrelation function for given word 
    in a text. Each word in the text corresponds to one "timestep".
    """
    acf = np.zeros((timesteps,))
    mask = [w==word for w in text]
    nwords_chosen = np.sum(mask)
    nwords_total = len(text)
    for t in range(timesteps):
        for i in range(1,nwords_total-t):
            acf[t] += mask[i]*mask[i+t]
        acf[t] /= nwords_chosen      
    return acf
    
def ave_word_acf(words, text, timesteps=100):
    """
    Calculate an average word-autocorrelation function 
    for a list of words in a text.
    """
    acf = np.zeros((len(words), timesteps))
    for n, word in enumerate(words):
        acf[n, :] = word_acf(word, text, timesteps)
    return np.average(acf, axis=0)

def ave_word_acf_pool(words, text, nproc=4, timesteps=100):
    """
    Calculate an average word-autocorrelation function 
    for a list of words in a text using multiprocessing.
    """
    word_acf_partial = partial(word_acf, text=text, timesteps=timesteps)
    with Pool(nproc) as p:
        results = p.map(word_acf_partial, words)
    acf = np.array(results)
    return np.average(acf, axis=0)

def setup(book, wc_book, nwords = 16):
    # load book text and preprocess it
    text = load_text(book)
    clean_text = preprocess_text(text)
    # load precomputed word counts and select top words
    word_count = load_word_counts(wc_book)
    top_words = [w[0] for w in word_count[:nwords]]

    return clean_text, top_words

if __name__ == '__main__':

    book = sys.argv[1]
    wc_book = sys.argv[2]
    filename = sys.argv[3]    

    nwords = 16
    timesteps = 100
    clean_text, top_words = setup(book, wc_book, nwords)

    # compute average autocorrelation and time the execution
    t0 = time.time()
    acf_ave = ave_word_acf(top_words, clean_text, timesteps)
    t1 = time.time()        
    nproc = 4
    acf_pool_ave = ave_word_acf_pool(top_words, clean_text, nproc, timesteps)
    t2 = time.time()        

    # assert that multiprocessing solution gives correct results
    np.testing.assert_array_equal(acf_ave, acf_pool_ave)
    
    print(f"serial time: {t1-t0}")
    print(f"parallel map time: {t2-t1}")

    np.savetxt(sys.argv[3], np.vstack((np.arange(1,101), acf_ave)).T, delimiter=',')
