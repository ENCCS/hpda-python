import sys
import numpy as np
from wordcount import load_word_counts, load_text, DELIMITERS
import time
import numba

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

@numba.jit(nopython=True, cache=True)
def word_acf_numba(mask, timesteps):
    """
    Calculate word-autocorrelation function for given word 
    in a text. Each word in the text corresponds to one "timestep".
    """
    acf = np.zeros((timesteps,))
    nwords_chosen = np.sum(mask)
    nwords_total = mask.size
    for t in range(timesteps):
        for i in range(1,nwords_total-t):
            acf[t] += mask[i]*mask[i+t]
        acf[t] /= nwords_chosen      
    return acf

@numba.jit(nopython=False, cache=True)
def word_acf_numba_py(word, text, timesteps):
    """
    Calculate word-autocorrelation function for given word 
    in a text. Each word in the text corresponds to one "timestep".
    """
    acf = np.zeros((timesteps,))
    mask = np.array([w==word for w in text]).astype(np.float64)

    nwords_chosen = np.sum(mask)
    nwords_total = len(text)
    for t in range(timesteps):
        for i in range(1,nwords_total-t):
            acf[t] += mask[i]*mask[i+t]
        acf[t] /= nwords_chosen      
    return acf

def word_acf_numpy(word, text, timesteps):
    """
    Calculate word-autocorrelation function for given word 
    in a text using numpy.correlate function. 
    Each word in the text corresponds to one "timestep".
    """
    mask = np.array([w==word for w in text]).astype(np.float64)
    acf = np.correlate(mask, mask, mode='full') / np.sum(mask)
    return acf[int(acf.size/2):int(acf.size/2)+timesteps]


def ave_word_acf(words, text, timesteps=100):
    """
    Calculate an average word-autocorrelation function 
    for a list of words in a text.
    """
    acf = np.zeros((len(words), timesteps))
    for n, word in enumerate(words):
        acf[n, :] = word_acf(word, text, timesteps)
    return np.average(acf, axis=0)

def ave_word_acf_numpy(words, text, timesteps=100):
    """
    Calculate an average word-autocorrelation function 
    for a list of words in a text.
    """
    acf = np.zeros((len(words), timesteps))
    for n, word in enumerate(words):
        acf[n, :] = word_acf_numpy(word, text, timesteps)
    return np.average(acf, axis=0)

def ave_word_acf_numba_py(words, text, timesteps=100):
    """
    Calculate an average word-autocorrelation function 
    for a list of words in a text.
    """
    acf = np.zeros((len(words), timesteps))
    for n, word in enumerate(words):
        acf[n, :] = word_acf_numba_py(word, text, timesteps)
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

    nwords = 10
    timesteps = 100
    clean_text, top_words = setup(book, wc_book, nwords)

    # compute average autocorrelation and time the execution
    t0 = time.time()
    acf_ave = ave_word_acf(top_words, clean_text, timesteps)
    t1 = time.time()        

    # numpy version using np.correlate
    acf_ave_numpy = ave_word_acf_numpy(top_words, clean_text, timesteps)
    t2 = time.time()  

    # numba version with nopython=True (needs rewrite of code to "remove" Python)
    acf_ave_numba = np.zeros((len(top_words), timesteps))
    for n, word in enumerate(top_words):
        mask = np.array([w==word for w in clean_text]).astype(np.float64)
        acf_ave_numba[n, :] = word_acf_numba(mask, timesteps)
    acf_ave_numba = np.average(acf_ave_numba, axis=0)
    t3 = time.time()

    # numba version with nopython=False
    acf_ave_numba_py = ave_word_acf_numba_py(top_words, clean_text, timesteps=100)
    t4 = time.time()  

    print(f"plain python execution time: {t1-t0}")
    print(f"numpy execution time: {t2-t1}")
    print(f"numba (nopython=True) execution time: {t3-t2}")
    print(f"numba (nopython=False) execution time: {t4-t3}")

    np.testing.assert_approx_equal(np.sum(acf_ave - acf_ave_numpy), 0.0)
    np.testing.assert_approx_equal(np.sum(acf_ave - acf_ave_numba), 0.0)
    np.testing.assert_approx_equal(np.sum(acf_ave - acf_ave_numba_py), 0.0)

    np.savetxt(filename, np.vstack((np.arange(1,101), acf_ave)).T, delimiter=',')
