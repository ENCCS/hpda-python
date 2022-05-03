import sys
import numpy as np
from wordcount import load_word_counts, load_text, DELIMITERS
import time
from mpi4py import MPI


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

def ave_word_acf_p2p(comm, my_words, text, timesteps=100):
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()
    # each rank computes its own set of acfs
    my_acfs = np.zeros((len(my_words), timesteps))
    for i, word in enumerate(my_words):
        my_acfs[i,:] = word_acf(word, text, timesteps)

    if rank == 0:
        results = []
        # append own results
        results.append(my_acfs)
        # receive data from other ranks and append to results
        for sender in range(1, n_ranks):
            results.append(comm.recv(source=sender, tag=12))
        # compute total 
        acf_tot = np.zeros((timesteps,))
        for i in range(n_ranks):
            for j in range(len(results[i])):
                acf_tot += results[i][j]
        return acf_tot
    else:
        # send data
        comm.send(my_acfs, dest=0, tag=12)

def ave_word_acf_gather(comm, my_words, text, timesteps=100):
    rank = comm.Get_rank()
    n_ranks = comm.Get_size() 
    # each rank computes its own set of acfs
    my_acfs = np.zeros((len(my_words), timesteps))
    for i, word in enumerate(my_words):
        my_acfs[i,:] = word_acf(word, text, timesteps)

    # gather results on rank 0
    results = comm.gather(my_acfs, root=0)
    # loop over ranks and results. result is a list of lists of ACFs
    if rank == 0:
        acf_tot = np.zeros((timesteps,))
        for i in range(n_ranks):
            for j in range(len(results[i])):
                acf_tot += results[i][j]
        return acf_tot

def setup(book, wc_book, nwords = 16):
    # load book text and preprocess it
    text = load_text(book)
    clean_text = preprocess_text(text)
    # load precomputed word counts and select top words
    word_count = load_word_counts(wc_book)
    top_words = [w[0] for w in word_count[:nwords]]

    return clean_text, top_words

def mpi_acf(book, wc_book, nwords = 16, timesteps = 100):
    # initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    # load book text and preprocess it
    clean_text, top_words = setup(book, wc_book, nwords)
    
    # distribute words among MPI tasks
    count = nwords // n_ranks
    remainder = nwords % n_ranks
    # first 'remainder' ranks get 'count + 1' tasks each
    if rank < remainder:
        first = rank * (count + 1)
        last = first + count + 1
    # remaining 'nwords - remainder' ranks get 'count' task each
    else:
        first = rank * count + remainder
        last = first + count 
    # each rank gets unique words
    my_words = top_words[first:last]
    print(f"My rank number is {rank} and first, last = {first}, {last}")

    # use collective function
    acf_tot = ave_word_acf_gather(comm, my_words, clean_text, timesteps)

    # use p2p function
    #acf_tot = ave_word_acf_p2p(comm, my_words, clean_text, timesteps)

    # only rank 0 has the averaged data
    if rank == 0:
        return acf_tot / nwords

if __name__ == '__main__':
    # load book text and preprocess it
    book = sys.argv[1]
    wc_book = sys.argv[2]
    filename = sys.argv[3]    
    acf = mpi_acf(book, wc_book, 16, 100)

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        nsteps = len(acf)
        output = np.vstack((np.arange(1,nsteps+1), acf)).T
        np.savetxt(filename, output, delimiter=',')
