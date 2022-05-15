# a list of all the books we are analyzing
DATA = glob_wildcards('data/{book}.txt').book

# the default rule
rule all:
    input:
        'results/results.txt', expand('results/acf_{book}.dat', book=DATA)

# count words in one of our books
# logfiles from each run are put in .log files"
rule count_words:
    input:
        wc='source/wordcount.py',
        book='data/{file}.txt'
    output: 'processed_data/{file}.dat'
    log: 'processed_data/{file}.log'
    shell:
        '''
            python {input.wc} {input.book} {output} >> {log} 2>&1
        '''

rule word_acf:
    input:
        acf='source/autocorrelation.py',
        book='data/{file}.txt',
        wcdata='processed_data/{file}.dat'
    output: 'results/acf_{file}.dat'
    threads: 4
    log: 'processed_data/acf_{file}.log'    
    shell:
        '''
            python {input.acf} {input.book} {input.wcdata} {output} >> {log} 2>&1
        '''


# generate results table
rule zipf_test:
    input:
        zipf='source/zipf_test.py',
        books=expand('processed_data/{book}.dat', book=DATA)
    params:
        nwords = 10
    output: 'results/results.txt'
    shell:  'python {input.zipf} {params.nwords} {input.books} > {output}'

