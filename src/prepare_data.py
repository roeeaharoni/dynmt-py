import codecs
import spacy
from collections import Counter

def load_data(path, vocab_size):

    # print 'loading preprocessing model...'
    # nlp = spacy.load('en')

    # doc = nlp(u'Hello, world. Here are two sentences.')
    # texts = [u'One document.', u'...', u'Lots of documents']
    # iter_texts = (texts[i % 3] for i in xrange(100000000))
    #
    # for i, doc in enumerate(nlp.pipe(iter_texts, batch_size=50, n_threads=4)):
    #     assert doc.is_parsed
    #     if i == 100:
    #         break

    # token = doc[0]
    # sentence = next(doc.sents)
    # assert token is sentence[0]
    # assert sentence.text == 'Hello, world.'

    tokenized = []
    vocab = []
    total_len = 0
    max_len = 0
    amount = 0
    print 'loading data from file:', path
    with codecs.open(path, encoding='utf8') as f:
        print 'tokenizing...'
        for line in f:
            tokens = line.split()
            amount += 1
            seq_len = len(tokens)
            total_len += seq_len
            if seq_len > max_len:
                max_len = seq_len
            tokenized.append(tokens)
            vocab += tokens
        print 'max len is {}, avg len is {}'.format(max_len, total_len/float(amount))

    mcommon = [ite for ite, it in Counter(vocab).most_common(vocab_size)]
    return tokenized, list(set(mcommon))