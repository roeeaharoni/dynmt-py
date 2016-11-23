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
    print 'loading data from file:', path
    with codecs.open(path, encoding='utf8') as f:
        print 'tokenizing...'
        # i = 0
        for line in f:
            # tokens = nlp(line)
            tokens = line.split()
            tokenized.append(tokens)
            vocab += tokens
            # i += 1
            # if i % 100 == 0:
            #     print 'processed {} lines'.format(i)

    mcommon = [ite for ite, it in Counter(vocab).most_common(vocab_size)]
    return tokenized, list(set(mcommon))