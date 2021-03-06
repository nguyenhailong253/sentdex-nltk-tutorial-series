## Grouping of words following a regex pattern (unlike chinking, which is removing)
import nltk
import ssl
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text) # training Punkt on the training text

tokenized = custom_sent_tokenizer.tokenize(sample_text) # split to sentences

def process_content():
    try:
        # for i in tokenized: # for each sentence
            words = nltk.word_tokenize(tokenized[0]) # split to array of words
            tagged = nltk.pos_tag(words) # tagging each word in the array (each element is a tuple of word and tagged value)
            
            ## We are trying to find chunk of words that follows the below regex pattern
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?} """ # RB is adverb tag
            # any form of adverb (RB.?) and we're looking for 0 or more of these (*)

            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)

            chunked.draw()
            # print(chunked)
    except Exception as e:
        print(e)

process_content()