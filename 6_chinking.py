## Removal of something (chink something from a chunk)
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
            
            chunkGram = r"""Chunk: {<.*>+} 
                                    }<VB.?|IN|DT|TO>{""" # Chink will be in between "}{"

            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)

            chunked.draw()
            # print(chunked)
    except Exception as e:
        print(e)

process_content()