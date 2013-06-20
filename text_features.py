
from nltk.stem.porter import PorterStemmer
import re

from mem import cache

class Stemmer(object):
    def __init__(self):
        self.splitter = re.compile("[^\w'`]+")
        self.stemmer = PorterStemmer()

    def stem_sentence(self, sentence):
        words = self.splitter.split(sentence)
        words = map(self.stemmer.stem, words)
        return ' '.join(words)


@cache
def stem(text):
    stemmer = Stemmer()
    return text.map(stemmer.stem_sentence)

