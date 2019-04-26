#!/usr/bin/python
import re, string, collections
from nltk.stem.porter import PorterStemmer
gPS = None
class TextUtils:
    def __init__(self):
        global gPS
        #print "Init Called"
        gPS = PorterStemmer()

    def clean_text(self, text):
        pattern = r"[{}]".format(string.punctuation)
        text = re.sub(pattern, " ", text)
        text = re.sub(r'\d+', " ", text)
        line = ''
        for word in text.split(" "):
            if len(word)>1:
                line = line + ' '+ word
        line = re.sub(r'\s+', ' ', line)
        line = re.sub(r'^\s+', '', line)
        line = re.sub(r'\s+$', '', line)
        return line

    def get_phrase_stem(self, phrase):
        stem = ''
        for word in self.clean_text(phrase).split(' '):
            stem = stem + " " + gPS.stem(word)
        return re.sub(r'^\s+|\s+$', '', stem)

    def get_word_stem(self, word):
        return gPS.stem(word)

class SpellCheck:
    def __init__(self):
        self.NWORDS = self.train(self.words(file('big.txt').read()))
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'

    def words(self, text):
        return re.findall('[a-z]+', text.lower())

    def train(self, features):
        model = collections.defaultdict(lambda: 1)
        for f in features:
            model[f] += 1
        return model

    def edits1(self, word):
        s = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes    = [a + b[1:] for a, b in s if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in s if len(b)>1]
        replaces   = [a + c + b[1:] for a, b in s for c in self.alphabet if b]
        inserts    = [a + c + b     for a, b in s for c in self.alphabet]
        return set(deletes + transposes + replaces + inserts)

    def known_edits2(self, word):
        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1) if e2 in self.NWORDS)

    def known(self, words):
        return set(w for w in words if w in self.NWORDS)

    def correct(self, word):
        candidates = self.known([word]) or self.known(self.edits1(word)) or self.known_edits2(word) or [word]
        return max(candidates, key=self.NWORDS.get)


class SplitText:
    def __init__(self):
        self.word_frequencies = {
        'file': 0.00123,
        'files': 0.00124,
        'save': 0.002,
        'ave': 0.00001,
        'as': 0.00555,
        'project': 0.00555,
        'management': 0.005255
        }

    def split_text(self, text, cache):
        if text in cache:
            return cache[text]
        if not text:
            return 1, []
        best_freq, best_split = 0, []
        for i in xrange(1, len(text) + 1):
            word, remainder = text[:i], text[i:]
            freq = self.word_frequencies.get(word, None)
            if freq:
                remainder_freq, remainder = self.split_text(remainder, cache)
                freq *= remainder_freq
                if freq > best_freq:
                    best_freq = freq
                    best_split = [word] + remainder
        cache[text] = (best_freq, best_split)
        return cache[text]

if __name__=='__main__':
    sp = SpellCheck()
    print sp.correct('projectmanagement')
    print SplitText().split_text('projectmanagement', {})
