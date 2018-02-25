#!/usr/bin/python

import string

class Dictionary():
    def __init__(self, cmu_path='./cmudict-0.7b'):
        print('Loading CMU Dictionary')
        dictionary = {}
        with open(cmu_path,'r') as cmudict:
            for row in cmudict.readlines():
                if row[0] in string.punctuation:
                    continue
                items = row.strip().split(' ')
                dictionary[items[0]] = items[1:]
        self.dictionary = dictionary

    def read_wordlist(self, wl_path = './wordlist.txt'):
        wordlist = {}
        dictionary = self.dictionary
        sections = {}
        with open(wl_path,'r') as wl:
            for row in wl.readlines():
                if row[0] == '#':
                    comment = row[1:].strip()
                    sections[comment] = []
                word = row.strip().upper()
                sections[comment].append(word)
                try:
                    wordlist[word] = dictionary[word]
                except KeyError as e:
                    raise KeyError(str(e)+' not in dictionary.')
        self.wordlist = wordlist
        self.sections = sections

    def _possible_choices(self,word):
        sections = self.sections
        if word in self.wordlist:
            main_pronunciation = self.wordlist[word]
        else:
            try:
                main_pronunciation = self.dictionary[word]
            except KeyError as e:
                raise KeyError(str(e)+' not in dictionary.')
        rule = None
        for key in sections:
            if word in sections[key]:
                rule = key
        choices = {}
        mp_copy = main_pronunciation[:]
        if rule == 't-deletion':
            choices[0] = mp_copy
            choices[1] = mp_copy[0] + [x for x in mp_copy[1:] if x != 'T']
            return(choices)
        elif rule == 't-flapping':
            choices[0] = mp_copy
            choices[1] = [x if x != 'T' else 'D' for x in mp_copy]
            return(choices)
        elif rule == 'schwa-reduction':
            choices[0] = mp_copy
            choices[1] = [x for x in mp_copy if '0' not in x]
            return(choices)
