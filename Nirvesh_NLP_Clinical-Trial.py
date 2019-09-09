#!/usr/bin/env python
# coding: utf-8

# In[2]:


import nltk 
from nltk.corpus import gutenberg 
from nltk import word_tokenize, pos_tag, ne_chunk
import urllib2
from bs4 import BeautifulSoup
from pprint import pprint 
import os
import numpy as np
import string 
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer 
import sys
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import itertools
from gensim import corpora, models
from textblob import TextBlob
from nltk.wsd import lesk
from nltk import word_tokenize
from nltk.corpus import wordnet as wn
import pandas as pd
from scipy.stats import itemfreq
import xml.etree.ElementTree as ET
from xml.dom import minidom


# In[3]:


#reading corpus of interest from clinical trial website by xml parsing

xmldoc =minidom.parse('/Users/Nirvesh/Desktop/drug_abuse.xml')
study= xmldoc.getElementsByTagName("clinical_study")[0]
elig = study.getElementsByTagName("eligibility")[0]
criteria = elig.getElementsByTagName("textblock")[0].firstChild.data
exclusion = criteria.encode("utf-8")
#print type(exclusion)
exclusion_criteria=exclusion[727:]
#exclusion_criteria


# In[4]:


#sentence detection
def sent_detect(text):
    SENTENCE_TOKENS_PATTERN = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|\!)\s' 
    regex_st = nltk.tokenize.RegexpTokenizer( 
    pattern=SENTENCE_TOKENS_PATTERN, 
    gaps=True) 
    sentences = regex_st.tokenize(exclusion_criteria) 
    return(sentences)
#print sent_detect(exclusion_criteria)


# In[5]:


#word tokenization
def token(sentence):
    default_wt = nltk.word_tokenize 
    words = default_wt(sentence) 
    return words  
#print token(exclusion_criteria)


# In[877]:


# filter all numbers from the corpus- numbers has no significance 
no_digit = filter(lambda i: not str.isdigit(i), token(exclusion_criteria))
string_ex_criteria = str(no_digit)



# In[878]:


# pattern to identify tokens themselves 
def pattern(text):
    TOKEN_PATTERN = r'\w+'         
    regex_wt = nltk.RegexpTokenizer(pattern=TOKEN_PATTERN, 
                                    gaps=False) 
    words2 = regex_wt.tokenize(text) 
    return words2 
#print pattern(string_ex_criteria)


# In[34]:





# In[880]:


#Remove special character

def remove_characters_after_tokenization(tokens): 
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation))) 
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens]) 
    k= ''.join(filtered_tokens) 
    return k
sp_char_free = remove_characters_after_tokenization(string_ex_criteria)
#print sp_char_free


# In[881]:


#remove stopwords

def remove_stopword(text):
    stopword_list = nltk.corpus.stopwords.words('english')
    #print stopword_list
    clean_filter_words = [word for word in token_char_free if word not in stopword_list] #using words from word tokenization
    return clean_filter_words
removed_stopword = remove_stopword(sp_char_free)
#print type(removed_stopword)

#convert list of stop words to string
def stop_word_string(something):
    string_of_stopword = ''
    for elem in removed_stopword:
        string_of_stopword += elem + '\t'
    return string_of_stopword

clean_sw = stop_word_string(removed_stopword) 
#print clean_sw


# In[882]:


#Lemmatize
def lemmatize(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    nltk_tokens = nltk.word_tokenize(text)
    b = ''
    for words in nltk_tokens:
        b += wordnet_lemmatizer.lemmatize(words) + '\t'
        #b += "Actual: %s  Lemma: %s"  % (w,wordnet_lemmatizer.lemmatize(w)) + "\n"
    return b
lem_word = lemmatize(clean_sw)
#print lem_word
    


# In[883]:


#stemming

 
def stemming(string_text):
    porter_stemmer = PorterStemmer()
    # First Word tokenization
    nltk_tokens = token(string_text)
    #Next find the roots of the word
    a ='' 
    for w in nltk_tokens:
        a += porter_stemmer.stem(w) + '\t' 
    return a
    
stem_word= stemming(lem_word)
#print stem_word


# In[885]:


#POS tagger

def pos_tagger(list_of_text):
    #stopword_string = ' '.join(remove_stopword(text))
    tokens = nltk.word_tokenize(list_of_text) 
    tagged_words = nltk.pos_tag(tokens, tagset='universal') 
    return tagged_words
pos_exclusion = pos_tagger(lem_word)
#print pos_exclusion

#identify all nouns
nouns = ([word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(lem_word)) if pos[0] == 'N'])
#print nouns


# In[992]:


#remove duplicate nouns

def remove_duplicates(list):
    return (set(list))
non_duplicate_nouns = remove_duplicates(nouns)
#non_duplicate_nouns
#print non_duplicate_nouns


# In[890]:



#Named Entity Recognition
def name_entity_recognition(words):
    NER = ne_chunk(pos_tag((words)))
    return NER
named_entity=name_entity_recognition(removed_stopword) # use list containg no stopwords
#print named_entity


# In[891]:


#Noun phrase extraction using textblob for semantic analyses
def noun_phrase(text):
    np_text = TextBlob(text)
    noun_phrases = np_text.noun_phrases
    return noun_phrases

exclusion_nounphrase =noun_phrase(clean_sw)
#print exclusion_nounphrase


# In[938]:


#LDA(Latent Dirichlet Allocation)using bag of words for topic modelling

import gensim
from gensim import corpora
def lda_bow_matrix(matrix):
    word_matrix = ["".join(criteria).split() for criteria in matrix]#non_duplicate_nouns]
    #print word_matrix
    dictionary = corpora.Dictionary(word_matrix)
    bow_corpus = [dictionary.doc2bow(criteria) for criteria in word_matrix]

    #Running LDA model
    Lda = gensim.models.ldamodel.LdaModel

    # Running and Trainign LDA model on the document term matrix.
    lda_model = Lda(bow_corpus, num_topics=3, id2word = dictionary, passes=10)
    #print lda_model

    lda_result = lda_model.print_topics(num_topics=3, num_words=20)
    return lda_result
                
final_lda =  lda_bow_matrix(non_duplicate_nouns)
#print final_lda

for idx, topic in lda_model.print_topics(-1):
    model = ('Topic: {} \nWords: {}'.format(idx, topic))
#     print model


# In[937]:


#LDA(Latent Dirichlet Allocation)using TFIDF for topic modelling
def lda_tfidf_matrix(matrix):
    word_matrix = ["".join(criteria).split() for criteria in matrix]#non_duplicate_nouns]
    #print word_matrix
    dictionary = corpora.Dictionary(word_matrix)
    bow_corpus = [dictionary.doc2bow(criteria) for criteria in word_matrix]
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    

    #Running LDA model
    Lda1 = gensim.models.ldamodel.LdaModel

    # Running and Trainign LDA model on the document term matrix.
    lda_model1 = Lda1(corpus_tfidf, num_topics=3, id2word = dictionary, passes=10)
    #print lda_model

    lda_result = lda_model1.print_topics(num_topics=3, num_words=20)
    return lda_result
                
final_lda1 =  lda_tfidf_matrix(non_duplicate_nouns)
#print final_lda

for idx, topic in lda_model.print_topics(-1):
    model = ('Topic: {} \nWords: {}'.format(idx, topic))
#     print model


# In[898]:


#creating a CSV file with the list values from LDA using BOW


# creating corpus of dictionary
word_matrix = ["".join(criteria).split() for criteria in non_duplicate_nouns]#non_duplicate_nouns]
#print word_matrix
dictionary = corpora.Dictionary(word_matrix)
doc_term_matrix = [dictionary.doc2bow(criteria) for criteria in word_matrix]

#write a csv file in the local directory 

mixture = [dict(lda_model[x]) for x in doc_term_matrix ]
pd.DataFrame(mixture).to_csv("exclusion_criteria.csv")

top_words_per_topic = []
for t in range(lda_model.num_topics):
    top_words_per_topic.extend([(t, ) + x for x in lda_model.show_topic(t, topn = 15)])

pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word', 'weight']).to_csv("exclusion_criteria.csv")



# In[1003]:


# Bag of characters vectorization

def vectorize_terms(terms):
    terms = [term.lower() for term in terms]
    terms = [np.array(list(term)) for term in terms]
    terms = [np.array([ord(char) for char in term])
                for term in terms]
    return terms

def boc_term_vectors(word_list):
    word_list = [word.lower() for word in word_list]
    unique_chars = np.unique(np.hstack([(word)for word in word_list]))

    word_list_term_counts = [{char: count for char, count in
                                itemfreq((word))} for word in word_list]
    boc_vectors = [np.array([int(word_term_counts.get(char, 0))
                    for char in unique_chars])
                    for word_term_counts in word_list_term_counts]
    return (unique_chars), boc_vectors
root = 'disease'
term1 = 'migraine'
term2 = 'HIV'
term3 = 'cva'
terms =[root, term1, term2, term3]


features, (boc_root, boc_term1, boc_term2, boc_term3) = boc_term_vectors(terms)

print 'Features:', features
print '''
root: {}
term1: {}
term2: {}
term3: {}
'''.format(boc_root, boc_term1, boc_term2, boc_term3)


# In[1001]:


#cosine Distance and Cosine score to assess the accuracy of  the LDA model 

root_term = root

root_boc_vector = boc_root
terms = [term1, term2, term3]
boc_vector_terms = [boc_term1, boc_term2, boc_term3]
def cosine_distance(u, v):
    distance = 1.0 - (np.dot(u, v) /
    (np.sqrt(sum(np.square(u))) * np.sqrt(sum(np.

    square(v))))

    )
    return distance

for term, boc_term in zip(terms, boc_vector_terms):
    print 'Analyzing similarity between root: {} and term: {}'.format(root_term,term)
distance = round(cosine_distance(root_boc_vector, boc_term),2)
similarity = 1 - distance
print 'Cosine distance is {}'.format(distance)
print 'Cosine similarity is {}'.format(similarity)
print '-'*40

