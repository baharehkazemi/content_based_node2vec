
import unidecode
import string
import numpy as np
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
import operator
import random


#Size of dictionary of words
dictionary_size = 20000
#Path to a text file that stores TF-IDF vectors
TF_IDF_RESULTS = ""
#Path to the raw text-file
TEXT_FILE = ""
#Mapping between each paper and words
PAPER_TO_WORD_MAP = ""
#Path to the dictionary file (each row has a distinct word of the dictionary)
DICTIONARY_FILE = ""
lmtzr = WordNetLemmatizer()
stemmer = PorterStemmer()
punc= string.punctuation
punc_set= set()
for x in punc:
    punc_set.add(x)

def has_punctuation(word):
    for x in word:
        if x in punc_set: return True
    return False

def has_digit(word):
    if re.search("\d",word) is None:
        return (False)
    else:
        return (True)

def clean_title_word(word):
    """
    :TASK= cleaning each word of the title to be represented with just letters
    :param word:
    :return:
    """
    if len(word)<3 or has_punctuation(word) or has_digit(word):
        return ("") #If the title-word is short (only one character), return ""
    else:
        try:
            word = unidecode.unidecode(unicode(word,"utf-8")) #First, convert string to UTF-8 and then decode it
            word= lmtzr.lemmatize((word.lower()))
            word = stemmer.stem(word)
        except:
            word = ""
        return word

def filter_tf_idf(dict,idf):

    new_dict = {}
    new_idf = {}
    for key in idf:
        if idf[key]>4 and idf[key]<8:
            new_idf[key]= idf[key]
    index = 0
    for key,value in new_idf.iteritems():
        new_dict[key] = index
        index+=1

    return(new_dict,new_idf)

def tf_idf(new_dict, paperid, word_frequency,text,new_idf,dirty_to_clean_map):

    # now we need to declare a vector of dictionary size and intialize to zero
    paper_wordcount= {}    # counts the frequency of a word in the document
    split= text[paperid].strip().split(" ")   #the reference document 's text is splitted
    cleansplit=[]
    for x in split:
        #x=clean_title_word(x)
        x = dirty_to_clean_map[x]
        if x!="":
            cleansplit.append(x)   # then the text is clean to be consistent

    feature_vec = np.zeros(len(new_idf))
    #print(cleansplit)
    for key,value in new_idf.iteritems():                              #for each word of the dictionary
        paper_wordcount[key]=0                   # set the frequency to 0
    for word in cleansplit:#this loop search the list
        if not paper_wordcount.has_key(word): continue
        paper_wordcount[word]+=1
    for key,value in paper_wordcount.items():
        index= new_dict[key]

        # now we want to update the feature vector with the true value of tf-idf for the paper and construct the vector
        feature_vec[index]= value*new_idf[key]
        #paper_wordcount[key]= value*idf[key]
    feature_vec= ["%.2f" % w for w in feature_vec]

    return feature_vec


if __name__=="__main__":


    dirty_to_clean_map = {}
    word_mentions = {}

    word_frequency= {}    # word frequency is a hash map where key is the word and value is frequency in all documents
    text={}               # a hash map for holding the text of each paper; key is papaer id anad value is abstract


    clean_paper_writer = open(PAPER_TO_WORD_MAP,"w")
    paper_id_to_real_id_map = {}
    with open(TEXT_FILE, 'r') as f:
        for doc_id,line in enumerate(f):    # doc_id is a counter of line which is actually counts id of paper
            if (doc_id%10000==0): print(doc_id)
            parts = line.strip().split("\t") #for each doc_id ,put its text in abstract
            paper_id_to_real_id_map[doc_id] = parts[0]
            abstract = " ".join(parts[1:])
            text[doc_id]= abstract
            splitted = abstract.split(" ")  # splitts components of t based on space

            list_of_clean_words = []
            for word in splitted:
                cleaned_word = clean_title_word(word)
                dirty_to_clean_map[word] = cleaned_word
                if cleaned_word=="":
                    pass
                else:

                    list_of_clean_words.append(cleaned_word)
                    if cleaned_word in word_frequency:
                        word_frequency[cleaned_word].add(doc_id)  # in the hash map idf, key is the word x and value is all the sentences include x
                        word_mentions[cleaned_word]+=1
                    else:
                        word_frequency[cleaned_word] = set([doc_id])  # the key x had not have any value yet,so the id is set
                        word_mentions[cleaned_word] = 1

            clean_paper_writer.write(parts[0]+"\t"+",".join(list_of_clean_words)+"\n")
    clean_paper_writer.close()

    dict= {}
    counter= 0
    for key,value in word_frequency.iteritems():
        dict[key]= counter
        counter+=1

    #print(idtokeyword[samp])
    print("word-frequency was extracted with size="+str(len(word_frequency)))
    print("text-size is:"+str(len(text)))

    idf = {}
    for key, value in word_frequency.items():
        idf[key] = np.log(110000.0 / len(value))
        #print (key , idf[key])

    #sorted_idf = sorted(idf.items(),key=operator.itemgetter(1),reverse=True)
    #idf = dict(sorted_idf[0:dictionary_size])

    #print(text[1000])
    writer_tf_idf = open(TF_IDF_RESULTS,"w")
    writer_dictionary = open(DICTIONARY_FILE,"w")

    #Filter IDF
    new_dict,new_idf = filter_tf_idf(dict,idf)
    print(len(new_idf))
    sorted_dict = sorted(new_dict.items(),key=operator.itemgetter(1))
    for s in sorted_dict: writer_dictionary.write(s[0]+"\t"+str(s[1])+"\n")

    for paperid,real_paper_id in paper_id_to_real_id_map.iteritems() :
        #print(paperid,tf_idf(new_dict, paperid, word_frequency, text, new_idf,dirty_to_clean_map))
        writer_tf_idf.write(str(real_paper_id)+"\t"+"|".join(tf_idf(new_dict, paperid, word_frequency, text, new_idf,dirty_to_clean_map)))
        writer_tf_idf.write("\n")
    #tf = np.zeros([100000, 63598])
writer_tf_idf.close()
writer_dictionary.close()