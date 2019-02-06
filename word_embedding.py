import collections
import operator
import sys
from sklearn.metrics.pairwise import pairwise_distances
#sys.path.append("/Users/u6042446/PycharmProjects/sample_test/com/TaxAccounting/TensorFlow/")

#Base path for storing data and reading required dataset
BASE_PATH = ""
#Original text-file
TEXT_FILE = ""
#Procssed text (after lemmetization and stemming)f3qwwqΩß
PROCESSED_TEXT_FILE = ""
#A maap of clean-to-raw words
CLEAN_TO_DIRTY_WORD = ""
#Frequency of words
WORD_FREQUENCY = ""
#List of words
WORD_FILE = ""
tf_idf_dictionary = ""
#Folder that contains all the embbedding-trained models
Embedding_model_path = ""

#A file that contains list of stop-words
STOP_WORDS = ""

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import WordToVec
reload(WordToVec)
from WordToVec import  generate_batch_cbow
from WordToVec import  generate_batch_of_data_cbow
from WordToVec import Word2Vec
from WordToVec import build_dataset
import numpy as np
import data_preparation
from data_preparation import clean_text,reformat_record
wordnet_lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def extract_raw_text_bag_of_words():
    """
    :return:
    """
    cs_data_map = {}
    word_frequency = {}
    clean_to_dirty_map = {}

    file_to_write = open(BASE_PATH+PROCESSED_TEXT_FILE,"w")
    with open(BASE_PATH+TEXT_FILE,"r") as file:
        for line in file:
            data = line.strip().split("\t")
            if len(data)!=3: continue
            paper_id = data[0]
            #clean the text
            abstract_clean_to_dirty_map,cleaned_text = reformat_record(data[1]+" "+data[2])#This is the cleaned-text
            cs_data_map[paper_id] = cleaned_text

            for word in cleaned_text.split(" "):
                if not word_frequency.has_key(word): word_frequency[word] = 0
                word_frequency[word]+=1

            for c_word,d_word in abstract_clean_to_dirty_map.iteritems():
                if not clean_to_dirty_map.has_key(c_word): clean_to_dirty_map[c_word] = set([])
                clean_to_dirty_map[c_word].add(d_word)

            file_to_write.write(str(paper_id)+"\t"+cleaned_text+"\n")
    file_to_write.close()

    word_file = open(BASE_PATH+WORD_FILE,"w")
    word_frequency_file = open(BASE_PATH+WORD_FREQUENCY,"w")
    clean_dirty_file = open(BASE_PATH+CLEAN_TO_DIRTY_WORD,"w")
    for cleaned_word,dirty_words in clean_to_dirty_map.iteritems():
        word_file.write(cleaned_word+"\n")
        clean_dirty_file.write(cleaned_word+"\t"+"|".join(list(dirty_words))+"\n")
        word_frequency_file.write(cleaned_word+"\t"+str(word_frequency[cleaned_word])+"\n")

    word_file.close()
    word_frequency_file.close()
    clean_dirty_file.close()
    return (cs_data_map)


def find_most_common_words(list_of_words,vocab_size,count):
    dict = {}
    for x in list_of_words:
        if x=="": continue
        if not dict.has_key(x): dict[x] = 0
        dict[x]+=1

    sorted_dict = sorted(dict.items(),key=operator.itemgetter(1),reverse=True)
    for x in sorted_dict[0:vocab_size]:
        count.append([x[0],x[1]])
    return (count)



def process_token(word):
    """
    :Do all pre-processing here
    :param word:
    :return:
    """
    lemmatized_word = wordnet_lemmatizer.lemmatize(word.lower())
    stem = stemmer.stem(lemmatized_word)
    return (stem)

def generate_dictionary_file():
    """

    :return:
    """
    word_dict = {}
    inverse_word_dict = {}
    word_index = 0

    stop_words = set([])
    with open(STOP_WORDS,"r") as file:
        for line in file:
            stop_words.add(line.strip().lower())

    with open(BASE_PATH+WORD_FREQUENCY,"r") as file:
        for line in file:
            data = line.strip().split("\t")
            unique_word = data[0]
            if unique_word.lower() in stop_words: continue #skip stop-words
            if unique_word.isdigit(): continue #skip digits
            if not unique_word.isalpha() and not unique_word.isdigit(): continue #skip alphanumerics
            if len(unique_word)<3: continue
            #unique_word = process_token(unique_word)
            if word_dict.has_key(unique_word): continue
            word_dict[unique_word] = word_index
            inverse_word_dict[word_index] = unique_word
            word_index+=1

    return (word_dict,inverse_word_dict)


def generate_corpus_data():
    """
    :return: hashmap of corupus data with KEY=documentId, VALUE=list of all tokens in the text
    """

    file1 = open(BASE_PATH+"tagged_text.txt", "w")
    word_dict,inv_word_dict = generate_dictionary_file()
    doc_text_map = {}
    with open(BASE_PATH+PROCESSED_TEXT_FILE, "r") as file:
        for line in file:
            data = line.strip().split("\t")
            doc_id = data[0]
            if not doc_text_map.has_key(doc_id): doc_text_map[doc_id] = []
            text = data[1]
            for token in text.split(" "):
                cleaned_token = token
                if cleaned_token in word_dict:
                    doc_text_map[doc_id].append(cleaned_token)


    for document_id,text_list in doc_text_map.iteritems():
        file1.write(str(document_id)+"\t")
        file1.write("\t".join(text_list)+"\n")

    file1.close()
    return (doc_text_map)

def extract_word_embeddings_model():
    """
    :TASK= extracts a pre-trained word2vec model on FTC-corpus
    :return:
    """
    extracted_model = Word2Vec.restore(Embedding_model_path)  # This is the extracted model

    return (extracted_model)


def sort(model, word, top_n):
    """
    :param self:
    :param word:
    :param top_n:
    :return:
    """
    assert word in model.dictionary
    i = model.dictionary[word]
    vec = model.final_embeddings[i].reshape(1, -1)
    # Calculate pairwise cosine distance and flatten to 1-d
    pdist = pairwise_distances(model.final_embeddings, vec, metric='cosine').ravel()
    top_results = []
    for index, i in enumerate(pdist.argsort()):
        if index < top_n and index > 0:
            top_results.append(model.reverse_dictionary[i])
    return top_results

def generate_embedding_storage(model):
    """
    :TASK extracts all embeddings from the trained model
    :param model: pre-trained model (CBOW)
    :return:
    """
    file_to_write = open(BASE_PATH+"/cs_embeddings.txt","w")
    for word,index in model.dictionary.iteritems():
        embedding = model.final_embeddings[index].reshape(1,-1)
        embedding_list = []
        [embedding_list.append(str(x)) for x in embedding[0]]
        file_to_write.write(word+"\t"+"|".join(embedding_list)+"\n")

    file_to_write.close()


if __name__=="__main__":

    tf_idf_dic = []
    with open(BASE_PATH+tf_idf_dictionary,"r") as file:
        for line in file:
            data = line.strip()
            tf_idf_dic.append(data)

    list_of_words = []
    with open(BASE_PATH+PROCESSED_TEXT_FILE,"r") as file:
        for counter,line in enumerate(file):
            splitted_data = line.strip().split("\t")
            text = splitted_data[1]
            for token in text.split(" "):
                if token=="": continue
                cleaned_word = token
                list_of_words.append(str(cleaned_word))
            list_of_words.append(" ")


    batch_size = 64
    vocab_size = len(tf_idf_dic)
    data, count, dictionary, reverse_dictionary, sentence_boundary_index = build_dataset(list_of_words,vocab_size,tf_idf_dic)
    n_steps_mine = int(len(data)/(batch_size*1.0))
    print(n_steps_mine)
    w2v = Word2Vec(vocabulary_size=vocab_size,architecture='cbow',n_steps=n_steps_mine,batch_size=batch_size,num_skips=4,embedding_size=100)
    w2v.fit(list_of_words,vocab_size,tf_idf_dic)
    saved_path = w2v.save(Embedding_model_path)



