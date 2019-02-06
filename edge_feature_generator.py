
#import unidecode
import string
import numpy as np
import re
#from nltk.stem.wordnet import WordNetLemmatizer
#from nltk.stem import PorterStemmer
import operator
import random

#Path to the embbedding for each paper: each row has two columns: Column1: paper_id, Column2: embedding vector
EMBED_FILE = ""
EDGE_FEATURE_FILE = ""
LINK_PREIDCTION_INSTANCES = ""

def hadmert_feature_generator(paperid1,paperid2,paper_to_embedding):

    #temp1 holds the embedding vector of paperid1
    temp1=np.array(paper_to_embedding[paperid1])
    temp2=np.array(paper_to_embedding[paperid2])
    resultvector= temp1*temp2
    _to_string_vector = []
    [_to_string_vector.append(str(x)) for x in resultvector]

    return np.array(_to_string_vector)

def average_feature_generator(paperid1,paperid2,paper_to_embedding):

    #temp1 holds the embedding vector of paperid1
    temp1=np.array(paper_to_embedding[paperid1])
    temp2=np.array(paper_to_embedding[paperid2])
    resultvector= (temp1+temp2)/2.0
    _to_string_vector = []
    [_to_string_vector.append(str(x)) for x in resultvector]

    return np.array(_to_string_vector)

if __name__=="__main__":

    #STAGE1: load all the document embeddings
    paper_to_embedding={}
    with open(EMBED_FILE,"r") as f1:
        for index,line in enumerate(f1):
            #transfering data to a hash map
            parts=line.strip().split("\t")
            temp_vec = []
            [temp_vec.append(float(x)) for x in parts[1].split("|")]
            paper_to_embedding[parts[0]]=np.array(temp_vec)
            #parts[0] is paper id and parts[1] is the embedding for the paper

    #STAGE2: genrate edge features using either Average or Hadamard (other operations may bbe used)
    writer= open(EDGE_FEATURE_FILE, "w")
    with open(LINK_PREIDCTION_INSTANCES, "r") as f2:
        for line in f2:
            parts=line.strip().split("\t")
            if parts[0] in paper_to_embedding and parts[1] in paper_to_embedding:
                #parts[0] is paperid1 and parts[1] is paperid2 in training tuple
                writer.write(str(parts[0]) + "\t" + str(parts[1])+ "\t"+"|".join(average_feature_generator(parts[0],parts[1],paper_to_embedding))+"\t"+parts[2]+"\n")


    writer.close()