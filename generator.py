import numpy as np
import random
import sys

#Main path to the data-folder
DATA_PATH = ""
#Path to the text-file, each row has two columns: column1: ID, column2: Textß
TEXT_FILE = ""
#Main path to store all the trained models
BASE_PATH = ""
#Folder name to keep trained models
Folder_name = ""
#Path to the training records where each row corresponds to the label of the generated data-point (see generate_citation_network.py)
training_records = ""
#This is the dictionary of words generated for the experiment. Each line denotes a distinct word in the dictionary.
word_dictionary_file = ""
#sys.path.append("/Users/u6042446/PycharmProjects/sample_test/com/TaxAccounting/TensorFlow/")
#NUM_NEIGHBORS = 5
import data_preparation
from data_preparation import clean_text,reformat_record


word_dictionary = {}
inverse_word_dictionary = {}
with open(DATA_PATH+word_dictionary_file,"r") as f:
    for index,line in enumerate(f):
        word_dictionary[line.strip()] = len(word_dictionary)
        inverse_word_dictionary[len(word_dictionary)-1] = line.strip()


labels_records = []
with open(training_records,"r") as f:
    for line in f:
        data = line.strip().split("\t")
        labels_records.append(data)

training_records = []
test_records = []


#Extract all the raw-data + data-cleaning
training_data = {}
with open(DATA_PATH+TEXT_FILE,"r") as f:
    for index,line in enumerate(f):
        data = line.strip().split("\t")
        paper_id = data[0]
        paper_text = data[-1]
        #clean data here
        #cleaned_paper_text = ..
        #cleaned_paper_text = clean_text(paper_text)
        training_data[paper_id] = paper_text


def generate_word_count(paper_id):
    global word_dictionary
    dict_size = len(word_dictionary)
    word_count = np.zeros([dict_size])
    text = training_data[paper_id]
    for word in text.split(" "):
        if word_dictionary.has_key(word):
            index = word_dictionary[word]
            word_count[index]+=1

    return (word_count)


train_index = 0
test_index = 0

def reset_indexes():
    global training_records
    global test_records

    total_records_length = len(labels_records)
    train_indexes = range(0,len(labels_records))

    train_index_last = int(total_records_length*.9)
    #shuffle training indexes
    random.shuffle(train_indexes)
    this_batch_train_indexes = list(np.array(train_indexes)[0:train_index_last])
    this_batch_test_indexes = list(np.array(train_indexes)[train_index_last:])

    training_records = list(np.array(labels_records)[this_batch_train_indexes])
    test_records = list(np.array(labels_records)[this_batch_test_indexes])

def reset_train_indexes():
    global train_index
    train_index = 0

def reset_test_indexes():
    global test_index
    test_index = 0

def get_next_train_batch(batch_size):
    global train_index
    global training_records
    global word_dictionary
    vocab_dimension = len(word_dictionary)

    #input_batch = np.ndarray(shape=(batch_size, vocab_dimension), dtype=np.float32)
    #labels = np.ndarray(shape=(batch_size , NUM_NEIGHBORS,vocab_dimension), dtype=np.int16)

    min_index = train_index
    max_index = min(train_index + batch_size - 1, len(training_records) - 1)
    input_batch = np.zeros([max_index - min_index + 1, vocab_dimension])
    labels = np.zeros([max_index - min_index + 1, vocab_dimension])
    target_nodes = []
    for index in range(min_index, max_index + 1):
        # index: batch_index
        record = training_records[index]
        target_node = record[0]
        target_nodes.append(target_node)

        neighbors = record[1].split("|")
        #Now, generate word-count vector for any of these articles
        target_word_count = generate_word_count(target_node)

        input_batch[index - min_index, :] = target_word_count
        temp = []
        base_word_count = np.zeros([vocab_dimension])
        for neighbor_index,neighbor in enumerate(neighbors):
            neighbor_word_count = generate_word_count(neighbor)
            base_word_count+= neighbor_word_count

        #labels[index - min_index,:] = np.sign(base_word_count)
        labels[index - min_index, :] = (base_word_count)
        #temp.append(neighbor)
        #neighbor_nodes.append(temp)
    train_index = max_index + 1
    #print(train_index)
    return (input_batch, labels)

def get_next_test_batch(batch_size):
    global test_index
    global test_records
    global word_dictionary
    vocab_dimension = len(word_dictionary)

    #input_batch = np.ndarray(shape=(batch_size, vocab_dimension), dtype=np.float32)
    #labels = np.ndarray(shape=(batch_size , NUM_NEIGHBORS,vocab_dimension), dtype=np.int16)

    min_index = test_index
    max_index = min(test_index + batch_size - 1, len(test_records) - 1)
    target_nodes = []

    input_batch = np.zeros([max_index-min_index+1, vocab_dimension])
    labels = np.zeros([max_index-min_index+1, vocab_dimension])
    for index in range(min_index, max_index + 1):
        # index: batch_index
        record = test_records[index]
        target_node = record[0]
        target_nodes.append(target_node)

        neighbors = record[1].split("|")
        #Now, generate word-count vector for any of these articles
        target_word_count = generate_word_count(target_node)

        input_batch[index - min_index, :] = target_word_count
        temp = []
        base_word_count = np.zeros([vocab_dimension])
        for neighbor_index,neighbor in enumerate(neighbors):
            neighbor_word_count = generate_word_count(neighbor)
            base_word_count+= neighbor_word_count

        #labels[index - min_index,:] = np.sign(base_word_count)
        labels[index - min_index, :] = base_word_count
        #temp.append(neighbor)
        #neighbor_nodes.append(temp)
    test_index = max_index + 1
    #print(test_index)
    return (input_batch, labels)

if __name__=="__main__":
    reset_indexes()
    print("DONE!")