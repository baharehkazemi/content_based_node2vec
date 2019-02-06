
DATA_PATH = "/Users/u6042446/Downloads/MicrosoftAcademicGraph/CS_DATA_SET/"
#Path to the text-file, each row has two columns: column1: ID, column2: Textß
TEXT_FILE = ""
#Main path to store all the trained models
BASE_PATH = ""
#Folder name to keep trained models
Folder_name = ""


import tensorflow as tf
import generator
import numpy as np
import os

#General parameters
DICT_SIZE = 8025
EMBEDDING_SIZE = 100

#Extract all the raw-data + data-cleaning
training_data = {}
with open(DATA_PATH+TEXT_FILE,"r") as f:
    if __name__ == '__main__':
        for index,line in enumerate(f):
            data = line.strip().split("\t")
            paper_id = data[0]
            paper_text = data[-1]
            #clean data here
            #cleaned_paper_text = ..
            cleaned_paper_text = paper_text
            training_data[paper_id] = paper_text


def multilayer_perceptron(x_input, mlp_weights, mlp_biases):
    """
    :param x_left, x_right: input_sentence embddings for left/right sentnces
    :param weights: input-layer projection (embedding_size \times input_size)
    :param biases: input_bias (embedding_size \times 1)
    :return: Add embeddings of each two sentence embeddings (we can try other operations too)
    """
    # Hidden layer with RELU activation
    layer_1_out = tf.add(tf.matmul(x_input, mlp_weights['h1']), mlp_biases['b1'])
    layer_1_out = tf.nn.sigmoid(layer_1_out)#sentence embeddings (embedding_size by 1)

    return layer_1_out

def softmax_layer(embedding_input, out_weights, out_biases):
    """

    :param layer1: output of the MLP layer (sentence embedding)
    :param weights: softmax-projection weight matrix (vocab_size \times embedding_size)
    :param biases: softmax-projection bias vector (vocab_size \times 1)
    :return: output of softmax (a probability vector)==> batch_size by vocab_size
    """

    #layer1_output = tf.reshape(layer1_output,[-1,batch_size*embedding_size])
    prediction = tf.nn.softmax(tf.matmul(embedding_input, out_weights['weight']) + out_biases['bias'])+1E-10*tf.ones(1)#vocab_size*1
    return (prediction)

def prediction(x_input,mlp_weights,mlp_biases,out_weights,out_biases):

    layer1_out = multilayer_perceptron(x_input,mlp_weights,mlp_biases)
    out_prediction = tf.nn.sigmoid(tf.add(tf.matmul(layer1_out,out_weights["weight"]),out_biases["bias"]))
    #out_prediction = softmax_layer(layer1_out,out_weights,out_biases)

    return (out_prediction)

def output_cost(y_labels):
    """
    :TASK= this will work well with a variable-size output txt
    :param x: input-data; size= num_batches*embedding_dimension
    :param y: output-labels; size = (num_batches*max_length)*vocabulary_size
    ==> for each batch and each token in the outputs sentence, a vector of V by 1 is provided where the i-th non-zero index corresponds to the index of the word in the dictionary
    :param mlp_weights: weight matrix of the MLP network
    :param mlp_biases: bias vector of the MLP network
    :param out_weights: Weights of the softmax layer (vocabulary_size*embedding_dimension matrix)
    :param out_biases: vector of the softmax layer (embedding_dimension*1 vector)
    :return:
    """

    #out_cost = -tf.reduce_mean(tf.add(y_labels*tf.log(pred+1E-30),tf.nn.relu((1-y_labels))*(tf.log(1-pred+1E-30))))
    #out_cost = -tf.reduce_sum(tf.nn.relu((1-y_labels))*tf.log(1-pred))

    out_cost = -tf.reduce_mean(tf.log(tf.add(tf.nn.relu(y_labels)*(pred+1E-30),tf.nn.relu(1-y_labels)*(1-pred+1E-30))))

    #out_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_labels))

    #total_number_of_effective_batch_tokens = tf.reduce_sum(y, reduction_indices=1)#total number of tokens per-batch
    #total_number_of_nonzero_cases = tf.reduce_sum(total_number_of_effective_batch_tokens,reduction_indices=0)#totoal number of tokens for all batches
    #caclulate the cost
    #per_batch_cost = -tf.reduce_sum(y*tf.log(pred),reduction_indices=1)#\sum_{t=1}^T p(v_t=y_t)==> v_t: t-th token, y_t= label for this token
    #all_batch_cost = tf.reduce_sum(per_batch_cost,reduction_indices=0)#summation of costs for all batches
    #all_batch_cost/= total_number_of_nonzero_cases#normalized cost

    return(out_cost)

def save(sess,path, model_name):
    save_path = tf.train.Saver().save(sess,
                                os.path.join(path, model_name+'.ckpt'))
    # save parameters of the model
    print("Model saved in file: %s" % save_path)
    return save_path

def restored_model(x_input,mlp_weights,mlp_biases):

    embedding = multilayer_perceptron(x_input,mlp_weights,mlp_biases)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    file_to_write = open(BASE_PATH+"embeddings.txt","w")
    new_saver = tf.train.import_meta_graph(BASE_PATH + "skipgram_noSoftmax_p3_q3" + ".ckpt.meta")
    new_saver.restore(sess, BASE_PATH + "skipgram_noSoftmax_p3_q3" + ".ckpt")

    embedding_matrix = []
    for key in generator.training_data.keys():
        record = generator.generate_word_count(key)
        #record = record/np.linalg.norm(record)
        z = np.zeros([1, DICT_SIZE])
        z[0, :] = record
        feed_dict = {x_input: z}
        embeddings = sess.run([embedding], feed_dict)
        extracted_embedding = embeddings[0][0]
        extracted_embedding = extracted_embedding/np.linalg.norm(extracted_embedding)
        embedding_matrix.append(extracted_embedding)

        str_embedding = []
        [str_embedding.append(str(x)) for x in extracted_embedding]
        file_to_write.write(str(key)+"\t"+"|".join(str_embedding)+"\n")

    file_to_write.close()

    return (embedding_matrix)

if __name__=="__main__":

    # Hyper-parameters
    learning_rate = 1E-3
    training_epochs = 5
    batch_size = 128
    display_step = 1

    #Neural-net structure
    mlp_weights = {
        'h1': tf.Variable(tf.random_normal([DICT_SIZE,EMBEDDING_SIZE]))
    }
    mlp_biases = {
        'b1': tf.Variable(tf.random_normal([EMBEDDING_SIZE]))
    }

    out_weights = {
        'weight': tf.Variable(tf.random_normal([EMBEDDING_SIZE, DICT_SIZE]))
    }

    out_biases = {
        'bias': tf.Variable(tf.random_normal([DICT_SIZE]))
    }

    total_num_instances = len(generator.labels_records)
    num_batches = total_num_instances / batch_size #Total number of batches

    # tf Graph input
    x_input = tf.placeholder("float", [None, DICT_SIZE])
    y_output = tf.placeholder("float", [None, DICT_SIZE])
    #embedding_matrix = restored_model(x_input,mlp_weights,mlp_biases)
    #sys.exit(1)
    os.mkdir(Folder_name)
    pred = prediction(x_input,mlp_weights,mlp_biases,out_weights,out_biases) #predictions
    cost_mlp = output_cost(y_output)
    optimizer_mlp = tf.train.AdamOptimizer(learning_rate).minimize(cost_mlp)

    #initialize tf_graph
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        #loop over epochs
        for t in range(0,training_epochs):
            #reset everything
            generator.reset_indexes()
            generator.reset_test_indexes()
            generator.reset_train_indexes()
            #Loop over batches
            train_cost = 0
            for n in range(0,num_batches):
                #get training-data
                target_batch,labels_batch = generator.get_next_train_batch(batch_size)
                if np.shape(target_batch)[0]==0: continue
                feed_dict_mlp = {x_input: target_batch,y_output: labels_batch}
                _, c = sess.run([optimizer_mlp, cost_mlp], feed_dict_mlp)

                train_cost+= c
                if n%100==0 and n>0:
                    #Test part
                    #print("train cost after " + str(n) + " rounds is:" + str(train_cost/10.0))
                    generator.reset_test_indexes()
                    num_test_records = len(generator.test_records)
                    test_target, test_labels = generator.get_next_test_batch(num_test_records)
                    feed_dict_mlp = {x_input: test_target, y_output: test_labels}
                    c = sess.run([cost_mlp], feed_dict_mlp)
                    print("test cost after "+str(n)+" rounds is:"+str(c))
                    train_cost = 0

        save(sess,BASE_PATH+Folder_name+"/",Folder_name)











