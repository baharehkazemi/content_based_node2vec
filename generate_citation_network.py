
#Path to the data-folder
DATA_PATH = ""
#Path to the text file: column1: text-id, column2: text
ABSTRACT_FILE = ""
#Edge-file: each row has an edge (two column any of which has the ids)
EDGE_FILE = ""
#Path to the instances for link-prediction
LINK_PREIDCTION_INSTANCES = ""


import networkx as nx
import numpy as np
import random
import sys
import os
import json


class CitationNetwork:
    def __init__(self):
        self.p = 5.0
        self.q = 5.0
        self.random_walk_length = 5
        self.max_samples_per_node = 10
        self.context_size = 3
        self.G = self.__create_graph()#create a directed graph
        self.relation_G = self.__create_relation_graph()#create a relation graph (regardless of the directions)
        self.G = self.remove_cycles(self.G)
        #self.relation_G = self.remove_cycles(self.relation_G)
        self.transition_matrix = {}
        self.all_nodes = []
        self.__get_all_the_nodes()


    def __get_all_the_nodes(self):

        with open(EDGE_FILE,"r") as f:
            tmp = set([])
            for line in f:
                data = line.strip().split("\t")
                tmp.add(data[0])
                tmp.add(data[1])
        self.all_nodes = list(tmp)


    def __create_graph(self):
        """
        :TASK forms a directed graph from citations
        :return:
        """
        G = nx.DiGraph()
        with open(EDGE_FILE,"r") as citation_file:
            for line in citation_file:
                data = line.strip().split("\t")
                _from_id = data[0]
                _to_id = data[1]
                G.add_edge(_to_id,_from_id)

        return (G)

    def __create_relation_graph(self):
        """
        :TASK forms a directed graph from citations
        :return:
        """
        G = nx.Graph()
        with open(EDGE_FILE,"r") as citation_file:
            for line in citation_file:
                data = line.strip().split("\t")
                _from_id = data[0]
                _to_id = data[1]
                G.add_edge(_to_id,_from_id)

        return (G)

    def generate_training_nodes(self):
        """
        :return:
        """

        self.training_instances = []
        for node,generated_paths in self.sampled_nodes.iteritems():
            if not generated_paths: continue
            for path in generated_paths:
                lag = 0
                while lag+self.context_size-1<=len(path):
                    self.training_instances.append(path[lag:lag+self.context_size])
                    lag+=1



    def generate_random_walks(self):
        """
        :Generate possible random walks
        :return:
        """

        self.sampled_nodes = {}
        for node in self.G.nodes():
            if not self.G.edge[node]: continue#No edge for this node, ignore
            nested_node_list = []
            for r in range(0,self.max_samples_per_node):
                #generate a random-walk of length l
                prev_node = ""
                current_node = node
                nodes_list = [node]
                for l in range(0,self.random_walk_length):
                    if prev_node=="":
                        #choose one of neighbors uniformly (forst-order markov model)
                        neighbors = self.G.edge[current_node].keys()
                        chosen_node_index = np.random.multinomial(1,np.ones(len(neighbors))*(1.0/len(neighbors))).argmax()
                        prev_node = current_node
                        current_node = neighbors[chosen_node_index]#sampled node

                    else:
                        if not self.transition_matrix[current_node]: break
                        if self.transition_matrix[current_node].has_key(prev_node):
                            if not self.transition_matrix[current_node][prev_node]: break
                            probs = self.transition_matrix[current_node][prev_node].values()
                            nodes = self.transition_matrix[current_node][prev_node].keys()
                            chosen_node_index = np.random.multinomial(1, probs).argmax()
                            prev_node = current_node
                            current_node = nodes[chosen_node_index]
                        else:
                            break
                    nodes_list.append(current_node)
                if len(nodes_list)>3: nested_node_list.append(nodes_list)
            self.sampled_nodes[node] = nested_node_list

    def remove_cycles(self,G):

        cycles = nx.simple_cycles(G)
        for cycle in cycles:
            node1_degree = nx.degree(G,cycle[0])
            node2_degree = nx.degree(G,cycle[-1])
            try:
                G.remove_edge(cycle[-1], cycle[0])
            except:
                pass
        return (G)

    def sedond_order_markov_transition(self,G):
        """
        :Task: run second-order Markov transition matrix
        :return:
        """

        transition_matrix = {}
        for node in G.nodes():
            #shape a P by S matrix with P being number of predecessors and V being number of successors
            S = G.successors(node)
            P = G.predecessors(node)
            if len(S)==0:
                transition_matrix[node] = np.array([])
                continue
            if len(P)==0:continue

            transition = np.zeros([len(P),len(S)])
            sum = 0
            for p_index,p in enumerate(P):
                #list of all predecessors
                for s_index,s in enumerate(S):
                    #list of all successors
                    #calculate shortest-path between (p,s)
                    distance = nx.shortest_path_length(G,p,s)
                    if distance==0:
                        transition[p_index,s_index] = 1.0/self.p
                        sum+= 1.0/self.p
                    elif distance==1:
                        transition[p_index,s_index] = 1.0
                        sum+= 1.0
                    elif distance==2:
                        transition[p_index,s_index] = 1.0/self.q
                        sum+= 1.0/self.q
            #normalization
            transition = transition/sum
            transition_matrix[node] = {}
            for p_index, p in enumerate(P):
                for s_index, s in enumerate(S):
                    if not transition_matrix[node].has_key(p): transition_matrix[node][p] = {}
                    transition_matrix[node][p][s] = transition[p_index,s_index]

        return (G,transition_matrix)

    def generate_training_instances(self,relation_G,neg_instances):
        """

        :param relation_G:
        :param neg_instances:
        :return:
        """
        positive_instances = []
        negative_instances = []

        generated_negative_nodes = {}
        for node in relation_G.edges():
            #each edge is a positive instance
            node1 = node[0]
            node2 = node[1]
            key = node1+"|"+node2
            positive_instances.append(key)
            #generate negative instances

            node1_neg_instances = []
            while (len(node1_neg_instances)<neg_instances/2):
                #draw a sample randomly
                random_node = self.all_nodes[random.randint(0,len(self.all_nodes)-1)]
                if relation_G.has_edge(node1,random_node): continue
                if not node1 in generated_negative_nodes:
                    generated_negative_nodes[node1] = set([])
                    generated_negative_nodes[node1].add(random_node)
                    node1_neg_instances.append(random_node)
                else:
                    if not random_node in generated_negative_nodes[node1]:
                        generated_negative_nodes[node1].add(random_node)
                        node1_neg_instances.append(random_node)
                    else:
                        pass

            node2_neg_instances = []
            while (len(node2_neg_instances) < neg_instances / 2):
                # draw a sample randomly
                random_node = self.all_nodes[random.randint(0, len(self.all_nodes)-1)]
                if relation_G.has_edge(node2,random_node): continue
                if not node2 in generated_negative_nodes:
                    generated_negative_nodes[node2] = set([])
                    generated_negative_nodes[node2].add(random_node)
                    node2_neg_instances.append(random_node)
                else:
                    if not random_node in generated_negative_nodes[node2]:
                        generated_negative_nodes[node2].add(random_node)
                        node2_neg_instances.append(random_node)
                    else:
                        pass

            for x in node1_neg_instances: negative_instances.append(node1+"|"+x)
            for x in node2_neg_instances: negative_instances.append(node2+"|"+x)

        return (positive_instances,negative_instances)




if __name__=="__main__":
    #Create citaion-network object
    citation = CitationNetwork()
    #Create positive/negsative instances for link-prodication
    positive_instances, negative_instances = citation.generate_training_instances(citation.relation_G, 10)

    writer = open(LINK_PREIDCTION_INSTANCES, "w")
    all_instances = positive_instances+negative_instances
    instance_indexes = list(range(0,len(all_instances)))
    random.shuffle(instance_indexes)
    for index in instance_indexes:
        if index<len(positive_instances):
            label = 1
        else:
            label = 0

        parts = all_instances[index].split("|")
        writer.write(parts[0]+"\t"+parts[1]+"\t"+str(label)+"\n")
    writer.close()
    #sys.exit(1)

    #for x in positive_instances:
     #   parts= x.split("|")
      #  writer.write(str(parts[0]) + "\t" + str(parts[1]+"\t"+"1"))
       # writer.write("\n")
    #for x in negative_instances:
     #   parts=x.split("|")
      #  writer.write(str(parts[0]) + "\t" + str(parts[1]+"\t"+"0"))
       # writer.write("\n")
    #writer.close()