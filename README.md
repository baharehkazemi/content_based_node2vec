# content_based_node2vec
This is the repository for content_based_node2vec project. The project aims to extend the traditional Node2vec technique to consider textual data in the nodes.

---- node2vec_trainer.py: this is the main file to train the content-based node2vec. The file admits two inputs. 1) a file that has the raw-text of each node. 2) an edge-based citation file with each row denoting a citation between any two nodes in the graph. The model is trained and the embedding for each node is generated at the end.
