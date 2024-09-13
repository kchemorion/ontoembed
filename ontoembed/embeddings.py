# ontoembed/embeddings.py

import networkx as nx
from gensim.models import Word2Vec
import numpy as np
import random

class OntologyEmbedding:
    def __init__(self, ontology_graph):
        self.graph = ontology_graph
        self.nx_graph = nx.Graph()
        self.vector_size = 64  # Embedding dimensions
    
    def build_networkx_graph(self):
        for s, p, o in self.graph:
            self.nx_graph.add_edge(str(s), str(o))
    
    def generate_walks(self, num_walks=10, walk_length=5):
        walks = []
        nodes = list(self.nx_graph.nodes())
        for _ in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walk = [node]
                current_node = node
                for _ in range(walk_length - 1):
                    neighbors = list(self.nx_graph.neighbors(current_node))
                    if neighbors:
                        current_node = random.choice(neighbors)
                        walk.append(current_node)
                    else:
                        break
                walks.append(walk)
        return walks
    
    def generate_embeddings(self, dimensions=64, window_size=5, workers=4):
        self.build_networkx_graph()
        walks = self.generate_walks()
        # Train Word2Vec model on walks
        model = Word2Vec(
            sentences=walks,
            vector_size=dimensions,
            window=window_size,
            min_count=0,
            sg=1,
            workers=workers
        )
        self.model = model
        # Create vector store (dictionary)
        self.vector_store = {node: model.wv[node] for node in model.wv.index_to_key}
        return self.vector_store
    
    def save_vector_store(self, output_file):
        # Save the vector store to a file (e.g., NumPy format)
        entity_list = list(self.vector_store.keys())
        vectors = np.array([self.vector_store[entity] for entity in entity_list])
        # Save entity list and vectors
        np.savez(output_file, entities=entity_list, vectors=vectors)
