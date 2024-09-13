# ontoembed/cli.py

import argparse
from ontoembed.ontologies import OntologyParser
from ontoembed.embeddings import OntologyEmbedding

def main():
    parser = argparse.ArgumentParser(description='OntoEmbed: Ontology Embedding Tool')
    parser.add_argument('-i', '--input', required=True, help='Path to the ontology file (OWL/RDF format)')
    parser.add_argument('-o', '--output', required=True, help='Path to the output vector store file (e.g., .npz file)')
    parser.add_argument('--dimensions', type=int, default=64, help='Dimensions of the embeddings (default: 64)')
    args = parser.parse_args()
    
    # Parse the ontology
    print(f"Parsing ontology from {args.input}...")
    ontology_parser = OntologyParser(args.input)
    entities = ontology_parser.get_entities()
    print(f"Found {len(entities)} entities.")
    
    # Generate embeddings
    print("Generating embeddings...")
    embedding_model = OntologyEmbedding(ontology_parser.graph)
    embedding_model.vector_size = args.dimensions
    vector_store = embedding_model.generate_embeddings(dimensions=args.dimensions)
    
    # Save vector store
    print(f"Saving vector store to {args.output}...")
    embedding_model.save_vector_store(args.output)
    print("Embedding generation and saving completed.")

if __name__ == '__main__':
    main()
