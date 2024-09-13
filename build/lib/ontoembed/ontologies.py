# ontoembed/ontologies.py

from rdflib import Graph, RDF, OWL

class OntologyParser:
    def __init__(self, ontology_file):
        self.graph = Graph()
        self.graph.parse(ontology_file)
    
    def get_entities(self):
        entities = set()
        # Get all classes
        for s in self.graph.subjects(RDF.type, OWL.Class):
            entities.add(s)
        # Get all properties
        for s in self.graph.subjects(RDF.type, OWL.ObjectProperty):
            entities.add(s)
        for s in self.graph.subjects(RDF.type, OWL.DatatypeProperty):
            entities.add(s)
        # Get all individuals
        for s in self.graph.subjects(RDF.type, None):
            if (s, RDF.type, OWL.Class) not in self.graph and \
               (s, RDF.type, OWL.ObjectProperty) not in self.graph and \
               (s, RDF.type, OWL.DatatypeProperty) not in self.graph:
                entities.add(s)
        return entities
