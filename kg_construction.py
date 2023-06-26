import json
import urllib
import urllib.parse


def generate_clusters_triples():
    # Load data from JSON file
    with open('cluster_data.json', 'r') as file:
        data = json.load(file)
        data = data["data"]
    rdf_declarations = []
    namespace_prefixes = {
        "musco": "http://example.com/ontloy/musco#",
        "conceptnet": "http://example.com/ontloy/conceptnet#"
    }

    for cluster in data:
        cluster_name = cluster['cluster_name']
        cluster_id = cluster['cluster_id']
        cluster_declaration = f"musco:{cluster_name}_cluster rdf:type musco:AbstractConceptCluster ."
        rdf_declarations.append(cluster_declaration)

        symbols = cluster['symbols']
        for symbol in symbols:
            symbol_declaration = f"conceptnet:{symbol} rdf:type musco:Concept ."
            rdf_declarations.append(symbol_declaration)

            cluster_symbol_declaration = f"musco:{cluster_name}_cluster musco:hasClusterConcept conceptnet:{symbol} ."
            rdf_declarations.append(cluster_symbol_declaration)
    for declaration in rdf_declarations:
        print(declaration)
    return rdf_declarations

# Commonsense KG: with info from Framester and Quokka
def generate_quokka_triples():
    with open('quokka_data.json', 'r') as file:
        json_data = json.load(file)

    triples = []

    for cluster in json_data.values():
        for subdict_key, subdict in cluster.items():
            for entry in subdict:
                causes = entry.get("#\u00a0causes", [])
                used_for = entry.get("#\u00a0usedFor", [])

                for cause in causes:
                    parsed_cause = urllib.parse.urlparse(cause)
                    concept_name = parsed_cause.path.split('/')[-1]
                    triple = f"conceptnet:{subdict_key} quokka:hasCause conceptnet:{concept_name} ."
                    triples.append(triple)

                for use in used_for:
                    parsed_use = urllib.parse.urlparse(use)
                    concept_name = parsed_use.path.split('/')[-1]
                    triple = f"conceptnet:{subdict_key} quokka:isProvokedBy conceptnet:{concept_name} ."
                    triples.append(triple)

    # Print the RDF triples
    for triple in triples:
        print(triple)

    return triples

# Generate RDF declarations for AbstractConceptClusters and ClusterConcepts
# generate_clusters_triples()
# generate_quokka_triples()


import csv

def filter_collocations(csv_file):
    filtered_collocations = []

    with open(csv_file, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)  # Skip the header row

        for row in reader:
            collocation = row[0]
            score = float(row[2])

            if score > 5.0:
                filtered_collocations.append(collocation)

    return filtered_collocations

csv_file = 'comfort_ententen21.csv'
filtered_collocations = filter_collocations(csv_file)

for collocation in filtered_collocations:
    print(collocation)