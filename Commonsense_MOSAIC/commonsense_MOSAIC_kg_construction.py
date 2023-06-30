import json
import urllib
import urllib.parse
import textwrap

import csv
import glob
import os
from collections import defaultdict
import pandas as pd

def generate_clusters_triples():
    # Load data from JSON file
    with open('cluster_data.json', 'r') as file:
        data = json.load(file)
        data = data["data"]
    cluster_triples = []
    rdf_prefix = "musco"

    for cluster in data:
        cluster_name = cluster['cluster_name']
        cluster_id = cluster['cluster_id']
        cluster_declaration = f"{rdf_prefix}:{cluster_name}_cluster rdf:type {rdf_prefix}:AbstractConceptCluster ."
        cluster_triples.append(cluster_declaration)

        symbols = cluster['symbols']
        for symbol in symbols:
            symbol_declaration = f"conceptnet:{symbol} rdf:type {rdf_prefix}:Concept ."
            cluster_triples.append(symbol_declaration)

            cluster_symbol_declaration = f"{rdf_prefix}:{cluster_name}_cluster {rdf_prefix}:hasClusterConcept conceptnet:{symbol} ."
            cluster_triples.append(cluster_symbol_declaration)
    for declaration in cluster_triples:
        print(declaration)
    return cluster_triples

# Commonsense KG: with info from Framester and Quokka
def generate_quokka_triples():
    with open('quokka_data.json', 'r') as file:
        json_data = json.load(file)

    quokka_triples = []
    rdf_prefix = "musco"

    for cluster_name, cluster in json_data.items():
        for subdict_key, subdict in cluster.items():
            for entry in subdict:
                causes = entry.get("#\u00a0causes", [])
                used_for = entry.get("#\u00a0usedFor", [])

                for cause in causes:
                    parsed_cause = urllib.parse.urlparse(cause)
                    concept_name = parsed_cause.path.split('/')[-1]
                    triple = f"conceptnet:{subdict_key} {rdf_prefix}:hasCause conceptnet:{concept_name} ."
                    quokka_triples.append(triple)
                    triple = f"{rdf_prefix}:{cluster_name} {rdf_prefix}:RelatedConcept conceptnet:{concept_name} ."
                    quokka_triples.append(triple)

                for use in used_for:
                    parsed_use = urllib.parse.urlparse(use)
                    concept_name = parsed_use.path.split('/')[-1]
                    triple = f"conceptnet:{subdict_key} {rdf_prefix}:isProvokedBy conceptnet:{concept_name} ."
                    quokka_triples.append(triple)
                    triple = f"{rdf_prefix}:{cluster_name} {rdf_prefix}:RelatedConcept conceptnet:{concept_name} ."
                    quokka_triples.append(triple)

    # Print the RDF triples
    # for triple in quokka_triples:
    #     print(triple)

    return quokka_triples



def create_commonsense_kg():
    # Prefixes and base declaration
    prefixes = textwrap.dedent("""\
    @prefix : <https://w3id.org/musco#> .
    @prefix dc: <http://purl.org/dc/elements/1.1/> .
    @prefix DUL: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix xml: <http://www.w3.org/XML/1998/namespace> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix skos: <http://www.w3.org/2004/02/skos/core#> .
    @prefix musco: <https://w3id.org/musco#> .
    @prefix terms: <http://purl.org/dc/terms/> .
    @prefix fschema: <https://w3id.org/framester/schema/> .
    @prefix conceptnet: <https://w3id.org/framester/conceptnet/5.7.0/c/en/> .

    @base <https://w3id.org/musco#> .

    """)
    cluster_triples = generate_clusters_triples()
    quokka_triples = generate_quokka_triples()
    combined_set = set(cluster_triples)
    combined_set.update(quokka_triples)

    # Output the turtle (.ttl) file
    output_filename = "Common-sense_MOSAIC.ttl"

    with open(output_filename, "w") as file:
        file.write(prefixes)
        for triple in combined_set:
            file.write(triple + "\n")

    print(f"CS MOSAIC KG created successfully for concept clusters and saved as '{output_filename}'.")

    return

# Execution
create_commonsense_kg()