import json
import csv
import os
from collections import defaultdict
import textwrap


def generate_clusters_triples():
    # Load data from JSON file
    with open('../Commonsense_MOSAIC/cluster_data.json', 'r') as file:
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

# DL KG: with distributional linguistic data from Sketch Engine
def generate_dl_triples(corpus_name, download_date):
    # RDF prefixes
    rdf_prefix = "musco"
    conceptnet_prefix = "conceptnet"
    xsd_prefix = "xsd"

    # RDF triples
    rdf_triples = []
    rdf_triples.append(
        f"{rdf_prefix}:{corpus_name}_{download_date}_sit rdf:type {rdf_prefix}:CollocationAnnotationSituation .")
    rdf_triples.append(
        f"{rdf_prefix}:{corpus_name}_{download_date}_desc rdf:type {rdf_prefix}:CollocationAnnotationDescription .")
    rdf_triples.append(
        f"{rdf_prefix}:{corpus_name}_{download_date}_sit  {rdf_prefix}:satisfies {rdf_prefix}:{corpus_name}_{download_date}_desc .")
    rdf_triples.append(
        f'{rdf_prefix}:{corpus_name}_{download_date}_sit  {rdf_prefix}:hasTimeInterval "{download_date}"^^{xsd_prefix}:decimal .')
    rdf_triples.append(
        f'{rdf_prefix}:{corpus_name}_{download_date}_sit  {rdf_prefix}:hasAnnotator {rdf_prefix}:DelfinaSolMartinezPandiani .')
    rdf_triples.append(f"{rdf_prefix}:collocate rdf:type {rdf_prefix}:CollocationRole .")
    rdf_triples.append(
        f"{rdf_prefix}:{corpus_name}_{download_date}_sit {rdf_prefix}:involvesDataset {rdf_prefix}:{corpus_name} .")

    rdf_triples = list(set(rdf_triples))
    # for triple in rdf_triples:
    #     print(triple)
    return rdf_triples
# keeps collocates only once, with the highest score encountered. also takes out random rows
def clean_sk_eng_csvs(corpus_name, download_date):
    folder_path = str(corpus_name + "_" + download_date)
    output_paths = []
    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and not file_path.endswith("_cleaned.csv"):
            # Perform actions with the file
            print("Processing file:", file_path)

            file_name_parts = file_path.split("_")
            concept_name = file_name_parts[0]
            csv_corpus_name = file_name_parts[1]
            csv_download_date = file_name_parts[2].split(".")[0]  # Remove the file extension
            output_path = str(file_path.split(".")[0] + "_cleaned.csv")
            output_paths.append(output_path)
            if not os.path.exists(output_path):
                filtered_rows = {}

                with open(file_path, "r") as file:
                    reader = csv.reader(file)
                    header = next(reader)  # Read and store the header row

                    for row in reader:
                        try:
                            word = row[2]
                            score = float(row[4])
                            if len(row) >= 4 and row[2] and float(row[-1]) > 5:
                                if word in filtered_rows:
                                    # Compare and update the score
                                    if score > float(filtered_rows[word][4]):
                                        filtered_rows[word] = row
                                else:
                                    filtered_rows[word] = row
                        except (ValueError, IndexError):
                            # Handle the error or skip the row
                            pass

                with open(output_path, "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(header)  # Write the header row

                    for row in filtered_rows.values():
                        writer.writerow(row[1:])
    return output_paths
def generate_collocation_annotations(corpus_name, download_date):
    output_paths = clean_sk_eng_csvs(corpus_name, download_date)
    rdf_triples = []
    for path in output_paths:
        file_name = path.split("/")[-1].replace("_cleaned.csv", "")
        # Extract concept name, corpus name, and download date from the modified file name
        concept_name, corpus_name, download_date = file_name.split("_")
        # Read the CSV file
        with open(path, "r") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            # Dictionary to store collocate information
            collocates = defaultdict(lambda: {"freq_sum": 0, "score_sum": 0, "count": 0})

            # Iterate over each row in the CSV
            for row in reader:
                collocate_word = row[1]
                score = float(row[3])

        # Generate RDF triples for each collocate
                annotation_id = f"{concept_name}_{collocate_word}_{corpus_name}"
                collocate_id = f"conceptnet:{collocate_word}"
                collocate_triples = [
                    f"musco:{corpus_name}_{download_date}_sit musco:involvesCollocationAnnotation musco:{annotation_id} .",
                    f"conceptnet:{concept_name} musco:hasCollocateTypedBy {collocate_id} .",
                    f"musco:{annotation_id} musco:aboutAnnotatedEntity conceptnet:{concept_name} .",
                    f"musco:{annotation_id} musco:annotationTypedBy {collocate_id} .",
                    f"musco:{annotation_id} musco:isClassifiedBy musco:collocate .",
                    f"musco:{annotation_id} musco:hasCollocationStrength '{score}'^^xsd:decimal ."
                ]

                rdf_triples.extend(collocate_triples)

    rdf_triples = list(set(rdf_triples))
  # Print or store the RDF triples as needed
  #   for triple in rdf_triples:
  #       print(triple)

    return rdf_triples


def create_dl_MOSAIC_kg(corpus_name, download_date):
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
    @prefix quokka: <http://etna.istc.cnr.it/quokka/concepts#> .

    @base <https://w3id.org/musco#> .

    """)
    cluster_triples = generate_clusters_triples()
    dl_triples = generate_dl_triples(corpus_name, download_date)
    collocation_triples = generate_collocation_annotations(corpus_name, download_date)
    combined_set = set(cluster_triples)
    combined_set.update(dl_triples)
    combined_set.update(collocation_triples)

    # Output the turtle (.ttl) file
    output_filename = "Distributional-linguistics_MOSAIC_kg.ttl"

    with open(output_filename, "w") as file:
        file.write(prefixes)
        for triple in combined_set:
            file.write(triple + "\n")

    print(f"DL MOSAIC KG created successfully and saved as '{output_filename}'.")

    return
# ## Generate RDF declarations for AbstractConceptClusters and ClusterConcepts
# cluster_triples = generate_clusters_triples()
# ## Generate RDF declarations for Distributional Linguistic KG
# dl_triples = generate_dl_triples("ententen21", "20230627")
# collocation_triples = generate_collocation_annotations("ententen21", "20230627")
# combined_set = set(cluster_triples)
# combined_set.update(dl_triples)
# combined_set.update(collocation_triples)



create_dl_MOSAIC_kg("ententen21", "20230627")