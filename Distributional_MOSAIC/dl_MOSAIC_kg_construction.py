import json
import csv
import os
from collections import defaultdict
import textwrap
import re
import csv
import googletrans
from googletrans import Translator

import csv
from googletrans import Translator

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
def generate_situation_triples(corpus_name, download_date, annotator):
    # RDF prefixes
    rdf_prefix = "musco"

    collocation_sit = f"{rdf_prefix}:{corpus_name}_{download_date}_sit"
    collocation_desc = f"{rdf_prefix}:{corpus_name}_{download_date}_desc"
    corpus = f"{rdf_prefix}:{corpus_name}"
    collocation_role = f"{rdf_prefix}:collocate"
    annotator = f"{rdf_prefix}:{annotator}"

    situation_triples = [
        ## Instance declarations
        f"{collocation_sit} rdf:type {rdf_prefix}:CollocationAnnotationSituation .",
        f"{collocation_desc} rdf:type {rdf_prefix}:CollocationAnnotationDescription .",
        f"{corpus} rdf:type {rdf_prefix}:Dataset .",
        f"{collocation_role} rdf:type {rdf_prefix}:CollocationRole .",
        f"{annotator} rdf:type {rdf_prefix}:Annotator .",

        ## Object property declarations
        f"{collocation_sit} {rdf_prefix}:satisfies {collocation_desc} .",
        f'{collocation_sit} {rdf_prefix}:hasAnnotator {annotator} .',
        f'{collocation_sit} {rdf_prefix}:involvesDataset {corpus} .',

        ## Data property declarations
        f"{collocation_sit} {rdf_prefix}:hasDate '{download_date}'^^xsd:dateTime .",
    ]
    # RDF triples
    situation_triples = list(set(situation_triples))
    # for triple in situation_triples:
    #     print(triple)
    return situation_triples

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
                            if (
                                len(row) >= 4
                                and row[2]
                                and float(row[-1]) > 5
                                and len(word) > 1
                                and not word.isdigit()
                                and all(c.isalpha() for c in word)
                            ):
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

def translate_corpus_csv(output_paths, corpus_lan):
    translated_output_paths = []
    translator = Translator(service_urls=['translate.google.com'])

    for file_path in output_paths:
        # Read the CSV file
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            lines = list(reader)

        # Extract the base name and extension from the input file
        base_name, extension = os.path.splitext(file_path)

        # Split the base name using '_' as the separator
        base_name_parts = base_name.split('_')

        # Translate the specific word (e.g., 'comodidad')
        if base_name_parts[1] == str(download_date + "/libertad"):
            translated_word = str(download_date + "/freedom")
        elif base_name_parts[1] == str(download_date + "/seguridad"):
            translated_word = str(download_date + "/safety")
        else:
            translated_word = translator.translate(base_name_parts[1], src=corpus_lan, dest='en').text.lower()
        #print(base_name_parts[1], "is translated to ", translated_word)

        # Replace the specific word with the translated version
        base_name_parts[1] = translated_word

        # Recreate the translated base name
        translated_base_name = '_'.join(base_name_parts)

        # Create the translated output file name
        output_file = f'{translated_base_name}{extension}'
        translated_output_paths.append(output_file)
        # print("NEW OUTPUT PATH", output_file)

        if os.path.exists(output_file):
            print(f'Translated CSV file "{output_file}" already exists. Skipping translation.')
            continue

        # Translate the lines
        translated_lines = []
        for line in lines:
            if len(line) >= 2:
                word = line[1]
                translation = translator.translate(word, src=corpus_lan, dest='en')
                translated_word = translation.text.lower()

                # Remove 'to' from verb phrases
                translated_word_parts = translated_word.split()
                if len(translated_word_parts) > 1 and translated_word_parts[0].lower() == 'to':
                    translated_word_parts = translated_word_parts[1:]
                translated_word = ' '.join(translated_word_parts)

                if len(translated_word_parts) > 1:
                    translated_word = '-'.join(translated_word_parts)

                translated_line = [line[0], translated_word] + line[2:]
                translated_lines.append(translated_line)

        # Write the translated lines to the output CSV file
        with open(output_file, 'w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(translated_lines)

        print(f'Translated CSV file saved as: {output_file}')
    print("thjis is the end of output paths", translated_output_paths)
    return translated_output_paths


def generate_collocation_triples(corpus_name, download_date, corpus_lan):
    rdf_prefix = "musco"
    collocation_triples = []
    print(corpus_lan)
    if corpus_lan != "en":
        output_paths = clean_sk_eng_csvs(corpus_name, download_date)
        output_paths = translate_corpus_csv(output_paths, corpus_lan)
        # create the translated csvs and better specify the output paths
    else:
        output_paths = clean_sk_eng_csvs(corpus_name, download_date)
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

            collocate_triples = []  # Initialize list to store triples for each CSV file

            # Iterate over each row in the CSV
            for row in reader:
                collocate_word = row[1]
                score = float(row[3])

                # Generate RDF triples for each collocate
                annotation_id = f"{rdf_prefix}:{concept_name}_{collocate_word}_{corpus_name}_{download_date}"
                collocate_le = f"{rdf_prefix}:le_{collocate_word}"
                annotated_concept = f"conceptnet:{concept_name}"
                collocate_concept = f"conceptnet:{collocate_word}"
                collocation_sit = f"{rdf_prefix}:{corpus_name}_{download_date}_sit"

                collocate_triples.extend([
                    ## Instance declarations
                    f"{annotation_id} rdf:type {rdf_prefix}:CollocationAnnotation .",
                    f"{collocate_le} rdf:type {rdf_prefix}:LexicalEntry .",
                    f"{annotated_concept} rdf:type {rdf_prefix}:Concept .",
                    f"{collocate_concept} rdf:type {rdf_prefix}:Concept .",

                    ## Object property declarations
                    f"{collocation_sit} {rdf_prefix}:involvesCollocationAnnotation {annotation_id} .",
                    f"{collocation_sit} {rdf_prefix}:involvesAnnotatedEntity {annotated_concept} .",
                    f"{rdf_prefix}:collocate {rdf_prefix}:classifies {annotation_id} .",
                    f"{annotation_id} {rdf_prefix}:aboutAnnotatedEntity {annotated_concept} .",
                    f"{annotation_id} {rdf_prefix}:annotationWithLexicalEntry {collocate_le} .",
                    f"{annotated_concept} {rdf_prefix}:isAnnotatedWithLexicalEntry {collocate_le} .",
                    f"{collocate_le} {rdf_prefix}:typedByConcept {collocate_concept} .",
                    f"{annotated_concept} {rdf_prefix}:hasCollocateTypedBy {collocate_concept} .",

                    ## Data property declarations
                    f"{annotation_id} {rdf_prefix}:hasAnnotationStrength '{score}'^^xsd:decimal ."
                ])

        collocation_triples.extend(collocate_triples)  # Add triples for the current CSV file to the main list
    return collocation_triples

def create_collocation_situation_kg(corpus_name, download_date, annotator, corpus_lan="en"):
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
    for triple in cluster_triples:
        print(triple)
    situation_triples = generate_situation_triples(corpus_name, download_date, annotator)
    for triple in situation_triples:
        print(triple)
    collocation_triples = generate_collocation_triples(corpus_name, download_date, corpus_lan)
    for triple in collocation_triples:
        print(triple)
    combined_set = set(cluster_triples)
    combined_set.update(situation_triples)
    combined_set.update(collocation_triples)

    # Output the turtle (.ttl) file
    output_filename = f"{corpus_name}_{download_date}_Distributional-linguistics_MOSAIC_kg.ttl"

    with open(output_filename, "w") as file:
        file.write(prefixes)
        for triple in combined_set:
            file.write(triple + "\n")

    print(f"DL MOSAIC KG created successfully for collocation situation {corpus_name}_{download_date} by {annotator} and saved as '{output_filename}'.")

    return









# Specifics of a collocation annotation situation
corpus_name = "parlaMint21"
corpus_lan = "es"
download_date = "20230703"
annotator = "DelfinaSolMartinezPandiani"

# create that collocation situation's DL KG
create_collocation_situation_kg(corpus_name, download_date, annotator, corpus_lan=corpus_lan)

