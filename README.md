# MOSAIC
## Multimodal Ontology for Social, Abstract and Intangible Concepts

This repo holds all content related to the MOSAIC project, which aims to use structured knowledge (the MOSAIC knowledge graph) as a tool for more explainable visual detection of abstract social concepts (ASCs), conceptualized as clusters.

[Reference to Hypericons for Explainability resources]([url](https://github.com/delfimpandiani/ARTstract_Seeing_abstract_concepts)https://github.com/delfimpandiani/ARTstract_Seeing_abstract_concepts), including baseline detectors (finetuned VGG).

The MOSAIC knowledge graph is based on the [MOSAIC ontology](MOSAIC/MOSAIC_ontology.ttl), which reuses the Descriptions and Situations (DnS) ontology design pattern to represent data annotated with abstract social concepts. The MOSAIC KG will have four iterations:

  1. [Common-sense MOSAIC](MOSAIC/Common-sense_MOSAIC.ttl), which situates 7 novel nodes for the 7 ASC clusters with the most important connections each of these have with ConceptNet concepts.

  2. Distributional Linguistic MOSAIC, which transforms distributional linguistic data from corpora analysis into structured knowledge, based on word collocates.

  3. Sensory-Perceptual MOSAIC, which transforms image data focusing on object, action, emotional, and color detection into structured knowledge.
 
  4. Parasitic MOSAIC, the combination of the three previous KGs, mirroring the latest cognitive science research on human ASC representation: a parasitic relationship between distributional and sensory-perceptual data.

![MOSAIC](https://github.com/delfimpandiani/mosaic/assets/44606644/a587b9dc-84ee-448a-aad7-25d17cb013a7)
