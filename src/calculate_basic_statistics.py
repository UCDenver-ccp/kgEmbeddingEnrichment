# calculate_basic_statistics.py --- Calculate basic distance statistics
#
# Filename: calculate_basic_statistics.py
# Author: Zachary Maas <zama8258@colorado.edu>
# Created: Thu Sep 24 14:42:32 2020 (-0600)
#
#

# Commentary:
#
# This file contains code to perform basic statistical calculations on
# generated vector embeddings from 'calculate_distances.py'. These
# statistics look at both distance distribution and distribution by
# category.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GNU Emacs.  If not, see <https://www.gnu.org/licenses/>.
#
#

# Code:

import re
import json
from collections import Counter
import numpy as np
from scipy.stats import hypergeom
from tqdm import tqdm
import matplotlib.pyplot as plt


def import_distance_distribution(distance_file_path):
    """Read the distance data file grabbing all distances and put them
    into a numpy array. Generates basic distribution plots of that
    array.
    """
    num_lines = 1015946
    num_distances = 255
    ids = np.empty(int(num_lines * num_distances), dtype=np.float32)
    distances = np.empty(int(num_lines * num_distances), dtype=np.float32)
    with open(distance_file_path, "r") as distance_file_handle:
        i = 0
        for line in tqdm(
            distance_file_handle, total=num_lines, desc=f"Importing data"
        ):
            for item in line.strip().split(sep=" ")[1:]:
                item_split = item.split(sep=":")
                ids[i] = item_split[0]
                distances[i] = item_split[1]
                i += 1
    return (ids, distances)


def plot_distance_distribution(distances):
    """Generates a histogram of distance distributions"""
    # Generate initial plot
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["text.usetex"] = "true"
    fig, ax = plt.subplots()
    ax.hist(distances, bins=50, color="black")
    # Set proper scales
    ax.set_yscale("log")
    ax.set_xscale("linear")
    # Remove unnecessary axis spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Set labels
    num_neighbors = 255
    ax.set_title(
        f"Histogram of $L_1$ Norm Nearest Neighbor Distribution\n(n={num_neighbors}/node)"
    )
    ax.set_xlabel("$L_1$ Norm From Neighbor to Node")
    ax.set_ylabel("Number of Neighbors")


def plot_category_distribution(category_counter, threshold=0.01, out_path=None):
    """Takes a Counter object containing and plots all groups that make up
    more than 'threshold' percent of the total

    """
    count_percent = [
        (k, v / sum(category_counter.values()))
        for k, v in category_counter.most_common()
    ]

    count_percent_thresh = [x for x in count_percent if x[1] > threshold]
    count_bar_x = [x[0] for x in count_percent_thresh]
    count_bar_y = [x[1] for x in count_percent_thresh]

    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(1)
    ax.bar(count_bar_x, count_bar_y, color="black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.autofmt_xdate()
    fig.subplots_adjust(bottom=0.3, left=0.2)
    ax.set_title(f"Distribution of Categories with Proportion > {threshold}")
    ax.set_ylabel("Relative Proportion of Items")
    if out_path:
        fig.savefig(out_path)


# Need to add a mapping from URL to category
url_regex_pairs = [
    (r"https://github.com/callahantiff/PheKnowLator/.*", "PheKnowLator"),
    (r"https://reactome.org/content/detail/.*", "Reactome"),
    (r"https://www.ncbi.nlm.nih.gov/snp/.*", "SNP"),
    (r"https://www.ncbi.nlm.nih.gov/gene/.*", "Gene"),
    (
        r"http://www.informatics.jax.org/marker/MGI.*",
        "Mouse Genomics Information",
    ),
    (r"https://uswest.ensembl.org/.*?/Transcript/.*", "Transcript"),
    (r"http://purl.obolibrary.org/obo/OMIM.*", "BBOP Ontology"),
    (r"http://purl.obolibrary.org/obo/HP.*", "Human Phenotype"),
    (r"http://purl.obolibrary.org/obo/RO.*", "Relation Ontology"),
    (r"http://purl.obolibrary.org/obo/BFO.*", "Relation Ontology"),
    (
        r"http://purl.obolibrary.org/obo/NCBITaxon.*",
        "NCBI Organismal Classification",
    ),
    (r"http://purl.obolibrary.org/obo/PATO.*", "Phenotype and Trait Ontology"),
    (
        r"http://purl.obolibrary.org/obo/BSPO.*",
        "Biological and Spatial Ontology",
    ),
    (r"http://www.ebi.ac.uk/efo/EFO.*", "Experimental Factor Ontology"),
    (
        r"http://purl.obolibrary.org/obo/SO.*",
        "Sequence Types and Features Ontology",
    ),
    (r"http://purl.obolibrary.org/obo/AEO.*", "Anatomical Entity Ontology"),
    (r"http://purl.obolibrary.org/obo/AGRO.*", "Agronomy Ontology"),
    (
        r"http://purl.obolibrary.org/obo/AMPHX.*",
        "The Amphioxus Development and Anatomy Ontology",
    ),
    (r"http://purl.obolibrary.org/obo/APO.*", "Ascomycete phenotype ontology"),
    (
        r"http://purl.obolibrary.org/obo/APOLLO_SV.*",
        "Apollo Structured Vocabulary",
    ),
    (r"http://purl.obolibrary.org/obo/ARO.*", "Antibiotic Resistance Ontology"),
    (r"http://purl.obolibrary.org/obo/BCGO.*", "Beta Cell Genomics Ontology"),
    (
        r"http://purl.obolibrary.org/obo/BCO.*",
        "Biological Collections Ontology",
    ),
    (r"http://purl.obolibrary.org/obo/BFO.*", "Basic Formal Ontology"),
    (
        r"http://purl.obolibrary.org/obo/BFO11.*",
        "Basic Formal Ontology (BFO) 1.1",
    ),
    (r"http://purl.obolibrary.org/obo/BSPO.*", "Biological Spatial Ontology"),
    (r"http://purl.obolibrary.org/obo/BTO.*", "BRENDA tissue / enzyme source"),
    (
        r"http://purl.obolibrary.org/obo/CARO.*",
        "Common Anatomy Reference Ontology",
    ),
    (
        r"http://purl.obolibrary.org/obo/CDAO.*",
        "Comparative Data Analysis Ontology",
    ),
    (r"http://purl.obolibrary.org/obo/CEPH.*", "Cephalopod Ontology"),
    (
        r"http://purl.obolibrary.org/obo/CHEBI.*",
        "Chemical Entities of Biological Interest",
    ),
    (
        r"http://purl.obolibrary.org/obo/CHEMINF.*",
        "Chemical Information Ontology",
    ),
    (
        r"http://purl.obolibrary.org/obo/CHIRO.*",
        "CHEBI Integrated Role Ontology",
    ),
    (r"http://purl.obolibrary.org/obo/CHMO.*", "Chemical Methods Ontology"),
    (
        r"http://purl.obolibrary.org/obo/CIDO.*",
        "Coronavirus Infectious Disease Ontology",
    ),
    (
        r"http://purl.obolibrary.org/obo/CIO.*",
        "Confidence Information Ontology",
    ),
    (r"http://purl.obolibrary.org/obo/CL.*", "Cell Ontology"),
    (r"http://purl.obolibrary.org/obo/CLO.*", "Cell Line Ontology"),
    (
        r"http://purl.obolibrary.org/obo/CLO-hPSCreg.*",
        "Cell Line Ontology - Human Pluripotent Stem Cell Registry",
    ),
    (
        r"http://purl.obolibrary.org/obo/CLO-NICR.*",
        "Cell Line Ontology - Chinese National Infrastructure of Cell Line Resource",
    ),
    (
        r"http://purl.obolibrary.org/obo/CLYH.*",
        "Clytia hemisphaerica Development and Anatomy Ontology",
    ),
    (r"http://purl.obolibrary.org/obo/CMO.*", "Clinical measurement ontology"),
    (
        r"http://purl.obolibrary.org/obo/COB.*",
        "Core Ontology for Biology and Biomedicine",
    ),
    (r"http://purl.obolibrary.org/obo/CRO.*", "Contributor Role Ontology"),
    (r"http://purl.obolibrary.org/obo/CTCAE-OAEview.*", "OAE CTCAE view"),
    (r"http://purl.obolibrary.org/obo/CTENO.*", "Ctenophore Ontology"),
    (
        r"http://purl.obolibrary.org/obo/CTO.*",
        "CTO: Core Ontology of Clinical Trials",
    ),
    (
        r"http://purl.obolibrary.org/obo/CVDO.*",
        "Cardiovascular Disease Ontology",
    ),
    (r"http://purl.obolibrary.org/obo/d-acts.*", "Ontology of Document Acts"),
    (
        r"http://purl.obolibrary.org/obo/DDANAT.*",
        "Dictyostelium discoideum anatomy",
    ),
    (
        r"http://purl.obolibrary.org/obo/DDPHENO.*",
        "Dictyostelium discoideum phenotype ontology",
    ),
    (
        r"http://purl.obolibrary.org/obo/DIDEO.*",
        "Drug-drug Interaction and Drug-drug Interaction Evidence Ontology",
    ),
    (r"http://purl.obolibrary.org/obo/DOID.*", "Human Disease Ontology"),
    (r"http://purl.obolibrary.org/obo/DPO.*", "Drosophila Phenotype Ontology"),
    (r"http://purl.obolibrary.org/obo/DRON.*", "The Drug Ontology"),
    (r"http://purl.obolibrary.org/obo/DUO.*", "Data Use Ontology"),
    (
        r"http://purl.obolibrary.org/obo/ECAO.*",
        "The Echinoderm Anatomy and Development Ontology",
    ),
    (r"http://purl.obolibrary.org/obo/ECO.*", "Evidence ontology"),
    (
        r"http://purl.obolibrary.org/obo/ECOCORE.*",
        "An ontology of core ecological entities",
    ),
    (
        r"http://purl.obolibrary.org/obo/ECTO.*",
        "Environmental conditions, treatments and exposures ontology",
    ),
    (r"http://purl.obolibrary.org/obo/EDAM.*", "EMBRACE Data and Methods"),
    (r"http://purl.obolibrary.org/obo/EFO.*", "Experimental Factor Ontology"),
    (
        r"http://purl.obolibrary.org/obo/EHDAA2.*",
        "Human developmental anatomy, abstract",
    ),
    (
        r"http://purl.obolibrary.org/obo/EMAPA.*",
        "Mouse Developmental Anatomy Ontology",
    ),
    (r"http://purl.obolibrary.org/obo/ENVO.*", "Environment Ontology"),
    (r"http://purl.obolibrary.org/obo/EPO.*", "Epidemiology Ontology"),
    (r"http://purl.obolibrary.org/obo/ERO.*", "eagle-i resource ontology"),
    (r"http://purl.obolibrary.org/obo/EUPATH.*", "VEuPathDB ontology"),
    (r"http://purl.obolibrary.org/obo/ExO.*", "Exposure ontology"),
    (r"http://purl.obolibrary.org/obo/FAO.*", "Fungal gross anatomy"),
    (
        r"http://purl.obolibrary.org/obo/FBBI.*",
        "Biological Imaging Methods Ontology",
    ),
    (r"http://purl.obolibrary.org/obo/FBBT.*", "Drosophila gross anatomy"),
    (r"http://purl.obolibrary.org/obo/FBCV.*", "FlyBase Controlled Vocabulary"),
    (r"http://purl.obolibrary.org/obo/FBdv.*", "Drosophila development"),
    (
        r"http://purl.obolibrary.org/obo/FIX.*",
        "Physico-chemical methods and properties",
    ),
    (r"http://purl.obolibrary.org/obo/FLOPO.*", "Flora Phenotype Ontology"),
    (
        r"http://purl.obolibrary.org/obo/FMA.*",
        "Foundational Model of Anatomy Ontology (subset)",
    ),
    (r"http://purl.obolibrary.org/obo/FOBI.*", "FOBI"),
    (r"http://purl.obolibrary.org/obo/FOODON.*", "FOODON"),
    (
        r"http://purl.obolibrary.org/obo/FOVT.*",
        "FuTRES Ontology of Vertebrate Traits",
    ),
    (
        r"http://purl.obolibrary.org/obo/FYPO.*",
        "Fission Yeast Phenotype Ontology",
    ),
    (r"http://purl.obolibrary.org/obo/GAZ.*", "Gazetteer"),
    (
        r"http://purl.obolibrary.org/obo/GECKO.*",
        "Genomics Cohorts Knowledge Ontology",
    ),
    (
        r"http://purl.obolibrary.org/obo/GENEPIO.*",
        "Genomic Epidemiology Ontology",
    ),
    (r"http://purl.obolibrary.org/obo/GENO.*", "Genotype Ontology"),
    (r"http://purl.obolibrary.org/obo/GEO.*", "Geographical Entity Ontology"),
    (
        r"http://purl.obolibrary.org/obo/GNO.*",
        "Glycan Naming and Subsumption Ontology (GNOme)",
    ),
    (r"http://purl.obolibrary.org/obo/GO.*", "Gene Ontology"),
    (r"http://purl.obolibrary.org/obo/HANCESTRO.*", "Human Ancestry Ontology"),
    (r"http://purl.obolibrary.org/obo/HAO.*", "Hymenoptera Anatomy Ontology"),
    (r"http://purl.obolibrary.org/obo/HOM.*", "Homology Ontology"),
    (r"http://purl.obolibrary.org/obo/HP.*", "Human Phenotype Ontology"),
    (r"http://purl.obolibrary.org/obo/HSAPDV.*", "Human Developmental Stages"),
    (r"http://purl.obolibrary.org/obo/HTN.*", "Hypertension Ontology"),
    (r"http://purl.obolibrary.org/obo/IAO.*", "Information Artifact Ontology"),
    (
        r"http://purl.obolibrary.org/obo/IAO-Onto-Meta.*",
        "IAO Ontology Metadata",
    ),
    (
        r"http://purl.obolibrary.org/obo/ICDO.*",
        "International Classification of Disease Ontology",
    ),
    (
        r"http://purl.obolibrary.org/obo/ICEO.*",
        "Integrative and Conjugative Element Ontology",
    ),
    (r"http://purl.obolibrary.org/obo/ICO.*", "Informed Consent Ontology"),
    (r"http://purl.obolibrary.org/obo/IDO.*", "Infectious Disease Ontology"),
    (r"http://purl.obolibrary.org/obo/IDOBRU.*", "Brucellosis Ontology"),
    (r"http://purl.obolibrary.org/obo/IDOMAL.*", "Malaria Ontology"),
    (r"http://purl.obolibrary.org/obo/INO.*", "Interaction Network Ontology"),
    (
        r"http://purl.obolibrary.org/obo/KISAO.*",
        "Kinetic Simulation Algorithm Ontology",
    ),
    (r"http://purl.obolibrary.org/obo/KTAO.*", "Kidney Tissue Atlas Ontology"),
    (r"http://purl.obolibrary.org/obo/LABO.*", "clinical LABoratory Ontology"),
    (r"http://purl.obolibrary.org/obo/LINCS-CLOview.*", "CLO LINCS view"),
    (r"http://purl.obolibrary.org/obo/LTHIDO.*", "LTHIDO"),
    (r"http://purl.obolibrary.org/obo/MA.*", "Mouse adult gross anatomy"),
    (
        r"http://purl.obolibrary.org/obo/MAMO.*",
        "Mathematical modeling ontology",
    ),
    (r"http://purl.obolibrary.org/obo/MAXO.*", "Medical Action Ontology"),
    (r"http://purl.obolibrary.org/obo/MCO.*", "Microbial Conditions Ontology"),
    (r"http://purl.obolibrary.org/obo/MF.*", "Mental Functioning Ontology"),
    (
        r"http://purl.obolibrary.org/obo/MFMO.*",
        "Mammalian Feeding Muscle Ontology",
    ),
    (r"http://purl.obolibrary.org/obo/MFOEM.*", "Emotion Ontology"),
    (r"http://purl.obolibrary.org/obo/MFOMD.*", "Mental Disease Ontology"),
    (
        r"http://purl.obolibrary.org/obo/MI.*",
        "Molecular Interactions Controlled Vocabulary",
    ),
    (r"http://purl.obolibrary.org/obo/MIAPA.*", "MIAPA Ontology"),
    (r"http://purl.obolibrary.org/obo/miRNAO.*", "microRNA Ontology"),
    (
        r"http://purl.obolibrary.org/obo/MIRO.*",
        "Mosquito insecticide resistance",
    ),
    (r"http://purl.obolibrary.org/obo/MMO.*", "Measurement method ontology"),
    (r"http://purl.obolibrary.org/obo/MMUSDV.*", "Mouse Developmental Stages"),
    (r"http://purl.obolibrary.org/obo/MOD.*", "Protein modification"),
    (r"http://purl.obolibrary.org/obo/MONDO.*", "Mondo Disease Ontology"),
    (r"http://purl.obolibrary.org/obo/MOP.*", "Molecular Process Ontology"),
    (r"http://purl.obolibrary.org/obo/MP.*", "Mammalian Phenotype Ontology"),
    (r"http://purl.obolibrary.org/obo/MPATH.*", "Mouse pathology"),
    (
        r"http://purl.obolibrary.org/obo/MPIO.*",
        "Minimum PDDI Information Ontology",
    ),
    (r"http://purl.obolibrary.org/obo/MRO.*", "MHC Restriction Ontology"),
    (r"http://purl.obolibrary.org/obo/MS.*", "Mass spectrometry ontology"),
    (r"http://purl.obolibrary.org/obo/NBO.*", "Neuro Behavior Ontology"),
    (
        r"http://purl.obolibrary.org/obo/NCBITaxon.*",
        "NCBI organismal classification",
    ),
    (r"http://purl.obolibrary.org/obo/NCIT.*", "NCI Thesaurus OBO Edition"),
    (r"http://purl.obolibrary.org/obo/NCRO.*", "Non-Coding RNA Ontology"),
    (
        r"http://purl.obolibrary.org/obo/NDF-RT.*",
        "National Drug File Reference Terminology",
    ),
    (
        r"http://purl.obolibrary.org/obo/NOMEN.*",
        "NOMEN - A nomenclatural ontology for biological names",
    ),
    (r"http://purl.obolibrary.org/obo/OAE.*", "Ontology of Adverse Events"),
    (
        r"http://purl.obolibrary.org/obo/OARCS.*",
        "Ontology of Arthropod Circulatory Systems",
    ),
    (
        r"http://purl.obolibrary.org/obo/OBA.*",
        "Ontology of Biological Attributes",
    ),
    (
        r"http://purl.obolibrary.org/obo/OBCS.*",
        "Ontology of Biological and Clinical Statistics",
    ),
    (
        r"http://purl.obolibrary.org/obo/OBI.*",
        "Ontology for Biomedical Investigations",
    ),
    (
        r"http://purl.obolibrary.org/obo/OBI-NIAID-GSC-BRC-view.*",
        "OBI NIAID-GSC-BRC view",
    ),
    (r"http://purl.obolibrary.org/obo/OBIB.*", "Ontology for Biobanking"),
    (
        r"http://purl.obolibrary.org/obo/OBIws.*",
        "OBI web service, development version",
    ),
    (r"http://purl.obolibrary.org/obo/OCE.*", "Ontology of Chemical Elements"),
    (
        r"http://purl.obolibrary.org/obo/OCMR.*",
        "Ontology of Chinese Medicine for Rheumatism",
    ),
    (
        r"http://purl.obolibrary.org/obo/OCVDAE.*",
        "Ontology of Cardiovascular Drug Adverse Events",
    ),
    (
        r"http://purl.obolibrary.org/obo/ODAE.*",
        "Ontology of Drug Adverse Events",
    ),
    (
        r"http://purl.obolibrary.org/obo/ODNAE.*",
        "Ontologyof Drug Neuropathy Adverse Events",
    ),
    (
        r"http://purl.obolibrary.org/obo/OGG.*",
        "The Ontology of Genes and Genomes",
    ),
    (
        r"http://purl.obolibrary.org/obo/OGG-At.*",
        "Ontology of Genes and Genomes - Arabidopsis thaliana",
    ),
    (
        r"http://purl.obolibrary.org/obo/OGG-Bru.*",
        "Ontology of Genes and Genomes - Brucella",
    ),
    (
        r"http://purl.obolibrary.org/obo/OGG-Ce.*",
        "Ontology of Genes and Genomes - Caenorhabditis elegans",
    ),
    (
        r"http://purl.obolibrary.org/obo/OGG-Dm.*",
        "Ontology of Genes and Genomes - Fruit Fly",
    ),
    (
        r"http://purl.obolibrary.org/obo/OGG-Dr.*",
        "Ontology of Genes and Genomes - Zebrafish",
    ),
    (
        r"http://purl.obolibrary.org/obo/OGG-Mm.*",
        "Ontology of Genes and Genomes - Mouse",
    ),
    (
        r"http://purl.obolibrary.org/obo/OGG-Pf.*",
        "Ontology of Genes and Genomes - Plasmodium falciparum",
    ),
    (
        r"http://purl.obolibrary.org/obo/OGG-Sc.*",
        "Ontology of Genes and Genomes - Yeast",
    ),
    (r"http://purl.obolibrary.org/obo/OGI.*", "Ontology for genetic interval"),
    (
        r"http://purl.obolibrary.org/obo/OGMS.*",
        "Ontology for General Medical Science",
    ),
    (
        r"http://purl.obolibrary.org/obo/OGSF.*",
        "Ontology of Genetic Susceptibility Factor",
    ),
    (
        r"http://purl.obolibrary.org/obo/OHD.*",
        "The Oral Health and Disease Ontology",
    ),
    (
        r"http://purl.obolibrary.org/obo/OHMI.*",
        "Ontology of Host-Microbiome Interactions",
    ),
    (
        r"http://purl.obolibrary.org/obo/OHPI.*",
        "Ontology of Host Pathogen Interactions",
    ),
    (r"http://purl.obolibrary.org/obo/OLATDV.*", "Medaka Developmental Stages"),
    (
        r"http://purl.obolibrary.org/obo/OLOBO.*",
        "Ontology for Linking Biological and Medical Ontologies",
    ),
    (r"http://purl.obolibrary.org/obo/OMIABIS.*", "Ontologized MIABIS"),
    (r"http://purl.obolibrary.org/obo/OMIT.*", "Ontology for MIRNA Target"),
    (r"http://purl.obolibrary.org/obo/OMO.*", "OBO Metadata Ontology"),
    (
        r"http://purl.obolibrary.org/obo/OMP.*",
        "Ontology of Microbial Phenotypes",
    ),
    (
        r"http://purl.obolibrary.org/obo/OMRSE.*",
        "Ontology of Medically Related Social Entities",
    ),
    (r"http://purl.obolibrary.org/obo/ONS.*", "ONS"),
    (
        r"http://purl.obolibrary.org/obo/ONTONEO.*",
        "Obstetric and Neonatal Ontology",
    ),
    (
        r"http://purl.obolibrary.org/obo/OOSTT.*",
        "Ontology of Organizational Structures of Trauma centers and Trauma systems",
    ),
    (
        r"http://purl.obolibrary.org/obo/OPL.*",
        "Ontology for Parasite LifeCycle",
    ),
    (
        r"http://purl.obolibrary.org/obo/OPMI.*",
        "Ontology of Precision Medicine and Investigation",
    ),
    (r"http://purl.obolibrary.org/obo/ORNASEQ.*", "Ontology of RNA Sequencing"),
    (
        r"http://purl.obolibrary.org/obo/OSCI.*",
        "Ontology for Stem Cell Investigations",
    ),
    (
        r"http://purl.obolibrary.org/obo/OVAE.*",
        "Ontology of Vaccine Adverse Events",
    ),
    (r"http://purl.obolibrary.org/obo/PATO.*", "Phenotype And Trait Ontology"),
    (
        r"http://purl.obolibrary.org/obo/PCO.*",
        "Population and Community Ontology",
    ),
    (
        r"http://purl.obolibrary.org/obo/PDRO.*",
        "The Prescription of Drugs Ontology",
    ),
    (
        r"http://purl.obolibrary.org/obo/PDUMDV.*",
        "Platynereis Developmental Stages",
    ),
    (
        r"http://purl.obolibrary.org/obo/PECO.*",
        "Plant Experimental Conditions Ontology",
    ),
    (
        r"http://purl.obolibrary.org/obo/PHIPO.*",
        "Pathogen Host Interaction Phenotype Ontology",
    ),
    (r"http://purl.obolibrary.org/obo/PLANA.*", "planaria-ontology"),
    (r"http://purl.obolibrary.org/obo/PLANP.*", "Planarian Phenotype Ontology"),
    (r"http://purl.obolibrary.org/obo/PNO.*", "Proper Name Ontology"),
    (r"http://purl.obolibrary.org/obo/PO.*", "Plant Ontology"),
    (r"http://purl.obolibrary.org/obo/PORO.*", "Porifera Ontology"),
    (r"http://purl.obolibrary.org/obo/PPO.*", "Plant Phenology Ontology"),
    (r"http://purl.obolibrary.org/obo/PR.*", "PRotein Ontology (PRO)"),
    (r"http://purl.obolibrary.org/obo/PSO.*", "Plant Stress Ontology"),
    (r"http://purl.obolibrary.org/obo/PW.*", "Pathway ontology"),
    (r"http://purl.obolibrary.org/obo/REX.*", "Physico-chemical process"),
    (r"http://purl.obolibrary.org/obo/RNAO.*", "RNA ontology"),
    (r"http://purl.obolibrary.org/obo/RO.*", "Relation Ontology"),
    (r"http://purl.obolibrary.org/obo/RS.*", "Rat Strain Ontology"),
    (r"http://purl.obolibrary.org/obo/RXNO.*", "Name Reaction Ontology"),
    (r"http://purl.obolibrary.org/obo/SBO.*", "Systems Biology Ontology"),
    (
        r"http://purl.obolibrary.org/obo/SDGIO.*",
        "Sustainable Development Goals Interface Ontology",
    ),
    (
        r"http://purl.obolibrary.org/obo/SEPIO.*",
        "Scientific Evidence and Provenance Information Ontology",
    ),
    (
        r"http://purl.obolibrary.org/obo/SIBO.*",
        "Social Insect Behavior Ontology",
    ),
    (
        r"http://purl.obolibrary.org/obo/SIO.*",
        "Semanticscience Integrated Ontology",
    ),
    (
        r"http://purl.obolibrary.org/obo/SO.*",
        "Sequence types and features ontology",
    ),
    (r"http://purl.obolibrary.org/obo/SPD.*", "Spider Ontology"),
    (
        r"http://purl.obolibrary.org/obo/STATO.*",
        "The Statistical Methods Ontology",
    ),
    (r"http://purl.obolibrary.org/obo/SWO.*", "Software ontology"),
    (r"http://purl.obolibrary.org/obo/SYMP.*", "Symptom Ontology"),
    (r"http://purl.obolibrary.org/obo/TADS.*", "Tick Anatomy Ontology"),
    (r"http://purl.obolibrary.org/obo/TAXRANK.*", "Taxonomic rank vocabulary"),
    (
        r"http://purl.obolibrary.org/obo/TGMA.*",
        "Mosquito gross anatomy ontology",
    ),
    (r"http://purl.obolibrary.org/obo/TO.*", "Plant Trait Ontology"),
    (
        r"http://purl.obolibrary.org/obo/TRANS.*",
        "Pathogen Transmission Ontology",
    ),
    (r"http://purl.obolibrary.org/obo/TSO.*", "Transportation System Ontology"),
    (r"http://purl.obolibrary.org/obo/TTO.*", "Teleost taxonomy ontology"),
    (r"http://purl.obolibrary.org/obo/TXPO.*", "Toxic Process Ontology"),
    (
        r"http://purl.obolibrary.org/obo/UBERON.*",
        "Uberon multi-species anatomy ontology",
    ),
    (r"http://purl.obolibrary.org/obo/UO.*", "Units of measurement ontology"),
    (r"http://purl.obolibrary.org/obo/UPA.*", "Unipathway"),
    (
        r"http://purl.obolibrary.org/obo/UPHENO.*",
        "Unified phenotype ontology (uPheno)",
    ),
    (r"http://purl.obolibrary.org/obo/VariO.*", "Variation Ontology"),
    (
        r"http://purl.obolibrary.org/obo/VICO.*",
        "Vaccination Informed Consent Ontology",
    ),
    (r"http://purl.obolibrary.org/obo/VIO.*", "Vaccine Investigation Ontology"),
    (r"http://purl.obolibrary.org/obo/VIVO-ISF.*", "VIVO-ISF"),
    (r"http://purl.obolibrary.org/obo/VO.*", "Vaccine Ontology"),
    (r"http://purl.obolibrary.org/obo/VT.*", "Vertebrate trait ontology"),
    (r"http://purl.obolibrary.org/obo/VTO.*", "Vertebrate Taxonomy Ontology"),
    (
        r"http://purl.obolibrary.org/obo/WBbt.*",
        "C. elegans Gross Anatomy Ontology",
    ),
    (
        r"http://purl.obolibrary.org/obo/WBLS.*",
        "C. elegans development ontology",
    ),
    (r"http://purl.obolibrary.org/obo/WBPhenotype.*", "C. elegans phenotype"),
    (r"http://purl.obolibrary.org/obo/XAO.*", "Xenopus Anatomy Ontology"),
    (
        r"http://purl.obolibrary.org/obo/XCO.*",
        "Experimental condition ontology",
    ),
    (r"http://purl.obolibrary.org/obo/XPO.*", "Xenopus Phenotype Ontology"),
    (
        r"http://purl.obolibrary.org/obo/ZECO.*",
        "Zebrafish Experimental Conditions Ontology",
    ),
    (
        r"http://purl.obolibrary.org/obo/ZFA.*",
        "Zebrafish anatomy and development ontology",
    ),
    (
        r"http://purl.obolibrary.org/obo/ZFS.*",
        "Zebrafish developmental stages ontology",
    ),
    (r"http://purl.obolibrary.org/obo/ZP.*", "Zebrafish Phenotype Ontology"),
]


def rels_to_categorical(rels_dic, regex_pairs=url_regex_pairs):
    """Converts a relation to it's broader ontological category"""
    rels_categorical = {}
    for identifier, url in tqdm(
        rels_dic.items(), desc="Converting ids to category"
    ):
        for pattern, category in regex_pairs:
            if re.match(pattern, url, flags=re.IGNORECASE):
                rels_categorical[identifier] = category
                break
            rels_categorical[identifier] = url
    return rels_categorical


def import_instance_rels(instance_rels_path):
    """Import instance relations from a given JSON file"""
    with open(instance_rels_path) as instance_rels_handle:
        rels_dic_ini = json.load(instance_rels_handle)
    rels_dic = {}
    for k, v in rels_dic_ini.items():
        rels_dic[v] = k
    return rels_dic


l_1_path = "/hdd/data/embeddingEnrichment/distances/l_2_norm.emb.distances"
l_1_values = import_distance_distribution(l_1_path)
l_1_ids = l_1_values[0]
l_1_distances = l_1_values[1]
rels = import_instance_rels(
    "/home/zach/Dropbox/phd/research/hunter/embeddingEnrichment/data/PheKnowLator_Instance_RelationsOnly_NotClosed_NoOWL_Triples_Integer_Identifier_Map.json"
)
rels_categorical = rels_to_categorical(rels)

# Get counts and percentage of each id globally
ids_present_categorical = [
    rels_categorical[i]
    for i in tqdm(l_1_ids, desc="Converting id -> categorical")
]
count_categorical = Counter(ids_present_categorical)
plot_category_distribution(
    count_categorical, threshold=0.01, out_path="l_2_data_dist.eps"
)


# plot_category_distribution(neighbor_counter_categorical, threshold=0)

# Use hypergeometric test, p = survival function
# What's an appropriate population to use?


# plot_category_distribution(
#     neighbor_counter_categorical, threshold=0, out_path="sample_data_dist.eps"
# )

# prob = test_prob(4, len(rels), 255 * len(prelim_ids))

# Test unpickling
# import pickle

# with open("/hdd/data/embeddingEnrichment/model.pckl", "rb") as model_handle:
#     model = pickle.load(model_handle)

#
# calculate_basic_statistics.py ends here
