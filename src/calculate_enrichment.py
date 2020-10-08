# calculate_enrichment.py --- enrichment calculation
#
# Filename: calculate_enrichment.py
# Author:
# Created: Tue Oct  6 13:50:23 2020 (-0600)
#
#

# Commentary:
#
#
# This file contains the base code required to do enrichment
# calculations for the embeddingEnrichment method.
#
#
# his program is free software: you can redistribute it and/or modify
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

from collections import Counter
import json
import logging
from import_graph import open_owl_graph
from tqdm import tqdm
from scipy.stats import fisher_exact
import rdflib


def import_instance_rels(instance_rels_path):
    """Import instance relations from a given JSON file"""
    with open(instance_rels_path) as instance_rels_handle:
        rels_dic_ini = json.load(instance_rels_handle)
    rels_dic = {}
    for k, v in rels_dic_ini.items():
        rels_dic[v] = k
    return rels_dic


def get_id_neighbors(distance_file_path, id_list):
    """Given id_list, import all nearest neighbors into one list"""
    neighbor_ids = []
    id_set = set(id_list)
    with open(distance_file_path, "r") as distance_file_handle:
        i = 0
        for line in tqdm(
            distance_file_handle, desc="Getting idlist neighbor ids"
        ):
            line_split = line.strip().split(sep=" ")
            curr_id = int(line_split[0].split(":")[0])
            if curr_id in id_set:
                for item in line_split[1:]:
                    item_split = item.split(sep=":")
                    neighbor_ids.append(int(item_split[0]))
                    i += 1
    return neighbor_ids


def test_prob(num_hits, pop_size, num_draws, num_matching=1):
    """Perform a hypergeometric test to see the probability of drawing the
    same entity num_hits many times. Need to check math

    """
    dist = hypergeom(pop_size, num_matching, num_draws)
    pval = dist.sf(num_hits - 1)
    return pval


def get_id_neighbor_list(id_list, neighbor_file_path):
    """Get all matched neighbors for id_list"""
    urls = set(f"https://www.ncbi.nlm.nih.gov/gene/{x}" for x in id_list)
    matches = [
        (k, v)
        for k, v in tqdm(rels.items(), desc="Finding matched ids")
        if v in urls
    ]
    if len(matches) != len(id_list):
        print("An error occurred looking up your gene id in the relations dict")
    matched_ids = [x[0] for x in matches]
    id_neighbors = get_id_neighbors(neighbor_file_path, matched_ids)
    neighbor_counter = Counter(id_neighbors)
    shared_neighbors = {
        x: count for x, count in neighbor_counter.items() if count > 1
    }
    shared_neighbors = [rels[i] for i in shared_neighbors]
    return shared_neighbors


def get_all_predicates(subject, graph):
    """Get all predicates with some relation to subject in graph using
    RDFlib
    """
    matched_predicates = []
    if isinstance(subject, str):
        subject = rdflib.URIRef(subject)
    matched_triples = graph.triples((subject, None, None))
    try:
        triples_list = list(matched_triples)
        if len(triples_list) > 0:
            for _, _, pred in triples_list:
                matched_predicates.append(pred)
    except EOFError:
        # In case the resulting generator is empty
        pass
    except Exception:
        # How to handle DBPageNotFoundError from Sleepycat
        # print(f"Lookup failed for {subject}")
        pass
    return matched_predicates


def get_nearest_onto_terms(subjects, onto_str_match, graph):
    """Get the nearest nodes matching onto_str_match in graph starting
    from curr_subjects. This search is independent of subsumption
    hierarchy and serves mostly to walk through the anonymous nodes
    that result from creation of the OWL graph.

    """
    collected_predicates = set()
    prev_len = len(collected_predicates) + 1
    curr_iter_preds = subjects.copy()
    anonymous_terms = []
    matched_onto_terms = []
    loop_count = 0
    # Check if we have a hit for ontology terms yet
    while (
        len(matched_onto_terms) == 0
        # or len(anonymous_terms) > 0
        or len(collected_predicates) != prev_len
    ):
        anonymous_terms = [
            x
            for x in curr_iter_preds
            if "github" in str(x) and x not in collected_predicates
        ]
        next_iter_preds = []
        for neighbor_id in tqdm(
            curr_iter_preds, desc=f"Initial onto search, loop {loop_count}"
        ):
            item_preds = [
                pred
                for pred in get_all_predicates(neighbor_id, graph)
                if pred not in collected_predicates
            ]
            next_iter_preds.extend(item_preds)
        # Use the new set of predicates for the next iteration
        collected_predicates.update(next_iter_preds)
        prev_len = len(collected_predicates)
        curr_iter_preds = next_iter_preds.copy()
        if len(curr_iter_preds) == 0:
            raise Exception("Graph walk exhausted.")
        matched_onto_terms.extend(
            [x for x in collected_predicates if onto_str_match in str(x)]
        )
        loop_count += 1
    return matched_onto_terms


def get_onto_hierarchy_terms(onto_terms, onto_str_match, graph):
    """Given MATCHED_ONTO_TERMS, returns all terms matching subClassOf for
    those terms all the way up their ontology hierarchy.

    """
    hierarchy_onto_terms = onto_terms.copy()
    hierarchy_onto_set = set(hierarchy_onto_terms)
    next_onto_terms = onto_terms.copy()
    loop_count = 0
    curr_onto_len = 0
    while len(hierarchy_onto_terms) != curr_onto_len:
        curr_onto_len = len(hierarchy_onto_terms)
        new_onto_terms = []
        for onto_term in tqdm(
            next_onto_terms, desc=f"Hierarchy walk, loop {loop_count}"
        ):
            try:
                loop_terms = [
                    p
                    for _, _, p in graph.triples(
                        (
                            onto_term,
                            rdflib.URIRef(
                                "http://www.w3.org/2000/01/rdf-schema#subClassOf"
                            ),
                            None,
                        )
                    )
                    if onto_str_match in str(p)
                ]
                new_onto_terms.extend(loop_terms)
            except Exception:
                # How to handle DBPageNotFoundError from Sleepycat
                # print(f"Lookup failed for {subject}")
                pass
        loop_count += 1
        hierarchy_onto_terms.extend(new_onto_terms)
        next_onto_terms = [
            x for x in new_onto_terms if x not in hierarchy_onto_set
        ]
        hierarchy_onto_set.update(new_onto_terms)
    return hierarchy_onto_terms


def get_onto_counter(subjects, onto_str_match, graph):
    """Gets a counter containing all matched ontology terms and their
    frequency for later statistical analysis.

    """
    matched_go_terms = get_nearest_onto_terms(subjects, onto_str_match, graph)
    hierarchy_go_terms = get_onto_hierarchy_terms(
        matched_go_terms, onto_str_match, graph
    )
    onto_counter = Counter(hierarchy_go_terms)
    return onto_counter


def read_gene_info(gene_list_path):
    """Helper function to read a NCBI gene list file and return a list of
    all gene identifiers for use in enrichment analysis."""
    gene_ids = []
    with open(gene_list_path, "r") as gene_list_handle:
        # Skip the header line
        gene_list_handle.readline()
        for line in gene_list_handle:
            gene_ids.append(int(line.strip().split(sep="\t")[1]))
    return gene_ids


# Constant paths for testing
l_1_path = "/hdd/data/embeddingEnrichment/distances/l_2_norm.emb.distances"
rels = import_instance_rels(
    "/home/zach/Dropbox/phd/research/hunter/embeddingEnrichment/data/PheKnowLator_Instance_RelationsOnly_NotClosed_NoOWL_Triples_Integer_Identifier_Map.json"
)

prelim_ids = [
    57181,
    7316,
    1263,
    4092,
    7566,
    5468,
    4023,
]

# Get neighbor ids
shared_neighbor_ids = get_id_neighbor_list(prelim_ids, l_1_path)

# Read gene ids
gene_list_path = "/hdd/data/embeddingEnrichment/Homo_sapiens.gene_info"
gene_ids = read_gene_info(gene_list_path)
all_gene_neighbor_ids = get_id_neighbor_list(gene_ids, l_1_path)

# Find GO terms
owl_identifier = "PheKnowLator_OWL"
dir_prefix = "/home/zach/data"
owl_uri = f"{dir_prefix}/rdflib_PheKnowLator_OWL_store"

# Pull the GO terms of interest
logging.getLogger("rdflib").setLevel(logging.CRITICAL)
with open_owl_graph(owl_uri, owl_identifier) as owl_graph:
    print("Finding hierarchy terms for experiment")
    term_counter = get_onto_counter(
        shared_neighbor_ids, "obolibrary", owl_graph
    )
    print("Finding hierarchy terms globally")
    global_counter = get_onto_counter(
        all_gene_neighbor_ids,
        "obolibrary",
        owl_graph,
    )

print("Performing statistical tests")
present_terms = {str(k): v for k, v in term_counter.items()}
global_terms = {
    str(k): v
    for k, v in global_counter.items()
    if str(k) in present_terms.keys()
}

# The actual test
p_thresh = 0.05
p_corr = p_thresh / len(term_counter)
term_total = sum(present_terms.values())
global_total = sum(global_terms.values())
# Build contigency table: [pos_local pos_global] [non_local non_global]
res_arr = []
for term, pos_local in term_counter.most_common():
    try:
        pos_global = global_terms[str(term)]
        non_local = term_total - pos_local
        non_global = global_total - pos_global
        _, p_value = fisher_exact(
            [[pos_global, pos_local], [non_global, non_local]]
        )
        res_arr.append((term, p_value))
    except KeyError:
        print(f"Missing key {term} in global.")
significant_arr = sorted(
    [(str(k), p) for k, p in res_arr if p < p_corr],
    key=lambda x: x[1],
)

#
# calculate_enrichment.py ends here
