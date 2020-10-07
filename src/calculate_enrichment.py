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
from os.path import basename
from import_graph import open_owl_graph
from tqdm import tqdm
from scipy.stats import hypergeom
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
    with open(distance_file_path, "r") as distance_file_handle:
        i = 0
        for line in tqdm(
            distance_file_handle, desc="Getting idlist neighbor ids"
        ):
            line_split = line.strip().split(sep=" ")
            curr_id = int(line_split[0].split(":")[0])
            if curr_id in id_list:
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
urls = [f"https://www.ncbi.nlm.nih.gov/gene/{x}" for x in prelim_ids]
matches = [
    (k, v)
    for k, v in tqdm(rels.items(), desc="Finding matched ids")
    if v in urls
]
if len(matches) != len(prelim_ids):
    print("An error occurred looking up your gene id in the relations dict")
matched_ids = [x[0] for x in matches]
id_neighbors = get_id_neighbors(l_1_path, matched_ids)
# id_neighbors_categorical = [rels_categorical[x] for x in id_neighbors]
# neighbor_counter_categorical = Counter(id_neighbors_categorical)
neighbor_counter = Counter(id_neighbors)
shared_neighbors = {
    x: count for x, count in neighbor_counter.items() if count > 1
}
shared_neighbor_ids = [rels[i] for i in shared_neighbors]


# Try iterating
def get_all_predicates(subject, graph):
    """Get all predicates with some relation to subject in graph using
    RDFlib
    """
    matched_predicates = []
    if subject is str:
        subject = rdflib.URIref(subject)
    matched_triples = graph.triples((subject, None, None))
    try:
        triples_list = list(matched_triples)
        if len(triples_list) > 0:
            for _, _, pred in triples_list:
                # print(f"{subj} {obj} {pred}")
                matched_predicates.append(pred)
    except EOFError:
        # In case the resulting generator is empty
        pass
    except Exception:
        # How to handle DBPageNotFoundError from Sleepycat
        print(f"Lookup failed for {subject}")
    return matched_predicates


owl_identifier = "PheKnowLator_OWL"
dir_prefix = "/home/zach/data"
owl_uri = f"{dir_prefix}/rdflib_PheKnowLator_OWL_store"
collected_predicates = set()
curr_iter_preds = shared_neighbor_ids.copy()
matched_onto_terms = []
with open_owl_graph(owl_uri, owl_identifier) as owl_graph:
    # Check if we have a hit for ontology terms yet
    print(len(collected_predicates))
    first_iter = 0
    while matched_onto_terms == []:
        # If no match, walk through predicates as subjects
        next_iter_preds = []
        for neighbor_id in curr_iter_preds:
            item_preds = [
                pred for pred in get_all_predicates(neighbor_id, owl_graph)
            ]
            next_iter_preds.extend(item_preds)
        # Use the new set of predicates for the next iteration
        collected_predicates.update(x for x in next_iter_preds)
        curr_iter_preds = next_iter_preds.copy()
        matched_onto_terms = [
            x for x in collected_predicates if "/GO" in str(x)
        ]
        # if len(prev_preds) == 0:
        #     break
filtered = [
    x
    for x in list(collected_predicates)
    if "gene" not in str(x)
    and "Transcript" not in str(x)
    and "github" not in str(x)
]

# if 'http://purl.obolibrary.org/obo/GO' in str(x)
with open_owl_graph(owl_uri, owl_identifier) as owl_graph:
    for s, o, p in owl_graph.triples(
        (
            rdflib.URIRef(
                "https://github.com/callahantiff/PheKnowLator/pkt/Nc29b3d4e99cac49385f6a97b66ea84df"
                # "https://github.com/callahantiff/PheKnowLator/pkt/N20215c47a9d7f7ff7b7fed7ef634d007"
            ),
            None,
            None,
        )
    ):
        print(s, o, p)

#
# calculate_enrichment.py ends here
