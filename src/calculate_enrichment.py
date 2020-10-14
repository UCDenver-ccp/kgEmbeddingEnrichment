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

import pickle
from collections import Counter
import json
from os.path import basename, exists
import triples_sqlite
from tqdm import tqdm
from scipy.stats import fisher_exact


def unpickle_or_process(pickle_path, alt_function):
    """Attempts to unpickle the file at path, otherwise runs 'function'
    and put its output as a pickle. Function should be a lambda to
    defer evaluation to only the false loop.

    """
    if exists(pickle_path):
        with open(pickle_path, "rb") as pickle_handle:
            pickled = pickle.load(pickle_handle)
    else:
        pickled = alt_function()
        with open(pickle_path, "wb") as pickle_handle:
            pickle.dump(pickled, pickle_handle)
    return pickled


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
                for item in line_split[0:65]:
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
    urls = set()
    for identifier in id_list:
        if "http" in str(identifier):
            urls.update([str(identifier)])
        else:
            # Assume gene id if not specified
            urls.update([f"https://www.ncbi.nlm.nih.gov/gene/{identifier}"])
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
        x: count for x, count in neighbor_counter.items()  # if count > 1
    }
    shared_neighbors = [rels[i] for i in shared_neighbors]
    return shared_neighbors


def get_all_predicates(subject, graph):
    """Get all predicates with some relation to subject in graph using
    RDFlib
    """
    matched_predicates = set()
    matched_triples = graph.triples(subj=subject, obj=None, pred=None)
    try:
        if matched_triples is not None:  # and len(matched_triples) > 0:
            for _, _, pred in (x for x in matched_triples):
                matched_predicates.update([pred])
    except EOFError:
        # In case the resulting generator is empty
        pass
    except ValueError:
        # In case the resulting list is empty
        pass
    # except Exception:
    #     # How to handle DBPageNotFoundError from Sleepycat
    #     # print(f"Lookup failed for {subject}")
    #     # subprocess.check_call(["db_recover", "-ch", owl_uri])
    #     # graph.open(owl_uri)
    #     return []
    #     # pass
    return list(matched_predicates)


# def get_nearest_onto_terms(subjects, onto_str_match, graph):
#     """Get the nearest nodes matching onto_str_match in graph starting
#     from curr_subjects. This search is independent of subsumption
#     hierarchy and serves mostly to walk through the anonymous nodes
#     that result from creation of the OWL graph.

#     """
#     collected_predicates = []
#     prev_len = len(collected_predicates) + 1
#     curr_iter_preds = subjects.copy()
#     matched_onto_terms = []
#     loop_count = 0
#     # Check if we have a hit for ontology terms yet
#     while len(collected_predicates) != prev_len:
#         next_iter_preds = set()
#         for neighbor_id in tqdm(
#             curr_iter_preds, desc=f"Initial onto search, loop {loop_count}"
#         ):
#             item_preds = [
#                 pred
#                 for pred in get_all_predicates(neighbor_id, graph)
#                 if pred not in collected_predicates
#                 and ("obolibrary" in str(pred) or "github" in str(pred))
#             ]
#             for k, item in enumerate(item_preds):
#                 if "github" in str(item):
#                     item_preds[k] = basename(item)
#             next_iter_preds.update(item_preds)
#         # Use the new set of predicates for the next iteration
#         prev_len = len(collected_predicates)
#         collected_predicates.extend(next_iter_preds)
#         curr_iter_preds = next_iter_preds.copy()
#         matched_onto_terms.extend(
#             [x for x in collected_predicates if onto_str_match in str(x)]
#         )
#         if len(curr_iter_preds) == 0:
#             return matched_onto_terms
#             raise Exception("Graph walk exhausted.")
#         loop_count += 1
#     return matched_onto_terms


# def get_onto_hierarchy_terms(onto_terms, onto_str_match, graph):
#     """Given MATCHED_ONTO_TERMS, returns all terms matching subClassOf for
#     those terms all the way up their ontology hierarchy.

#     """
#     hierarchy_onto_terms = onto_terms.copy()
#     hierarchy_onto_set = set(hierarchy_onto_terms)
#     next_onto_terms = onto_terms.copy()
#     loop_count = 0
#     curr_onto_len = 0
#     while len(hierarchy_onto_terms) != curr_onto_len:
#         curr_onto_len = len(hierarchy_onto_terms)
#         new_onto_terms = []
#         for onto_term in tqdm(
#             next_onto_terms, desc=f"Hierarchy walk, loop {loop_count}"
#         ):
#             # try:
#             loop_terms = [
#                 p
#                 for _, _, p in [
#                     x
#                     for x in graph.triples(
#                         subj=onto_term,
#                         obj="http://www.w3.org/2000/01/rdf-schema#subClassOf",
#                     )
#                 ]
#                 if onto_str_match in str(p)
#             ]
#             for k, item in enumerate(loop_terms):
#                 if "github" in str(item):
#                     loop_terms[k] = basename(item)
#             new_onto_terms.extend(loop_terms)
#             # except Exception:
#             #     # How to handle DBPageNotFoundError from Sleepycat
#             #     # print(f"Lookup failed for {subject}")
#             #     # subprocess.check_call(["db_recover", "-ch", owl_uri])
#             #     # graph.open(owl_uri)
#             #     pass
#             loop_count += 1
#         hierarchy_onto_terms.extend(new_onto_terms)
#         next_onto_terms = [
#             x for x in new_onto_terms if x not in hierarchy_onto_set
#         ]
#         hierarchy_onto_set.update(new_onto_terms)
#     return hierarchy_onto_terms


# def get_onto_counter(subjects, onto_str_match, graph):
#     """Gets a counter containing all matched ontology terms and their
#     frequency for later statistical analysis.

#     """
#     matched_go_terms = get_nearest_onto_terms(subjects, onto_str_match, graph)
#     hierarchy_go_terms = get_onto_hierarchy_terms(
#         matched_go_terms, onto_str_match, graph
#     )
#     onto_counter = Counter(hierarchy_go_terms)
#     return onto_counter


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
distance_path = "/hdd/data/embeddingEnrichment/distances/l_1_norm.emb.distances"
rels = import_instance_rels(
    "/home/zach/Dropbox/phd/research/hunter/embeddingEnrichment/data/PheKnowLator_Instance_RelationsOnly_NotClosed_NoOWL_Triples_Integer_Identifier_Map.json"
)

prelim_ids = [
    "http://purl.obolibrary.org/obo/PR_000005079",
    "http://purl.obolibrary.org/obo/PR_000015642",
    "http://purl.obolibrary.org/obo/PR_000006342",
    "http://purl.obolibrary.org/obo/PR_000014688",
    "http://purl.obolibrary.org/obo/PR_000013073",
    "http://purl.obolibrary.org/obo/PR_000007897",
    "http://purl.obolibrary.org/obo/PR_000012604",
    "http://purl.obolibrary.org/obo/PR_000001752",
    "http://purl.obolibrary.org/obo/PR_000016871",
    "http://purl.obolibrary.org/obo/PR_000009274",
    "http://purl.obolibrary.org/obo/PR_000006348",
    "http://purl.obolibrary.org/obo/PR_000004188",
    "http://purl.obolibrary.org/obo/PR_000030257",
    "http://purl.obolibrary.org/obo/PR_000004621",
    "http://purl.obolibrary.org/obo/PR_000008292",
    "http://purl.obolibrary.org/obo/PR_000012750",
    "http://purl.obolibrary.org/obo/PR_000008829",
    "http://purl.obolibrary.org/obo/PR_000010511",
    "http://purl.obolibrary.org/obo/PR_000003560",
    "http://purl.obolibrary.org/obo/PR_000013767",
    "http://purl.obolibrary.org/obo/PR_000007096",
]

# Get neighbor ids
shared_neighbor_ids = unpickle_or_process(
    "/home/zach/data/shared_ids.pickle",
    lambda: get_id_neighbor_list(prelim_ids, distance_path),
)
# shared_neighbor_ids = get_id_neighbor_list(prelim_ids, distance_path)

# Read gene ids
# gene_list_path = "/hdd/data/embeddingEnrichment/Homo_sapiens.gene_info"
# gene_ids = read_gene_info(gene_list_path)
protein_ids = [x for x in rels.values() if "/PR_" in x]
all_gene_neighbor_ids = unpickle_or_process(
    "/home/zach/data/protein_ids.pickle",
    lambda: get_id_neighbor_list(protein_ids, distance_path),
)
# all_gene_neighbor_ids = get_id_neighbor_list(protein_ids, distance_path)

# Find GO terms
owl_identifier = "PheKnowLator_OWL"
dir_prefix = "/home/zach/data"
owl_uri = f"{dir_prefix}/rdflib_PheKnowLator_OWL_store"


def hash_filter_p(x):
    return len(str(x)) == 33 and " " not in str(x)


def get_onto_neighbors_sql(search_list, onto_str_match, graph):
    new_keys = Counter()
    prev_len = -1
    curr_len = len(new_keys) - 1
    next_list = search_list
    loop_iter = 0
    while len(next_list) != prev_len:
        prev_len = len(next_list)
        curr_len = len(new_keys)
        print(f"Initial search loop {loop_iter}, new items: {len(next_list)}")
        query_list_str = ",".join(['"' + str(x) + '"' for x in next_list])
        query = (
            f"select predicate from triples where subject in ({query_list_str})"
        )
        all_matches = graph.connection.execute(query)
        found_list = set(
            x[0]
            for x in tqdm(all_matches.fetchall())
            if (hash_filter_p(x[0]) or onto_str_match in str(x[0]))
            and x[0] not in new_keys
        )
        next_list = [x for x in found_list if x not in new_keys]
        new_keys.update(found_list)
        loop_iter += 1
    return new_keys


def counter_len(counter):
    return sum(counter.values())


def walk_onto_tree_sql(search_list, onto_str_match, graph):
    tree_counter = Counter()
    curr_len = counter_len(tree_counter) - 1
    prev_len = curr_len - 1
    next_list = search_list.keys()
    loop_iter = 0
    while (
        counter_len(tree_counter) != curr_len
        and len(next_list) != 0
        and prev_len != curr_len
    ):
        prev_len = curr_len
        curr_len = len(tree_counter)
        print(f"Graph search loop {loop_iter}, new items: {len(next_list)}")
        query_list_str = ",".join(['"' + str(x) + '"' for x in next_list])
        subclass_term_str = '"http://www.w3.org/2000/01/rdf-schema#subClassOf"'
        query = f"select predicate from triples where subject in ({query_list_str}) and object = {subclass_term_str}"
        all_matches = graph.connection.execute(query)
        found_list = [
            x[0]
            for x in tqdm(all_matches.fetchall())
            if (hash_filter_p(x[0]) or onto_str_match in str(x[0]))
            # and x[0] not in tree_counter.keys()
        ]
        next_list = found_list
        # next_list = found_list
        tree_counter.update(x for x in found_list if not hash_filter_p(x))
        loop_iter += 1
    return tree_counter


# Pull the GO terms of interest
# logging.getLogger("rdflib").setLevel(logging.CRITICAL)
# with open_owl_graph(owl_uri, owl_identifier) as owl_graph:
uri = "/home/zach/data/triples.db"
owl_graph = triples_sqlite.SQLiteTripleGraph(uri)
owl_graph.open(uri)


def get_onto_counter(subjects, onto_str_match, graph):
    """Gets a counter containing all matched ontology terms and their
    frequency for later statistical analysis.

    """

    ini_terms = get_onto_neighbors_sql(subjects, onto_str_match, owl_graph)
    tree_counter = walk_onto_tree_sql(ini_terms, onto_str_match, owl_graph)
    # onto_counter = Counter(tree_terms)
    return tree_counter


# Only make the counter for global terms if the file isn't on disk
global_counter_file = "/home/zach/data/global_counter.pickle"
if exists(global_counter_file):
    print("Reading global terms from file")
    with open(global_counter_file, "rb") as global_counter_handle:
        global_counter = pickle.load(global_counter_handle)
else:
    print("Finding hierarchy terms globally")
    global_counter = get_onto_counter(
        all_gene_neighbor_ids,
        "obolibrary",
        owl_graph,
    )
    print("Writing global terms to file")
    with open(global_counter_file, "wb") as global_counter_handle:
        pickle.dump(global_counter, global_counter_handle)

# Make the counter for the experiment
print("Finding hierarchy terms for experiment")
local_counter_file = "/home/zach/data/local_counter.pickle"
if exists(local_counter_file):
    print("Reading local terms from file")
    with open(local_counter_file, "rb") as local_counter_handle:
        local_counter = pickle.load(local_counter_handle)
else:
    print("Finding hierarchy terms locally")
    local_counter = get_onto_counter(
        shared_neighbor_ids,
        "obolibrary",
        owl_graph,
    )
    print("Writing local terms to file")
    with open(local_counter_file, "wb") as local_counter_handle:
        pickle.dump(local_counter, local_counter_handle)

owl_graph.close()

# For making output more informative
labels_file_path = "/hdd/data/embeddingEnrichment/PheKnowLator_Instance_RelationsOnly_NotClosed_NoOWL_NodeLabels.txt"
labels_dic = {}
with open(labels_file_path, "r") as labels_file_handle:
    for line in labels_file_handle:
        try:
            line_split = line.strip().split("\t")
            labels_dic[line_split[0]] = line_split[1]
        except IndexError:
            pass


print("Performing statistical tests")


def write_enrichment_results(
    local_counter, global_counter, labels_dic, filter_str, out_path
):
    """Write out enrichment results using local_counter and global_counter
    as the references for Fisher exact testing with Bonferroni
    testing. Labels dic uses a PheKnowLator nodelabels file to write
    out the labels for each enriched item, to facilitate easier
    interpretation by humans. The resulting tab separted file is
    written to OUT_PATH.

    """
    # Normalize keys to strings, in case an int popped up somewhere
    print(f"Analyzing enrichment for {filter_str}")
    present_terms = {str(k): v for k, v in local_counter.items()}
    global_terms = {
        str(k): v
        for k, v in global_counter.items()
        if str(k) in present_terms.keys()
    }

    # Prepare for testing by pulling items
    interest_items = Counter(
        {k: v for k, v in local_counter.items() if filter_str in k}
    )
    p_thresh = 1e-10
    p_corr = p_thresh / len(interest_items)
    term_total = sum(present_terms.values())
    global_total = sum(interest_items.values())

    # Build contingency table: [pos_local pos_global] [non_local non_global]
    # Do the test for each item we've collected
    res_arr = []
    for term, pos_local in tqdm(
        interest_items.most_common(), desc="Fisher Exact Tests"
    ):
        try:
            pos_global = global_terms[str(term)]
            non_local = term_total - pos_local
            non_global = global_total - pos_global
            odds, p_value = fisher_exact(
                [[pos_local, pos_global], [non_local, non_global]],
                alternative="less",
            )
            res_arr.append((term, p_value, odds))
        except KeyError:
            # FIXME find missing keys in global version
            print(f"Missing key {term} in global.")
            pass

    # Grab only significant items passing bonferroni threshold
    significant_arr = sorted(
        [(str(k), p, o) for k, p, o in res_arr if p < p_corr],
        key=lambda x: x[1],
    )

    # Write output
    with open(out_path, "w+") as out_file_handle:
        for k, v, o in significant_arr:
            base = basename(k)
            desc = labels_dic[base]
            out_file_handle.write(f"{k}\t{v}\t{o}\t{desc}\n")


for ontology in ("DOID", "GO", "CHEBI", "PR", "obolibrary"):
    write_enrichment_results(
        local_counter,
        global_counter,
        labels_dic,
        ontology,
        f"/home/zach/data/{ontology}_enrichment_results.txt",
    )


#
# calculate_enrichment.py ends here
