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

import json
import numpy as np
import scipy.stats as st
from tqdm import tqdm
import matplotlib.pyplot as plt


def calculate_distance_distribution(distance_file_path):
    """Read the distance data file grabbing all distances and put them
    into a numpy array. Generates basic distribution plots of that
    array.
    """
    num_lines = 1015946
    num_distances = 255
    distances = np.empty(int(num_lines * num_distances), dtype=np.float32)
    with open(distance_file_path, "r") as distance_file_handle:
        i = 0
        for line in tqdm(distance_file_handle, total=num_lines):
            for item in line.strip().split(sep=" ")[1:]:
                distances[i] = item.split(sep=":")[1]
                i += 1
    return distances


def plot_distance_distribution(distances):
    """Generates a histogram of distance distributions"""
    # Generate initial plot
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["text.usetex"] = "true"
    fig, ax = plt.subplots()
    ax.hist(l_1_distances, bins=50, color="black")
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


def import_instance_rels(instance_rels_path):
    with open(instance_rels_path) as instance_rels_handle:
        rels_dic_ini = json.load(instance_rels_handle)
    rels_dic = {}
    for k, v in rels_dic_ini.items():
        rels_dic[v] = k
    return rels_dic


l_1_path = "/hdd/data/embeddingEnrichment/distances/l_2_norm.emb.distances"
# l_1_distances = calculate_distance_distribution(l_1_path)
rels = import_instance_rels(
    "/home/zach/Dropbox/phd/research/hunter/embeddingEnrichment/data/PheKnowLator_Instance_RelationsOnly_NotClosed_NoOWL_Triples_Integer_Identifier_Map.json"
)


#
# calculate_basic_statistics.py ends here
