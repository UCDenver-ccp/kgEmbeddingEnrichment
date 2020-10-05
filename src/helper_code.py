#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import needed libraries
import json
import pandas as pd
from tqdm import tqdm


## DOWNLOAD DATA FIRST
# drop url into a browser and you will be prompted to download data

# FILE 1 - owl-nets node patch data
# https://www.dropbox.com/s/73fxs9rm2cdxufn/OWL-NETS_InstanceBuild_BNode_Patch.json?dl=1

# FILE 2 - owl-nets edges
# https://www.dropbox.com/s/98p2y5dqy562dlq/PheKnowLator_Instance_RelationsOnly_NotClosed_NoOWL_Triples_Identifiers
# .txt?dl=1

# FILE 3 - owl-nets node metadata
# https://www.dropbox.com/s/035qlwylap0zhqq/PheKnowLator_Instance_RelationsOnly_NotClosed_NoOWL_NodeLabels.txt?dl=1


##### OWL-NETS INSTANCE-BASED BUILD PATCH #####

# replace BNodes in OWL-NETS edge list with OWL-Class
patch_file = ''  # file 1 above

with open(patch_file, 'r') as read_file:
    bnode_patch = json.load(read_file)

# read in OWL-NETS edge list
edge_ids = ''  # file 2 above

with open(edge_ids) as f_in:
    owl_nets_edges = [x.split('\t')[0::2] for x in tqdm(f_in.read().splitlines()[1:])]
f_in.close()

# iterate over OWL-NETS nodes and replace BNodes
updated_node_ids = []

for edge in tqdm(owl_nets_edges):
    if 'pkt' in edge[0]:
        updated_node_ids.append([bnode_patch[edge[0]], edge[1]])
    elif 'pkt' in edge[1]:
        updated_node_ids.append([edge[0], bnode_patch[edge[1]]])
    elif 'pkt' in edge[0] and 'pkt' in edge[1]:
        updated_node_ids.append([bnode_patch[edge[0]], bnode_patch[edge[1]]])
    else:
        updated_node_ids.append(edge)


##### OWL-NETS ACCESS LABELS #####

# read in node metadata
label_file = ''  # file 3 above
data_label = pd.read_csv(label_file, sep='\t', header=0, low_memory=False)

# convert labels to dictionary - for node identifiers and labels
label_content = {}
for idx, row in tqdm(data_label.iterrows(), total=data_label.shape[0]):
    label_content[row['node_id']] = row['label']

# label patched nodes
labeled_nodes, missing_node_labels = [], []

for edge in tqdm(updated_node_ids):
    sub = edge[0].split('/')[-1].split('=')[-1]
    obj = edge[1].split('/')[-1].split('=')[-1]

    if sub in label_content.keys():
        labeled_nodes.append([sub, label_content[sub]])
    if sub not in label_content.keys():
        missing_node_labels.append(edge[0])
    if obj in label_content.keys():
        labeled_nodes.append([obj, label_content[obj]])
    if obj not in label_content.keys():
        missing_node_labels.append(edge[1])
