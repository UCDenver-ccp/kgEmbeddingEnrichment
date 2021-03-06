# calculate_distances.py --- Calculate distances of a KG embedding
#
# Filename: calculate_distances.py
# Author: Zachary Maas <zama8258@colorado.edu>
# Created: Tue Sep  8 15:02:09 2020 (-0600)
#
#

# Commentary:
#
# This code calculates a few different distance metrics for a given
# knowledge graph embedding, writing out a separate file for each
# metric storing each node and the N closest nodes along with their
# distances. By default N=256 and the calculated norms are L1, L2, and
# cosine similarity.
#
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

import argparse
from tqdm import tqdm
import numpy as np
from numba import njit
from joblib import Parallel, delayed


def import_data(embedding_filename: str):
    """Read the file located at 'embedding_filename'. This returns two
    objects - a numpy array of the embedding values and a pandas
    dataframe for outputting the nearest neighbors into.

    """
    with open(embedding_filename, "r") as embedding_file_handle:
        # The first line contains the number of lines and the number
        # of dimensions of the embedding
        embedding_metadata = (
            embedding_file_handle.readline().strip().split(sep=" ")
        )
        embedding_num_items = int(embedding_metadata[0])
        embedding_dimension = int(embedding_metadata[1])

        # Pre-allocate the embedding matrix for speed
        index_matrix = np.empty((embedding_num_items, 1))
        embedding_matrix = np.empty(
            (embedding_num_items, embedding_dimension), np.float32
        )

        #
        for idx, line in tqdm(
            enumerate(embedding_file_handle.readlines()),
            unit="lines",
            desc="input data",
        ):
            line_data = line.strip().split(sep=" ")
            index_matrix[idx] = line_data[0]
            embedding_matrix[idx] = line_data[1:]
    return (index_matrix, embedding_matrix)


@njit
def l_1_norm(row_m, row_n):
    """Calculate the manhattan distance between row_m and row_n"""
    return np.linalg.norm(np.subtract(row_m, row_n), ord=1)


@njit
def l_2_norm(row_m, row_n):
    """Calculate the euclidean distance between row_m and row_n"""
    return np.linalg.norm(np.subtract(row_m, row_n), ord=2)


@njit
def l_inf_norm(row_m, row_n):
    """Calculate the L_\\inf norm between row_m and row_n"""
    return np.linalg.norm(np.subtract(row_m, row_n), ord=np.inf)


@njit
def neg_cosine_similarity(row_m, row_n):
    """Calculate the cosine similarity between row m and row n. Note that
    we take the negative value so that most similar things have positive
    values.
    """
    return -(
        np.dot(row_m, row_n) / (np.linalg.norm(row_m) * np.linalg.norm(row_n))
    )


@njit
def get_distances(
    index_matrix, embedding_matrix, current_row_pos, norm_function
):
    """Create a separate function for the distance getting operations so
    that we can run it through numba's JIT. This function iterates
    over every row in the matrix calculating a distance relative to
    the current item using the provided norm.

    """
    current_row = embedding_matrix[current_row_pos]
    distances = np.empty(index_matrix.size)
    for comparison_row_pos in range(index_matrix.size):
        distances[comparison_row_pos] = norm_function(
            current_row,
            embedding_matrix[comparison_row_pos],
        )
    return distances


def calculate_min_item_distances(
    index_matrix,
    embedding_matrix,
    current_row_pos: int,
    num_closest: int,
    norm_function,
):
    """Calculate the closest 'num_closest'-1 items for row 'current_row_idx'
    in embedding matrix. Returns an array of index/distance pairs for
    easier processing further down the line. 'num_closest' - 1 is used
    as an optimization so that full vectorization can be used without
    having to worry about dropping the current row of interest.

    """
    distances = get_distances(
        index_matrix, embedding_matrix, current_row_pos, norm_function
    )
    # Experimental linear time indices of smallest elements
    # distance_indices = np.argpartition(distances, -num_closest)[-num_closest:]
    distance_indices = distances.argsort()[:num_closest]
    # Stack the distance and index (identifier) to keep IDs known
    index_distance_pairs = np.column_stack(
        (distances[distance_indices], index_matrix[distance_indices])
    )
    sorted_index_distance_pairs = index_distance_pairs[
        index_distance_pairs[:, 0].argsort(kind="mergesort")
    ]
    return sorted_index_distance_pairs


def distance_iteration(
    line,
    index_matrix,
    embedding_matrix,
    num_closest,
    norm_fns,
):
    """Sloppy attempt to get numba to parallelize the main loop further"""
    min_item_distances = []
    for norm_fn in norm_fns:
        min_item_distances.append(
            calculate_min_item_distances(
                index_matrix=index_matrix,
                embedding_matrix=embedding_matrix,
                current_row_pos=line,
                num_closest=num_closest,
                norm_function=norm_fn,
            )
        )
    return min_item_distances


def calculate_distances(
    embedding_filename: str, output_path: str, num_closest: int = 256
):
    """Calculate the pairwise distances for the KG embedding using all
    provided norms, saving the maximum N values to a file for each
    type of embedding.
    """
    norm_fns = (l_1_norm, l_2_norm, l_inf_norm, neg_cosine_similarity)

    index_matrix, embedding_matrix = import_data(embedding_filename)

    def curr_iter(line):
        """Wrapper for the iteration function to allow for threaded mapping"""
        return distance_iteration(
            line=line,
            index_matrix=index_matrix,
            embedding_matrix=embedding_matrix,
            num_closest=num_closest,
            norm_fns=norm_fns,
        )

    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(curr_iter)(line)
        for line in tqdm(
            range(len(embedding_matrix)),
            unit="calcs",
            desc="distance calculations",
        )
    )

    # Write each norm function one at a time
    for norm_fn_idx, norm_fn in enumerate(norm_fns):
        fn_name = norm_fn.__name__
        result_file_path = f"{output_path}/{fn_name}.emb.distances"
        with open(result_file_path, "w") as result_file_handle:
            for res_item in tqdm(results, unit="lines", desc=fn_name):
                # Grab that norm's specific output
                fn_output = res_item[norm_fn_idx]
                # Write each norm item
                for item in fn_output:
                    result_file_handle.write(f"{int(item[1])}:{item[0]} ")
                # Break the line
                result_file_handle.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate distances from vector embeddings"
    )
    parser.add_argument(
        "-d",
        "--data",
        help="the embedding data file",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        help="the directory to write output to",
    )

    args = parser.parse_args()
    calculate_distances(
        embedding_filename=args.data, output_path=args.output_path
    )

#
# calculate_distances.py ends here
