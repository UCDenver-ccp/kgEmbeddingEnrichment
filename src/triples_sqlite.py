# triples_sqlite.py --- Wrappers for handling a graph using SQLite
#
# Filename: triples_sqlite.py
# Author: Zachary Maas <zama8258@colorado.edu>
# Created: Sat Oct 10 12:37:21 2020 (-0600)
#
#

# Commentary:
#
#
# After persistent issues with using the Berkeley DB backend for
# accessing a generated OWL database, I decided to write my own
# implementation of the graph class, roughly compatible with RDFlib,
# which uses SQLite3 as a backend and queries it directly.
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

import sqlite3
from tqdm import tqdm


class SQLiteTripleGraph:
    """A RDFlib compatible triple-query-able Graph class that uses SQLite
    as the backend for greater compatibility

    """

    def __init__(self, identifier="triples"):
        """Initialize the graph class, and indicate that we want to use the
        table identified using IDENTIFIER

        """
        self.database = None
        self.connection = None
        self.table = identifier

    def open(self, uri):
        """Open the database"""
        self.database = sqlite3.connect(uri)
        self.connection = self.database.cursor()

    def close(self):
        """Close the database"""
        self.connection.close()
        self.database.commit()
        self.database.close()

    def parse(self, file_path):
        """Read from TSV file into the graph database."""
        create_string = "create table if not exists triples (subject string, object string, predicate string);"
        self.connection.execute(create_string)
        with open(file_path, "r") as file_path_handle:
            # self.connection.execute("begin transaction;")
            for line in tqdm(file_path_handle, desc="Creating database"):
                line_split = line.strip().split("\t")
                try:
                    subj = line_split[0]
                    obj = line_split[1]
                    pred = line_split[2]
                    insert_string = "insert into triples (subject, object, predicate) values (?, ?, ?);"
                    self.connection.execute(insert_string, (subj, obj, pred))
                except IndexError:
                    pass
            # self.connection.execute("commit;")

    def triples(self, subj=None, obj=None, pred=None):
        """Returns all triples matching subject, object, or predicate in any
        combination, returning a generator of those triples.

        """
        # Get the number of terms
        query_terms = [x for x in [subj, obj, pred] if x is not None]
        num_query_terms = len(query_terms)
        # Determine the column to query
        cols = []
        if subj:
            cols.append("subject")
        if obj:
            cols.append("object")
        if pred:
            cols.append("predicate")
        # Run each query
        try:
            if num_query_terms == 0:
                # Empty yield if empty query
                return []
            if num_query_terms == 1:
                # One term query
                query_string = f"select * from triples where {cols[0]} = ?;"
                matches = self.connection.execute(
                    query_string, (query_terms[0],)
                )
                return matches
            if num_query_terms == 2:
                # Two term query
                query_string = f"select * from triples where {cols[0]} = ? and {cols[1]} = ?;"
                matches = self.connection.execute(
                    query_string,
                    (
                        query_terms[0],
                        query_terms[1],
                    ),
                )
                return matches
            if num_query_terms == 2:
                # Three term query
                query_string = f"select * from triples where {cols[0]} = ? and {cols[1]} = ? and {cols[2]} = ?;"
                matches = self.connection.execute(
                    query_string,
                    (
                        query_terms[0],
                        query_terms[1],
                        query_terms[2],
                    ),
                )
                return matches
        except IndexError:
            print("Index issue on {cols}")
            return []


# uri = "/home/zach/data/triples.db"
# to_import = "/hdd/data/embeddingEnrichment/PheKnowLator_Instance_RelationsOnly_NotClosed_OWL_Triples_Identifiers.txt"
# graph = SQLiteTripleGraph()
# graph.open(uri)
# graph.parse(to_import)
# graph.close()

#
# triples_sqlite.py ends here
