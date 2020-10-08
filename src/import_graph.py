# import_graph.py --- Imports an OWL Graph using RDFlib
#
# Filename: import_graph.py
# Author: Zachary Maas
# Created: Fri Oct  2 11:12:47 2020 (-0600)
#
#

# Commentary:
#
#
# This file contains utility functions for importing PheKnowLator OWL
# graphs using RDFlib. Use of the graph is needed so that associated
# terms in the subsumption hierarchy can be found for each nearest
# neighbor to allow for enrichment to be calculated effectively.
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

from contextlib import contextmanager
import rdflib


@contextmanager
def open_owl_graph(uri, identifier, graph_path=None):
    """Provides a context manager for working with an OWL graph while also
    automatically closing it afterward. URI is the location of the
    graph store directory and IDENTIFIER is the name of the graph
    within that store. Optional argument GRAPH_PATH specifies an
    appropriately formatted RDF file to import when opening the graph.

    """
    try:
        # Only force create if a path is provided
        create_graph = bool(graph_path)
        # Open and load the on-disk store
        graph_id = rdflib.URIRef(identifier)
        graph = rdflib.Graph("Sleepycat", identifier=graph_id)
        graph.open(uri, create=create_graph)
        # Parse the file at GRAPH_PATH if set
        if graph_path:
            graph.parse(graph_path)
        yield graph
    finally:
        try:
            graph.close()
        except Exception:
            pass


#
# import_graph.py ends here
