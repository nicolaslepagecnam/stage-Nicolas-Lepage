#!pip install --upgrade gmsh # line to run on Colab to download gmsh

# Read file
import gmsh
from mpi4py import MPI
from dolfinx.graph import create_adjacencylist
from dolfinx.cpp.mesh import to_type, cell_entity_type
from dolfinx.mesh import create_mesh, create_meshtags
from dolfinx.io import XDMFFile, cell_perm_gmsh, distribute_entity_data, extract_gmsh_geometry, extract_gmsh_topology_and_markers, ufl_mesh_from_gmsh
import numpy as np

filename = directory+"/mesh/cylinder.msh"
gmsh.initialize()
gmsh.model.add("Mesh from file")
gmsh.merge(filename)

rank = MPI.COMM_WORLD.rank
if rank == 0:
    # Get mesh geometry
    x = extract_gmsh_geometry(gmsh.model)

    # Get mesh topology for each element
    topologies = extract_gmsh_topology_and_markers(gmsh.model)

    # Get information about each cell type from the msh files
    num_cell_types = len(topologies.keys())
    cell_information = {}
    cell_dimensions = np.zeros(num_cell_types, dtype=np.int32)

    for i, element in enumerate(topologies.keys()):
        properties = gmsh.model.mesh.getElementProperties(element)
        name, dim, order, num_nodes, local_coords, _ = properties
        cell_information[i] = {"id": element, "dim": dim, "num_nodes": num_nodes}
        cell_dimensions[i] = dim

    # Sort elements by ascending dimension
    perm_sort = np.argsort(cell_dimensions)

    # Broadcast cell type data and geometric dimension
    cell_id = cell_information[perm_sort[-1]]["id"]
    tdim = cell_information[perm_sort[-1]]["dim"]
    num_nodes = cell_information[perm_sort[-1]]["num_nodes"]
    cell_id, num_nodes = MPI.COMM_WORLD.bcast([cell_id, num_nodes], root=0)
    if tdim - 1 in cell_dimensions:
        num_facet_nodes = MPI.COMM_WORLD.bcast( cell_information[perm_sort[-2]]["num_nodes"], root=0)
        gmsh_facet_id = cell_information[perm_sort[-2]]["id"]
        marked_facets = np.asarray(topologies[gmsh_facet_id]["topology"], dtype=np.int64)
        facet_values = np.asarray(topologies[gmsh_facet_id]["cell_data"], dtype=np.int32)

    cells = np.asarray(topologies[cell_id]["topology"], dtype=np.int64)
    cell_values = np.asarray(topologies[cell_id]["cell_data"], dtype=np.int32)

else:
    cell_id, num_nodes = MPI.COMM_WORLD.bcast([None, None], root=0)
    cells, x = np.empty([0, num_nodes], np.int64), np.empty([0, gdim])
    cell_values = np.empty((0,), dtype=np.int32)
    num_facet_nodes = MPI.COMM_WORLD.bcast(None, root=0)
    marked_facets = np.empty((0, num_facet_nodes), dtype=np.int64)
    facet_values = np.empty((0,), dtype=np.int32)

# Create distributed mesh
gdim = 2
ufl_domain = ufl_mesh_from_gmsh(cell_id, gdim)
gmsh_cell_perm = cell_perm_gmsh(to_type(str(ufl_domain.ufl_cell())), num_nodes)
cells = cells[:, gmsh_cell_perm]
mesh = create_mesh(MPI.COMM_WORLD, cells, x[:, :gdim], ufl_domain)
tdim = mesh.topology.dim
fdim = tdim - 1

# Permute facets from MSH to DOLFINx ordering
# FIXME: Last argument is 0 as all facets are the same for tetrahedra
facet_type = cell_entity_type(to_type(str(ufl_domain.ufl_cell())), fdim, 0)
gmsh_facet_perm = cell_perm_gmsh(facet_type, num_facet_nodes)
marked_facets = np.asarray(marked_facets[:, gmsh_facet_perm], dtype=np.int64)

local_entities, local_values = distribute_entity_data(mesh, fdim, marked_facets, facet_values)
mesh.topology.create_connectivity(fdim, tdim)
adj = create_adjacencylist(local_entities)
# Create DOLFINx MeshTags
ft = create_meshtags(mesh, fdim, adj, np.int32(local_values))
ft.name = "Facet tags"

# Output DOLFINx meshes to file
with XDMFFile(MPI.COMM_WORLD, "cylinder.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(ft)

gmsh.finalize()
