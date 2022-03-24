# !pip install --upgrade gmsh # line to run on Colab to download gmsh

import gmsh
from mpi4py import MPI
import numpy as np

gmsh.initialize()
#**** Creation of the domain geometry
L = 1.6
H = 1
c_x, c_y = 0.2, 0.5
r = 0.05
gdim = 2

rank = MPI.COMM_WORLD.rank
if rank == 0:
    rectangle = gmsh.model.occ.addRectangle(0,0,0, L, H, tag=1)
    obstacle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)
    #fluid = rect - obstacle
    fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle)]) 
    gmsh.model.occ.synchronize()

#**** Creation of the tags of each entites of the domain (surface = fluid, edges=BC)
fluid_marker = 1
inlet_marker = 2
outlet_marker = 3
side_marker = 4
obstacle_marker = 5

# tag the fluid domain 
if rank == 0:
    volumes = gmsh.model.getEntities(dim=gdim)
    # Create physical groups for fluid (dim=2=volumes[0][0])
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
    gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")

# tag the boundaries
inflow, outflow, sides, obstacle = [], [], [], []
if rank == 0:
    boundaries = gmsh.model.getBoundary(volumes, oriented=False)
    for boundary in boundaries:
        #we use the getCenterOfMass function to locate the boundaries
        center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
        if np.allclose(center_of_mass, [0, H/2, 0]):
            inflow.append(boundary[1])
        elif np.allclose(center_of_mass, [L, H/2, 0]):
            outflow.append(boundary[1])
        elif np.allclose(center_of_mass, [L/2, H, 0]) or np.allclose(center_of_mass, [L/2, 0, 0]):
            sides.append(boundary[1])
        else:
            obstacle.append(boundary[1])

    # Create physical groups for boundaries (dim=1)
    gmsh.model.addPhysicalGroup(1, sides, side_marker)
    gmsh.model.setPhysicalName(1, side_marker, "Side")
    gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
    gmsh.model.setPhysicalName(1, inlet_marker, "Inlet")
    gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
    gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")
    gmsh.model.addPhysicalGroup(1, obstacle, obstacle_marker)
    gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")

# Create distance field from obstacle.
# Add threshold of mesh sizes based on the distance field
# LcMax -                  /--------
#                      /
# LcMin -o---------/
#        |         |       |
#       Point    DistMin DistMax
res_min = r / 4.
res_max = r /1.
if rank == 0:
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "EdgesList", obstacle)
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "IField", 1)
    gmsh.model.mesh.field.setNumber(2, "LcMin", res_min)
    gmsh.model.mesh.field.setNumber(2, "LcMax", res_max)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 1*r)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 8*r)

    # We take the minimum of the two fields as the mesh size
    gmsh.model.mesh.field.add("Min", 5)
    gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
    gmsh.model.mesh.field.setAsBackgroundMesh(5)

    # Generating the mesh
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 8)
    gmsh.option.setNumber("Mesh.RecombineAll", 2)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.optimize("Netgen")


# Finalize the mesh and write on disk
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(gdim)
gmsh.write("cylinder.msh")
gmsh.finalize()
