#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:51:13 2021

@author: florianma
"""

import dolfin as df
import matplotlib.pyplot as plt
import numpy as np
import pygmsh
import meshio
from mpi4py import MPI


def create_entity_mesh(mesh, cell_type, prune_z=False,
                       remove_unused_points=False):
    """
    Given a meshio mesh, extract mesh and physical markers for a given entity.
    We assume that all unused points are at the end of the mesh.points
    (this happens when we use physical markers with pygmsh)
    """
    cells = mesh.get_cells_type(cell_type)
    try:
        # If mesh created with gmsh API it is simple to extract entity data
        cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    except KeyError:
        # If mehs created with pygmsh, we need to parse through cell sets and sort the data
        cell_entities = []
        cell_data = []
        cell_sets = mesh.cell_sets_dict
        for marker, set in cell_sets.items():
            for type, entities in set.items():
                if type == cell_type:
                    cell_entities.append(entities)
                    cell_data.append(np.full(len(entities), int(marker)))
        cell_entities = np.hstack(cell_entities)
        sorted = np.argsort(cell_entities)
        cell_data = np.hstack(cell_data)[sorted]
    if remove_unused_points:
        num_vertices = len(np.unique(cells.reshape(-1)))
        # We assume that the mesh has been created with physical tags,
        # then unused points are at the end of the array
        points = mesh.points[:num_vertices]
    else:
        points = mesh.points

    # Create output mesh
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells},
                           cell_data={"name_to_read": [cell_data]})
    if prune_z:
        out_mesh.prune_z_0()
    return out_mesh


def create_channel_mesh(lcar):
    with pygmsh.geo.Geometry() as geom:
        r = .05
        p = [geom.add_point([.20, .20], lcar),
             geom.add_point([0.0, .0], lcar),
             geom.add_point([2.2, .0], lcar),
             geom.add_point([2.2, .41], lcar),
             geom.add_point([0.0, .41], lcar),
             geom.add_point([.2 + r, .20], lcar),
             geom.add_point([.20, .2 + r], lcar),
             geom.add_point([.2 - r, .20], lcar),
             geom.add_point([.20, .2 - r], lcar)]
        c = [geom.add_line(p[1], p[2]),  # btm
             geom.add_line(p[2], p[3]),
             geom.add_line(p[3], p[4]),  # top
             geom.add_line(p[4], p[1]),
             geom.add_circle_arc(p[5], p[0], p[6]),
             geom.add_circle_arc(p[6], p[0], p[7]),
             geom.add_circle_arc(p[7], p[0], p[8]),
             geom.add_circle_arc(p[8], p[0], p[5])]
        ll1 = geom.add_curve_loop([c[0], c[1], c[2], c[3]])
        ll2 = geom.add_curve_loop([c[4], c[5], c[6], c[7]])
        s = [geom.add_plane_surface(ll1, [ll2])]
        fluid = geom.add_surface_loop(s)

        # Add physical markers to gmsh
        # Triangles:
        # - 0: Fluid
        # Lines:
        # - 1: Top and bottom wall
        # - 2: Cylinder wall
        # - 3: Inlet
        # - 4: Outlet
        top_and_bottom = [c[0], c[2]]
        cylinderwall = c[4:]
        inlet = [c[3]]
        outlet = [c[1]]
        geom.add_physical(fluid, "0")
        geom.add_physical(top_and_bottom, "1")
        geom.add_physical(cylinderwall, "2")
        geom.add_physical(inlet, "3")
        geom.add_physical(outlet, "4")
        # When using physical markers, unused points are placed last in the mesh
        msh = geom.generate_mesh(dim=2)

    # Write mesh to XDMF which we can easily read into dolfin
    if MPI.COMM_WORLD.rank == 0:
        #  Pick one of ['abaqus', 'ansys', 'avsucd', 'cgns', 'dolfin-xml',
        # 'exodus', 'flac3d', 'gmsh', 'gmsh22', 'h5m', 'hmf', 'mdpa', 'med',
        # 'medit', 'nastran', 'neuroglancer', 'obj', 'off', 'permas', 'ply',
        # 'stl', 'su2', 'svg', 'tecplot', 'tetgen', 'ugrid', 'vtk', 'vtu',
        # 'wkt', 'xdmf'], xdmf fails
        input_mesh = create_entity_mesh(msh, "triangle", True, True)
        meshio.write("mesh.xdmf", input_mesh, file_format="xdmf")
        # meshio.write("mesh.xdmf", input_mesh)
        meshio.write("mf.xdmf", create_entity_mesh(msh, "line", True),
                     file_format="xdmf")
    MPI.COMM_WORLD.barrier()


class SimpleMesh():
    """
    Simple representation of mesh as points (geometry) and connecitivities (topology)
    Used for
    """

    def __init__(self, dolfin_mesh):
        self.points = dolfin_mesh.coordinates()
        self.simplices = dolfin_mesh.cells()


def plot(mesh):
    """
    2D plot of mesh
    """
    from scipy.spatial import delaunay_plot_2d
    fig = delaunay_plot_2d(SimpleMesh(mesh))
    ax = fig.gca()
    ax.set_aspect("equal")
    return fig, ax


class ChannelProblemSetup():
    def __init__(self, parameters, mesh_name, facet_name,
                 bc_dict={"obstacle": 2, "channel_walls": 1, "inlet": 3,
                          "outlet": 4}):
        """
        Create the required function spaces, functions and boundary conditions
        for a channel flow problem
        """
        self.bc_dict = bc_dict
        self.mesh = df.Mesh()
        with df.XDMFFile(mesh_name) as infile:
            infile.read(self.mesh)

        mvc = df.MeshValueCollection("size_t", self.mesh,
                                     self.mesh.topology().dim() - 1)
        with df.XDMFFile(facet_name) as infile:
            infile.read(mvc, "name_to_read")
        self.mf = mf = df.cpp.mesh.MeshFunctionSizet(self.mesh, mvc)

        self.V = V = df.VectorFunctionSpace(self.mesh, 'P',
                                            parameters["degree velocity"])
        self.Q = Q = df.FunctionSpace(self.mesh, 'P',
                                      parameters["degree pressure"])
        self.rho = df.Constant(parameters["density [kg/m3]"])
        self.mu = df.Constant(parameters["viscosity [Pa*s]"])
        self.dt = df.Constant(parameters["dt [s]"])
        self.g = df.Constant((0, 0))
        self.vu, self.vp = df.TestFunction(V), df.TestFunction(Q)
        self.u_, self.p_ = df.Function(V), df.Function(Q)
        self.u_1, self.p_1 = df.Function(V), df.Function(Q)
        self.u_k, self.p_k = df.Function(V), df.Function(Q)
        self.u, self.p = df.TrialFunction(V), df.TrialFunction(Q)  # unknown!

        self.U_m = U_m = parameters["velocity [m/s]"]
        x = [0, .41 / 2]  # center of the channel
        Ucenter = 4.*U_m*x[1]*(.41-x[1])/(.41*.41)
        U0_str = "4.*U_m*x[1]*(.41-x[1])/(.41*.41)"
        self.U_mean = np.mean(2 / 3 * Ucenter)

        U0 = df.Expression((U0_str, "0"), U_m=U_m, degree=2)
        bc0 = df.DirichletBC(V, df.Constant((0, 0)), mf, bc_dict["obstacle"])
        bc1 = df.DirichletBC(V, df.Constant((0, 0)), mf, bc_dict["channel_walls"])
        bc2 = df.DirichletBC(V, U0, mf, bc_dict["inlet"])
        bc3 = df.DirichletBC(Q, df.Constant(0), mf, bc_dict["outlet"])
        self.bcu = [bc0, bc1, bc2]
        self.bcp = [bc3]
        self.ds_ = df.Measure("ds", domain=self.mesh, subdomain_data=mf)
        return

    def stokes(self):
        P2 = df.VectorElement("CG", self.mesh.ufl_cell(), 2)
        P1 = df.FiniteElement("CG", self.mesh.ufl_cell(), 1)
        TH = P2 * P1
        VQ = df.FunctionSpace(self.mesh, TH)
        mf = self.mf
        self.no_slip = df.Constant((0., 0))
        U0_str = "4.*U_m*x[1]*(.41-x[1])/(.41*.41)"
        U0 = df.Expression((U0_str, "0"), U_m=self.U_m, degree=2)
        bc0 = df.DirichletBC(VQ.sub(0), df.Constant((0, 0)), mf,
                             self.bc_dict["obstacle"])
        bc1 = df.DirichletBC(VQ.sub(0), df.Constant((0, 0)), mf,
                             self.bc_dict["channel_walls"])
        bc2 = df.DirichletBC(VQ.sub(0), U0, mf, self.bc_dict["inlet"])
        bc3 = df.DirichletBC(VQ.sub(1), df.Constant(0), mf,
                             self.bc_dict["outlet"])
        bcs = [bc0, bc1, bc2, bc3]

        vup = df.TestFunction(VQ)
        up = df.TrialFunction(VQ)
        up_ = df.Function(VQ)

        u, p = df.split(up)  # Trial
        vu, vp = df.split(vup)  # Test
        u_, p_ = df.split(up_)  # Function holding the solution
        inner, grad, dx, div = df.inner, df.grad, df.dx, df.div
        F = self.mu*inner(grad(vu), grad(u))*dx - inner(div(vu), p)*dx \
            - inner(vp, div(u))*dx
        df.solve(df.lhs(F) == df.rhs(F), up_, bcs=bcs)
        self.u_.assign(df.project(u_, self.V))
        self.p_.assign(df.project(p_, self.Q))
        return

    def normal_stresses(self):
        k = self.bc_dict["obstacle"]
        n = df.FacetNormal(self.mesh)
        tau = (self.mu*(df.grad(self.u_) + df.grad(self.u_).T)
               - self.p_*df.Identity(2))
        normal_stresses_x = -df.assemble(df.dot(tau, n)[0] * self.ds_(k))
        normal_stresses_y = df.assemble(df.dot(tau, n)[1] * self.ds_(k))
        return np.array([normal_stresses_x, normal_stresses_y])

    def plot(self):
        u, p = self.u_, self.p_
        mesh = self.mesh

        velocity = u.compute_vertex_values(mesh)
        velocity.shape = (2, -1)
        magnitude = np.linalg.norm(velocity, axis=0)
        x, y = mesh.coordinates().T
        u, v = velocity
        tri = mesh.cells()
        pressure = p.compute_vertex_values(mesh)
        # print(x.shape, y.shape, u.shape, v.shape)

        fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True,
                                       figsize=(12, 6))
        ax1.quiver(x, y, u, v, magnitude)
        ax2.tricontourf(x, y, tri, pressure, levels=40)
        ax1.set_aspect("equal")
        ax2.set_aspect("equal")
        ax1.set_title("velocity")
        ax2.set_title("pressure")
        return fig, (ax1, ax2)


if __name__ == "__main__":
    create_channel_mesh(lcar=0.02)
