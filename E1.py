#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:47:46 2021

@author: florianma
"""
import dolfin as df
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
inner, dot = df.inner, df.dot
grad, dx = df.grad, df.dx

mu = 1.23
mesh = df.UnitSquareMesh(50, 50)
SUPG_stabilization = False
SUPG_stabilization = True


def left(x, on_boundary):
    return (x[0] < 1e-6) and on_boundary


def right(x, on_boundary):
    return (abs(x[0]-1.0) < 1e-6) and on_boundary


def func(h, C, alpha):
    return C * h**alpha


Ns = np.array([8, 16, 32, 64, 128, 256])
hs = 1/Ns
mus = np.array([1.0, .3, .1, .01])

L1 = np.empty((len(mus), len(hs)))
L2 = np.empty((len(mus), len(hs)))

for i, mu in enumerate(mus):
    B = 1/mu
    A = 1/(np.exp(B)-1)
    C = -A
    u_e = df.Expression(("A*exp(B*x[0]) + C"), A=A, B=B, C=C, degree=2)
    for j, N in enumerate(Ns):
        mesh = df.UnitSquareMesh(N, N)

        V = df.FunctionSpace(mesh, 'P', 1)
        v = df.TestFunction(V)
        u_ = df.Function(V)
        u = df.TrialFunction(V)

        if SUPG_stabilization:
            alpha = 2
            h = 1/N
            magnitude = 1
            Pe = magnitude * h / (2.0 * mu)
            tau = h / (2.0*magnitude) * (1.0/np.tanh(Pe) - 1.0/Pe)
            beta = df.Constant(tau*alpha)
            v = v + beta*h*v.dx(0)

        F = df.Constant(mu)*inner(grad(u), grad(v)) * dx + inner(u.dx(0), v)*dx

        bc1 = df.DirichletBC(V, df.Constant(0), left)
        bc2 = df.DirichletBC(V, df.Constant(1), right)
        # Neumann on y = 0 and y = 1 enforced implicitly
        df.solve(df.lhs(F) == df.rhs(F), u_, bcs=[bc1, bc2])

        # interpolate(u_e, VV)
        # u_numerical = u_.vector().vec().array
        # X = V.tabulate_dof_coordinates()
        X = mesh.coordinates()
        u_numerical = u_.compute_vertex_values(mesh)
        # u_analytical = A*np.exp(B*X[:, 0]) + C
        # u_analytical.shape = (N+1, N+1)
        # u_numerical.shape = (N+1, N+1)
        # e = (u_analytical-u_numerical)
        # e.shape = (N+1, N+1)
        # e = e[:, 1:-1]  # exclude Dirichtlet BC
        L1[i, j] = df.errornorm(u_, df.project(u_e, V), norm_type='H1')
        L2[i, j] = df.errornorm(u_, df.project(u_e, V), norm_type='l2')
        if N == 8:
            x = np.linspace(0, 1, 1000)
            u_analytical = A*np.exp(B*x) + C
            fig, ax = plt.subplots()
            # ax.plot(X[:, 0], u_numerical, color="r", marker="o", label="numerical solution")
            # u_numerical.shape = (9, 9)
            # ax.plot(X[:9, 0], u_numerical[:9], color="k", marker="o", label="numerical solution")
            ax.plot(X[-9:, 0], u_numerical[-9:], color="b", marker="o", label="numerical solution")
            ax.plot(x, u_analytical, "g-", label="analytical solution")
            plt.legend()
            ax.set_xlabel("x")
            ax.set_ylabel("u(x)")
            ax.set_title("1D solution of the PDE\n mu = {:.2f}".format(mu))
            plt.show()

r1f = np.empty((len(mus), len(hs)-1))
r2f = np.empty((len(mus), len(hs)-1))

for i, mu in enumerate(mus):
    for j in range(1, len(Ns)):
        r1f[i, j-1] = np.log(L1[i, j]/L1[i, j-1])/np.log(hs[j]/hs[j-1])
        r2f[i, j-1] = np.log(L2[i, j]/L2[i, j-1])/np.log(hs[j]/hs[j-1])


print("mu & C_alpha & alpha & C_beta & beta \\\\ [0.5ex] ")
print("\\hline\\hline")
for i, mu in enumerate(mus):
    [C_alpha, alpha], _ = curve_fit(func, hs, L1[i], p0=[4., 1.])
    # print("L1: ", C_alpha, alpha)
    [C_beta, beta], _ = curve_fit(func, hs, L2[i], p0=[1, 2.])
    # print("L2: ", C_beta, beta,  "(FEniCS)")
    print("{:.2f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ ".format(
        mu, C_alpha, alpha, C_beta, beta))
    print("\\hline")


#     # [C_alpha, alpha], _ = curve_fit(func, hs, L0[i], p0=[1, -2])
#     # print("L0: ", C_alpha, alpha)


#     # [C_beta2, beta2], _ = curve_fit(func, hs, L2[i], p0=[4., 1.])
#     # print("L2: ", C_beta2, beta2)

    # print(mu, C_alpha, alpha, C_beta, beta)

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(hs, L1[i], "ro", label="curve fit: C * h**alpha = {:.4f} * h**{:.4f}".format(C_alpha, alpha))
    ax1.plot(hs, func(hs, C_alpha, alpha), "g.", label="computed")
    ax2.plot(hs, L2[i], "ro", label="curve fit: C * h**alpha = {:.4f} * h**{:.4f}".format(C_beta, beta))
    ax2.plot(hs, func(hs, C_beta, beta), "g.", label="computed")
    ax1.legend()
    ax2.legend()
    ax1.set_xlabel("mesh size h")
    ax1.set_ylabel("H1 norm")
    ax2.set_xlabel("mesh size h")
    ax2.set_ylabel("L2 norm")
    plt.suptitle("mu = {:.2f}".format(mu))
    plt.show()


# fig, ax = plt.subplots()
# ax.plot(X[:, 0], u_numerical, "go", label="numerical solution")
# ax.plot(X[:, 0], u_analytical, "r.", label="analytical solution")
# # ax.set_aspect("equal")
# plt.legend()
# ax.set_xlabel("x")
# ax.set_ylabel("u(x)")
# ax.set_title("1D solution of the PDE")
# plt.show()

# fig, ax = plt.subplots()
# plt.plot(X[:, 1], u_numerical, "go")
# plt.plot(X[:, 1], u_analytical, "r.")
# plt.show()

# fig, ax = plt.subplots()
# df.plot(mesh)
# plt.show()
