#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:30:48 2021

@author: florianma

Simulation gets slower and slower. 1st iteration takes 2s, 10th iteration 30s.
"""
from dolfin import (FacetNormal, assemble, div, dot, ds, dx, grad, inner, lhs,
                    nabla_grad, rhs, solve)


class ImplicitTentativeVelocityStep():
    def __init__(self, domain):
        rho, mu, dt = domain.rho, domain.mu, domain.dt
        u, u_1, vu = domain.u, domain.u_1, domain.vu
        p_1 = domain.p_1
        n = FacetNormal(domain.mesh)
        acceleration = rho * inner((u - u_1) / dt, vu) * dx
        pressure = inner(p_1, div(vu)) * dx - dot(p_1 * n, vu) * ds
        diffusion = (-inner(mu * (grad(u) + grad(u).T), grad(vu)) * dx
                     + dot(mu * (grad(u) + grad(u).T) * n, vu) * ds)
        convection = rho*dot(dot(u_1, nabla_grad(u)), vu) * dx
        # needs to be reassembled!
        F_impl = -acceleration - convection + diffusion + pressure
        self.a, self.L = lhs(F_impl), rhs(F_impl)
        self.domain = domain
        self.A = assemble(self.a)
        [bc.apply(self.A) for bc in domain.bcu]
        return

    def solve(self, reassemble_A=True):
        bcu = self.domain.bcu
        u_ = self.domain.u_
        self.A = assemble(self.a)
        [bc.apply(self.A) for bc in self.domain.bcu]
        b = assemble(self.L)
        [bc.apply(b) for bc in bcu]
        solve(self.A, u_.vector(), b, 'bicgstab', 'hypre_amg')
        return


class ExplicitTentativeVelocityStep():
    def __init__(self, domain):
        rho, mu, dt = domain.rho, domain.mu, domain.dt
        u, u_1, p_1, vu = domain.u, domain.u_1, domain.p_1, domain.vu
        n = FacetNormal(domain.mesh)
        acceleration = rho * inner((u - u_1) / dt, vu) * dx
        diffusion = (-inner(mu * (grad(u_1) + grad(u_1).T), grad(vu)) * dx
                     + dot(mu * (grad(u_1) + grad(u_1).T) * n, vu) * ds)
        pressure = inner(p_1, div(vu)) * dx - dot(p_1 * n, vu) * ds
        convection = rho * dot(dot(u_1, nabla_grad(u_1)), vu) * dx
        F_impl = -acceleration - convection + diffusion + pressure
        self.a, self.L = lhs(F_impl), rhs(F_impl)
        self.domain = domain
        self.A = assemble(self.a)
        [bc.apply(self.A) for bc in domain.bcu]
        return

    def solve(self, reassemble_A=False):
        bcu = self.domain.bcu
        u_ = self.domain.u_
        if reassemble_A:
            self.A = assemble(self.a)
            [bc.apply(self.A) for bc in bcu]
        b = assemble(self.L)
        [bc.apply(b) for bc in bcu]
        solve(self.A, u_.vector(), b, 'bicgstab', 'hypre_amg')
        return


class PressureStep():
    def __init__(self, domain):
        rho, dt = domain.rho, domain.dt
        p, p_1, vp = domain.p, domain.p_1, domain.vp
        p_1, u_ = domain.p_1, domain.u_
        self.a = dot(grad(p), grad(vp)) * dx
        self.L = (dot(grad(p_1), grad(vp)) * dx
                  - (rho / dt) * div(u_) * vp * dx)
        self.A = assemble(self.a)
        [bc.apply(self.A) for bc in domain.bcp]
        self.domain = domain
        return

    def solve(self):
        bcp = self.domain.bcp
        p_ = self.domain.p_
        b = assemble(self.L)
        [bc.apply(b) for bc in bcp]
        solve(self.A, p_.vector(), b, 'bicgstab', 'hypre_amg')
        return


class VelocityCorrectionStep():
    def __init__(self, domain):
        rho, dt = domain.rho, domain.dt
        u, u_, vu = domain.u, domain.u_, domain.vu
        p_1, p_ = domain.p_1, domain.p_
        self.a = dot(u, vu) * dx
        self.L = dot(u_, vu)*dx - dt / rho * dot(nabla_grad(p_ - p_1), vu)*dx
        self.A = assemble(self.a)
        [bc.apply(self.A) for bc in domain.bcu]
        self.domain = domain
        return

    def solve(self):
        bcu = self.domain.bcu
        u_ = self.domain.u_
        b = assemble(self.L)
        [bc.apply(b) for bc in bcu]
        solve(self.A, u_.vector(), b, 'bicgstab', 'hypre_amg')
        return
