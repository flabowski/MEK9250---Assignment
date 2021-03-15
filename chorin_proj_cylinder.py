#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:47:33 2021

@author: florianma
"""
import argparse
from tqdm import trange  # Progress bar
import matplotlib.pyplot as plt
from cylinder import create_channel_mesh, ChannelProblemSetup, plot
from chorins_projection import (ImplicitTentativeVelocityStep,
                                ExplicitTentativeVelocityStep, PressureStep,
                                VelocityCorrectionStep)


def test(args):
    # all the IO and printing happens here
    print(args.d_velocity, args.d_pressure)
    my_parameters = {"density [kg/m3]": 1.0,
                     "viscosity [Pa*s]": 1e-3,
                     "characteristic length [m]": .1,
                     "velocity [m/s]": 1.5,
                     "dt [s]": 0.1
                     }
    my_parameters["degree velocity"] = int(args.d_velocity)
    my_parameters["degree pressure"] = int(args.d_pressure)
    create_channel_mesh(lcar=0.01)
    my_domain = ChannelProblemSetup(my_parameters, "mesh.xdmf", "mf.xdmf")

    cfl = .1
    dt = cfl*my_domain.mesh.hmin()/my_domain.U_mean
    my_parameters["dt [s]"] = dt
    my_domain.dt.assign(dt)

    ps = PressureStep(my_domain)
    vcs = VelocityCorrectionStep(my_domain)
    if int(args.explicit) == 1:
        print("explicit scheme")
        tvs = ExplicitTentativeVelocityStep(my_domain)
    else:
        print("semi-implicit scheme")
        tvs = ImplicitTentativeVelocityStep(my_domain)

    my_domain.stokes()
    my_domain.u_1.assign(my_domain.u_)
    my_domain.p_1.assign(my_domain.p_)

    rho = my_parameters["density [kg/m3]"]
    U = my_domain.U_mean
    L = my_parameters["characteristic length [m]"]
    mu = my_parameters["viscosity [Pa*s]"]
    Re = rho*U*L/mu
    print("Re = ", Re)
    print("rho = ", rho)
    print("mu = ", mu)
    print("dt = ", dt)
    C_d = 0.
    C_l = 0.
    for n in trange(8000):
        tvs.solve(reassemble_A=True)
        ps.solve()
        vcs.solve()

        my_domain.u_1.assign(my_domain.u_)
        my_domain.p_1.assign(my_domain.p_)
        normal_stresses = my_domain.normal_stresses()
        C = 2*normal_stresses/(rho*U**2*L)
        if abs(C[0]) > C_d:
            C_d = abs(C[0])
        if abs(C[1]) > C_l:
            C_l = abs(C[1])
        if (n % 100) == 0:
            print("max C_d, max C_l", C_d, C_l)
            fig, ax = my_domain.plot()
            plt.savefig("tst.png")
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--d_velocity', help='description for option1')
    parser.add_argument('--d_pressure', help='description for option2')
    parser.add_argument('--explicit', help='description for option3')

    args = parser.parse_args()
    test(args)
    # test()
