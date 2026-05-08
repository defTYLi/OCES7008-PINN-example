# OCES7008 PINN Example

This repository contains a minimal PyTorch implementation of a physics-informed neural network (PINN) for a two-dimensional Poisson problem.

The example is used as an implementation aid for the OCES7008 final review report on physics-informed neural networks for ocean circulation and flow modeling.

## Problem

The PINN solves

$$
-(\psi_{xx}+\psi_{yy}) = f(x,y)
$$

on a unit square with homogeneous boundary condition

$$
\psi = 0
$$

on the boundary.

The reference solution is

$$
\psi_{\mathrm{true}}(x,y)=\sin(\pi x)\sin(\pi y).
$$

## Requirements

```bash
pip install torch
