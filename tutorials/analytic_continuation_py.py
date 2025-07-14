# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import matplotlib.pyplot as pl
import pylibsparseir as sparse_ir

# %%
beta = 40.0
wmax = 2.0
basis = sparse_ir.FiniteTempBasis('F', beta, wmax, eps=2e-8)

# %%
pl.semilogy(basis.s / basis.s[0], '-+')
pl.title(r'singular values $s_\ell/s_0$ of $K(\tau, \omega)$')
pl.xlabel(r'index $\ell$');


# %%
def semicirc_dos(w):
    return 2/np.pi * np.sqrt((np.abs(w) < wmax) * (1 - np.square(w/wmax)))

def insulator_dos(w):
    return semicirc_dos(8*w/wmax - 4) + semicirc_dos(8*w/wmax + 4)

# For testing, compute exact coefficients g_l for two models
rho1_l = basis.v.overlap(semicirc_dos)
rho2_l = basis.v.overlap(insulator_dos)
g1_l = -basis.s * rho1_l
g2_l = -basis.s * rho2_l

# Put some numerical noise on both of them (30% of basis accuracy)
rng = np.random.RandomState(4711)
noise = 0.3 * basis.s[-1] / basis.s[0]
g1_l_noisy = g1_l + rng.normal(0, noise, basis.size) * np.linalg.norm(g1_l)
g2_l_noisy = g2_l + rng.normal(0, noise, basis.size) * np.linalg.norm(g2_l)

# %%
