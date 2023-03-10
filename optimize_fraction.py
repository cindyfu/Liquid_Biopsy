import numpy as np
from optimize import create_same_clone_matrix
import gurobipy as gp

### use gurobi

def create_sum_same_clone(tree_list, node_list, gene2idx, tree_freq_list=None):
    num_tree = len(tree_list)
    if tree_freq_list is None:
        tree_freq_list = np.ones(num_tree)
    sum_tree_freq = np.sum(tree_freq_list)
    for i in range(num_tree):
        tree = tree_list[i]
        node_dict = node_list[i]
        sam_clo_matrix = create_same_clone_matrix(tree, node_dict, gene2idx)
        #sam_clo_matrix /= (np.sum(sam_clo_matrix, axis=0) + 1) # normalization
        if i == 0:
            sam_clo_matrix_sum = np.zeros(sam_clo_matrix.shape)
        sam_clo_matrix_sum += sam_clo_matrix * tree_freq_list[i]
    return sam_clo_matrix_sum


def get_gp_1d_arr_bin_var(model, m):
    X = np.empty((m), dtype=gp.Var)
    for i in range(0, m):
        X[i] = model.addVar(vtype=gp.GRB.BINARY)
    # mod.update()
    return X

def return_value(x):
    n = x.shape[0]
    z = np.empty(n)
    for i in range(n):
        z[i] = x[i].X
    return z


def optimize_fraction(S, n_genes, n_markers):
    model = gp.Model('opt_frac')
    z = get_gp_1d_arr_bin_var(model, n_genes)
    model.addConstr(gp.quicksum([z[i] for i in range(n_genes)]) == n_markers, name='n_marker constraint')
    model.setObjective(gp.quicksum(z[i] * S[i,j] * z[j] for i in range(n_genes) for j in range(n_genes)), gp.GRB.MINIMIZE)
    model.optimize()
    return return_value(z)


def select_markers_fractions_gp(gene_list, n_markers, tree_list, node_list, gene2idx, tree_freq_list):
    S = create_sum_same_clone(tree_list, node_list, gene2idx, tree_freq_list)
    print(S)
    n_genes = len(gene_list)
    best_z = optimize_fraction(np.tril(S), n_genes, n_markers)
    #best_z = optimize_fraction(S, n_genes, n_markers)
    selected_markers = []
    for idx in range(len(best_z)):
        if best_z[idx] == 1:
            selected_markers.append(gene_list[idx])
    return selected_markers

### try to use cvxpy to solve but failed
### keep getting errors:
### cvxpy.error.DCPError: Problem does not follow DCP rules. Specifically:
### The objective is not DCP, even though each sub-expression is.
### You are trying to minimize a function that is concave.

import cvxpy as cp


def solve_opt_frac(S, n_genes, n_markers):
    ones = np.ones(n_genes)
    A = np.random.random(S.shape)
    z = cp.Variable(n_genes,)
    prob = cp.Problem(cp.Minimize(cp.quad_form(z, S)),[cp.sum(z) == n_markers])
    prob.solve()
    return z.value


def select_markers_fractions_cxv(gene_list, n_markers, tree_list, node_list, gene2idx, tree_freq_list):
    S = create_sum_same_clone(tree_list, node_list, gene2idx, tree_freq_list)
    rand_perturb = np.random.uniform(0, 10**(-5), size= S.shape)
    S += rand_perturb + np.transpose(rand_perturb)
    print(S)
    n_genes = len(gene_list)
    best_z = solve_opt_frac(S, n_genes, n_markers)
    selected_markers = []
    for idx in range(len(best_z)):
        if best_z[idx] == 1:
            selected_markers.append(gene_list[idx])
    return selected_markers


