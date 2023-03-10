import numpy as np
import matplotlib.pyplot as plt
import graphviz
from graphviz import Digraph
from scipy.special import comb, perm
import os
import sys
from itertools import combinations, permutations
from collections import deque
from functools import reduce
import copy
from ete3 import Tree
import re
import random
from pathlib import Path
from scipy.stats import norm
from itertools import combinations
import copy
import json
import os


### functions for selecting the gene markers
class MetropolisHasting:
    '''
        gene_list: a list of total sequenced genes
        tree_list: a list of dict for tree structure
        node_list: a list of dict for tree node - mutation_list
        clonal_freq_list: a list of dicts of the frequency for each node
        tree_freq_list: a list of the frequencies of each possible tree
        n_markers: number for the gene markers
    '''
    def __init__(self, gene_list, n_markers, tree_list, node_list, clonal_freq_list, tree_freq_list, gene2idx):
        self.gene_list = gene_list
        self.n_markers = n_markers
        self.n_genes = len(self.gene_list)
        self.tree_list = tree_list
        self.n_trees = len(self.tree_list)
        self.gene2idx = gene2idx
        self.idx2gene = {value: key for key, value in self.gene2idx.items()}
        self.node_list = []
        for node_dict in node_list:
            temp = {}
            for key, values in node_dict.items():
                temp.setdefault(int(key), values)
            self.node_list.append(temp)
        self.clonal_freq_list = clonal_freq_list
        self.tree_freq_list = tree_freq_list
        self.mut2node_list = self.genes_clones_mapping(self.node_list)
        self.a2d_list = self.ancestor2descendant_mat(self.tree_list)
        self.entropy_distribution = calculate_pairwise_uncertainty(self.tree_list, self.node_list, self.gene2idx, 'entropy')
        self.memory = {tree_idx: np.full((self.n_genes, self.n_genes, 3), np.inf) for tree_idx in range(self.n_trees)}
        #self.memory = {tree_idx: np.full((len(self.node_list[tree_idx].keys()), len(self.node_list[tree_idx].keys()), 3), np.inf) for tree_idx in range(self.n_trees)}
        self.ll_list = []
        self.selected_indices_list = []
        print('dict', self.__dict__)

    def initialization(self, method=None):
        assert method in [None, 'heurstic']
        if method is None:
            self.selected_array = np.zeros((self.n_genes))
            self.selected_indices = np.random.choice(self.n_genes, self.n_markers, replace=False)
            self.selected_array[self.selected_indices] = 1
    #         else:
    #             self.selected

    def calculate_probability(self, indices, const):
        entropy_sub = self.entropy_distribution[indices, indices]
        entropy_mean = np.mean(entropy_sub, axis=0)
        unnorm_prob = np.zeros((self.n_genes))
        unnorm_prob[indices] = np.exp(entropy_mean / const)
        norm_prob = unnorm_prob / np.sum(unnorm_prob)
        return norm_prob

    def save_model(self, path="saved_model.json", overwrite=True):
        #         save_dict = {
        #             'gene_list': self.gene_list,
        #             'n_markers': self.n_markers,
        #             'n_genes': self.n_genes,
        #             'tree_list': self.tree_list,
        #             'n_trees': self.n_trees,
        #             'node_list': self.node_list,
        #             'clonal_freq_list': self.clonal_freq_list,
        #             'tree_freq_list': self.tree_freq_list,
        #             'll_list': self.ll_list,
        #             'annealing': self.annealing,
        #             'const_min': self.const_min,
        #             'const_max': self.const_max,
        #             'const': self.const,
        #             'read_depth': self.read_depth,
        #             'sample_freq': self.sample_freq,
        #             'test_size': self.test_size
        #         }
        save_dict = self.__dict__
        if not overwrite:
            if os.path.exists(path):
                path_sub = path.rsplit(".", 1)[0]
                path = path_sub + "_v2.json"
        with open(path, "w") as output_file:
            json.dump(save_dict, output_file)

    def load_model(self, path="saved_model.json"):
        with open(path, ) as input_file:
            model = json.load(input_file)
            for key, value in model.items():
                setattr(self, key, value)

    def sampling(self, n_iter=100, read_depth=10000, sample_freq=False, test_size=1000, annealing=False,
                 const_max=1, const_min=1, load_model=False, path="saved_model.json"):
        if not load_model:
            self.n_iter = n_iter
            self.read_depth = read_depth
            self.sample_freq = sample_freq
            self.test_size = test_size
            self.annealing = annealing
        if not self.annealing:
            assert const_max == const_min
            self.const = const_min
        else:
            self.const_min = const_min
            self.const_max = const_max
        if load_model:
            self.load_model(path)
        if len(self.ll_list) == 0:
            if self.annealing:
                self.const = self.const_max
            self.new_selected_indices = copy.deepcopy(self.selected_indices)
            self.log_prob = - self.calculate_trees_weighted_entropy(read_depth, sample_freq, test_size) / self.const
            self.ll_list.append(self.log_prob)
            self.selected_indices_list.append(self.selected_indices)
        for t in range(self.n_iter):
            # remove step
            if self.annealing:
                self.const = (self.const_min - self.const_max) / self.n_iter * (t+1) + self.const_max
            remove_prob = self.calculate_probability(self.selected_indices, self.const)
            remove_idx = np.random.choice(self.n_genes, 1, p=remove_prob)
            # add step
            unselected_indices = list(set(range(self.n_genes)) - set(self.selected_indices))
            add_prob = self.calculate_probability(unselected_indices, self.const)
            add_idx = np.random.choice(self.n_genes, 1, p=add_prob)
            # calculate the probability of forward
            prob_forward = remove_prob[remove_idx] * add_prob[add_idx]
            # make the move
            self.new_selected_indices = copy.deepcopy(self.selected_indices)
            self.new_selected_indices[self.new_selected_indices == remove_idx[0]] = add_idx[0]
            self.new_selected_indices = np.sort(self.new_selected_indices)
            unselected_indices = list(set(range(self.n_genes)) - set(self.new_selected_indices))
            # calculate the probability of backward
            remove_prob = self.calculate_probability(self.new_selected_indices, self.const)
            add_prob = self.calculate_probability(unselected_indices, self.const)
            prob_backward = remove_prob[add_idx] * add_prob[remove_idx]
            log_prob = - self.calculate_trees_weighted_entropy(read_depth, sample_freq, test_size) / self.const
            ratio = np.exp(self.log_prob - self.ll_list[-1]) * prob_backward / prob_forward
            u = np.random.uniform(0, 1)
            print([self.idx2gene[idx] for idx in self.new_selected_indices], log_prob)
            if u < min(1, ratio):
                self.selected_indices = self.new_selected_indices
                self.log_prob = log_prob

            if t % 1 == 0:
                self.best_iter = np.argmax(np.array(self.ll_list))
                self.best_ll = self.ll_list[self.best_iter]
                print('iteration:', t, 'best ll:', self.best_ll)
                #self.save_model(path=path)
            self.ll_list.append(self.log_prob)
            self.selected_indices_list.append(self.selected_indices)

        self.best_ll = np.max(self.ll_list)
        self.best_iter = np.where(self.ll_list == self.best_ll)[0]
        self.best_selected_indices = np.array(self.selected_indices_list)[self.best_iter, :]
        #self.save_model(path=path)
        self.final_best_selected_indices = []
        for idx in range(self.best_selected_indices.shape[0]):
            self.final_best_selected_indices.append(tuple([self.idx2gene[item] for item in sorted(self.best_selected_indices[idx])]))
        self.final_best_selected_indices = set(self.final_best_selected_indices)
        print('Summary: Best likelihood: ', self.best_ll)
        print('         Best selected gene markers: ', self.final_best_selected_indices)
        return self.best_iter, self.best_ll, self.best_selected_indices

    def adjust_tree_distribution(self, correct_tree_idx, read_depth=1000, sample_freq=False, test_size=1000):
        relation_list = ['same', 'ancestor', 'descendant']
        correct_tree = self.tree_list[correct_tree_idx]
        read_count_markers = []
        mean_count_markers = []
        clonal_freq = self.clonal_freq_list[correct_tree_idx]
        geneidx2markeridx = {}
        for i, marker in enumerate(self.new_selected_indices):
            geneidx2markeridx[marker] = i
            p = clonal_freq[str(self.mut2node_list[correct_tree_idx][self.idx2gene[marker]])]
            read_count = np.random.binomial(read_depth, p, test_size)
            read_count_markers.append(read_count)
            mean_count_markers.append(np.mean(read_count)/read_depth)
        # if sample_freq:
        #     clonal_freq = simulate_freq(correct_tree, k)
        rejected_tree_indices = []
        a2d_matrix_correct = self.tree_list[correct_tree_idx]
        mut2node_dict_correct = self.mut2node_list[correct_tree_idx]
        for idx in range(len(self.tree_list)):
            if idx != correct_tree_idx:
                tree = self.tree_list[idx]
                a2d_matrix = self.a2d_list[idx]
                mut2node_dict = self.mut2node_list[idx]
                for (marker1, marker2) in list(combinations(self.new_selected_indices, 2)):
                    node1, node2 = mut2node_dict[self.idx2gene[marker1]], mut2node_dict[self.idx2gene[marker2]]
                    node1_correct, node2_correct = mut2node_dict_correct[self.idx2gene[marker1]], mut2node_dict_correct[self.idx2gene[marker2]]
                    freq_hat_1, freq_hat_2 = mean_count_markers[geneidx2markeridx[marker1]], mean_count_markers[geneidx2markeridx[marker2]]
                    if node1 == node2:
                        relation = 'same'
                    elif a2d_matrix[node1, node2] == 1:
                        relation = 'ancestor'
                    elif a2d_matrix[node1, node2] == 0 and a2d_matrix[node2, node1] == 1:
                        relation = 'descendant'
                    else:
                        relation = 'null'
                    if relation == 'null':
                        continue
                    if self.memory[correct_tree_idx][marker1, marker2, relation_list.index(relation)] == np.inf:
                        # node_correct - 1 is due to the fact that the normal node 0 contains no mutations, thus being omitted
                        bool_reject, W, z = wald_test(freq_hat_1, freq_hat_2, self.n_markers * (self.n_markers - 1) / 2, relation, read_depth)
                        #bool_reject, W, z = wald_test(freq_hat_1, freq_hat_2, 1, relation, read_depth)
                        #print(bool_reject, W, z)
                        self.memory[correct_tree_idx][marker1, marker2, relation_list.index(relation)] = bool_reject
                    else:
                        bool_reject = self.memory[correct_tree_idx][marker1, marker2, relation_list.index(relation)]
                    if bool_reject:
                        rejected_tree_indices.append(idx)
        return rejected_tree_indices

    def calculate_trees_weighted_entropy(self, read_depth=100, sample_freq=False, test_size=1000):
        weighted_entropy = 0
        weights = np.array(self.tree_freq_list) #/ np.sum(np.array(self.tree_freq_list))
        for tree_idx in range(self.n_trees):
            entropy = self.calculate_single_tree_entropy(tree_idx, read_depth, sample_freq, test_size)
            weighted_entropy += weights[tree_idx] * entropy
        return weighted_entropy

    def calculate_single_tree_entropy(self, correct_tree_idx, read_depth=100, sample_freq=False, test_size=1000):
        rejected_tree_indices = self.adjust_tree_distribution(correct_tree_idx, read_depth, sample_freq, test_size)
        tree_freq = np.array(self.tree_freq_list)
        if len(rejected_tree_indices) != 0:
            tree_freq[rejected_tree_indices] = 0
        tree_freq = tree_freq[tree_freq != 0]
        tree_freq = (tree_freq).astype('float') / np.sum(tree_freq)
        entropy = - np.sum(tree_freq * np.log(tree_freq))
        return entropy

    def genes_clones_mapping(self, node_list):
        mut2node_list = []
        for tree_idx in range(self.n_trees):
            mut2node_list.append(mut2node(node_list[tree_idx]))
        return mut2node_list

    @staticmethod
    def boltzmann_factor(energy_1, energy_2, const=1):
        return np.exp((energy_2 - energy_1) / const)

    @staticmethod
    def ancestor2descendant_mat(tree_list):
        a2d_mat_list = []
        for tree_idx in range(len(tree_list)):
            tree = tree_list[tree_idx]
            a2d_matrix = ancestor2descendant(tree)
            a2d_mat_list.append(a2d_matrix)
        return a2d_mat_list


def calculate_concentration(tree_freq_list, method='sum_square'):
    '''
    For both methods, the more concentrated the tree distribution is, the value is closer to 0.
    The less concentrated, sum_square is closer to 1, the entropy is closer to infinity.
    '''
    freq_list_np = np.array(tree_freq_list)
    if method == 'sum_square':
        return 1 - np.sum(np.square(freq_list_np))
    elif method == 'entropy':
        return - np.sum(freq_list_np * np.log(freq_list_np))


def wald_test(freq_hat_1, freq_hat_2, correction_rate, relation='ancestor', depth=100, alpha=0.05):
    '''
    return True if reject
    '''
    #print(freq_hat_1, freq_hat_2)
    assert relation in ['ancestor', 'descendant', 'same']
    if relation == 'ancestor':
        W = (freq_hat_2 - freq_hat_1) / np.sqrt((freq_hat_1 * (1 - freq_hat_1) + freq_hat_2 * (1 - freq_hat_2)) / depth)
        z = norm.ppf(alpha / correction_rate)
    elif relation == 'descendant':
        W = (freq_hat_1 - freq_hat_2) / np.sqrt((freq_hat_1 * (1 - freq_hat_1) + freq_hat_2 * (1 - freq_hat_2)) / depth)
        z = norm.ppf(alpha / correction_rate)
    elif relation == 'same':
        W = np.abs((freq_hat_1 - freq_hat_2) / np.sqrt(
            (freq_hat_1 * (1 - freq_hat_1) + freq_hat_2 * (1 - freq_hat_2)) / depth))
        z = norm.ppf(alpha / correction_rate / 2)
    #print(W)
    if W > - z:
        return True, W, - z
    else:
        return False, W, z


def simulate_freq(tree, k, alpha=0.3, beta=0.3):
    ### need to rewrite
    freq_true = np.random.beta(alpha, beta, k)
    freq_obs = np.random.dirichlet(freq_true)
    freq_sum = {}
    bfs_order = bfs_structure(tree)
    for node_idx in range(len(bfs_order) - 1, -1, -1):
        node = bfs_order[node_idx]
        if node not in simulate_tree_template.keys():
            freq_sum[node] = freq_obs[node]
        else:
            freq_sum[node] = sum([freq_sum[child_node] for child_node in tree[node]]) + freq_obs[node]
    return freq_sum


# test block
def calculate_tree_entropy(tree_freq_list, rejected_tree_indices):
    tree_freq = np.array(tree_freq_list)
    if len(rejected_tree_indices) != 0:
        tree_freq[rejected_tree_indices] = 0
    tree_freq = tree_freq[tree_freq != 0]
    tree_freq = (tree_freq).astype('float')/np.sum(tree_freq)
    entropy = - np.sum(tree_freq * np.log(tree_freq))
    return entropy


def calculate_square_sum(tree_freq_list, rejected_tree_indices):
    tree_freq = np.array(tree_freq_list)
    if len(rejected_tree_indices) != 0:
        tree_freq[rejected_tree_indices] = 0
    tree_freq = tree_freq[tree_freq != 0]
    tree_freq = (tree_freq).astype('float')/np.sum(tree_freq)
    sq_sum = 1 - np.sum(np.square(tree_freq))
    return sq_sum


# utility functions

def mut2node(node_dict):
    mut2node_dict = {}
    for node, mut_list in node_dict.items():
        for mut in mut_list:
            mut2node_dict[mut] = int(node)
    return mut2node_dict


def bfs_structure(tree):  # O(k)
    order = []
    root = find_root(tree)
    q = deque([root])
    while len(q) != 0:
        node = q.popleft()
        order.append(node)
        if node in tree.keys():
            for child in tree[node]:
                q.append(child)
    return order

def bfs(root, tree):  #O(k)
    order = []
    q = deque([root])
    while len(q) != 0:
        node = q.popleft()
        order.append(node)
        if node in tree.keys():
            for child in tree[node]:
                q.append(child)
    return order

def find_root(tree):
    non_root = []
    for item in tree.values():
        non_root += list(item)
    for node in tree.keys():
        if node not in non_root:
            return node


def ancestor2descendant(tree):
    order = bfs_structure(tree)
    a2d = np.zeros((len(order), len(order)))
    for node in order[::-1]:
        if node in tree.keys():
            for child in tree[node]:
                a2d[int(node)][int(child)] = 1
                a2d[int(node)] += a2d[int(child)]
    return a2d


def generate_cp(tree):
    return {c: p for p in tree.keys() for c in tree[p]}  # child: parent


def generate_tree(cp_tree):
    tree = {}
    for child, parent in cp_tree.items():
        if parent in tree.keys():
            tree[parent].append(child)
        else:
            tree[parent] = [child]
    return tree


def root_searching(tree):  # O(depth of tree) <= O(k)
    tree_cp = generate_cp(tree)
    start_node = list(tree_cp.keys())[0]
    iter_count = 0
    while True:
        iter_count += 1
        start_node = tree_cp[start_node]
        if start_node not in tree_cp.keys():
            break
        if iter_count >= 100:
            print("The directed tree exists self-loop.")
            return None
    return start_node


### count ancestor-descendant relationships of all pairs of mutations
def find_all_ancestors(tree, node_dict):
    root = root_searching(tree)
    cp_tree = generate_cp(tree)
    order = bfs(root, tree)
    ancestors_dict = {}
    ancestors_node_dict = {}
    for node in order:
        if node is root:
            ancestors_node_dict.setdefault(root, [])
            continue
        parent = cp_tree[node]
        ancestors_node_dict.setdefault(node, ancestors_node_dict[parent] + [parent])  # inherit the ancestors of the parent
        mut_anc = []
        for n in ancestors_node_dict[node]:
            if n != root:
                mut_anc += node_dict[n]
        for mut in node_dict[node]:
            ancestors_dict.setdefault(mut, mut_anc)
    return ancestors_dict, ancestors_node_dict


def create_ancestor_descendant_matrix(tree, node_dict, gene2idx):
    ancestors_dict, ancestors_node_dict = find_all_ancestors(tree, node_dict)
    num_muts = len(ancestors_dict.keys())
    anc_des_matrix = np.zeros((num_muts, num_muts))
    for mut, ancestors in ancestors_dict.items():
        if len(ancestors) >= 1:
            index = (np.array([gene2idx[anc] for anc in ancestors]), np.repeat(gene2idx[mut], len(ancestors)))
            anc_des_matrix[index] += 1
    return anc_des_matrix


def create_same_clone_matrix(tree, node_dict, gene2idx):
    root = root_searching(tree)
    order = bfs(root, tree)
    sam_clo_matrix = np.zeros((len(gene2idx.keys()), len(gene2idx.keys())))

    for node in order:
        if node != root:
            muts = [gene2idx[mut] for mut in node_dict[node]]
            if len(muts) >= 2:
                indices = list(permutations(muts, r=2))
                for idx in indices:
                    sam_clo_matrix[idx] = 1
    return sam_clo_matrix


def calculate_entropy(matrix):
    return -np.sum(matrix * np.log(matrix, out=np.zeros_like(matrix), where=(matrix != 0)), axis=0)


def calculate_pairwise_uncertainty(tree_list, node_list, gene2idx, method='entropy', tree_freq_list=None):
    num_tree = len(tree_list)
    if tree_freq_list is None:
        tree_freq_list = np.ones(num_tree)
    sum_tree_freq = np.sum(tree_freq_list)
    for i in range(num_tree):
        tree = tree_list[i]
        node_dict = node_list[i]
        anc_des_matrix = create_ancestor_descendant_matrix(tree, node_dict, gene2idx)
        sam_clo_matrix = create_same_clone_matrix(tree, node_dict, gene2idx)
        if i == 0:
            anc_des_matrix_sum = np.zeros(anc_des_matrix.shape)
            sam_clo_matrix_sum = np.zeros(anc_des_matrix.shape)
        anc_des_matrix_sum += anc_des_matrix * tree_freq_list[i]
        sam_clo_matrix_sum += sam_clo_matrix * tree_freq_list[i]
    des_anc_matrix_sum = anc_des_matrix_sum.T
    no_rel_matrix_sum = np.ones(
        anc_des_matrix.shape) * num_tree - anc_des_matrix_sum - des_anc_matrix_sum - sam_clo_matrix_sum
    full_matrix_sum = np.concatenate((anc_des_matrix_sum[np.newaxis, :] / sum_tree_freq,
                                      des_anc_matrix_sum[np.newaxis, :] / sum_tree_freq,
                                      sam_clo_matrix_sum[np.newaxis, :] / sum_tree_freq,
                                      no_rel_matrix_sum[np.newaxis, :] / sum_tree_freq), axis=0)
    if method == 'entropy':
        return calculate_entropy(full_matrix_sum)