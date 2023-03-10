from optimize import *
from optimize_fraction import *
import pandas as pd
from zipfile import ZipFile
import json
import gzip


directory=Path('MOONSHOT2')
# apply the real data to method
# step1. generate the input format
'''     gene_list: a list of total sequenced genes
        tree_list: a list of dict for tree structure
        node_list: a list of dict for tree node - mutation_list
        clonal_freq_list: a list of dicts of the frequency for each node
        tree_freq_list: a list of the frequencies of each possible tree
        n_markers: number for the gene markers
'''
gene_list, tree_list, node_list, clonal_freq_list, tree_freq_list = [], [], [], [], []


for file in directory.glob('*.xlsx'):
    patient_num = int(file.stem.split('_')[1])
    for bootstrap_num in range(1, 51, 1):
        summ_file = f"MOONSHOT2/{patient_num}/bootstrap{bootstrap_num}/result.summ.json.gz"
        muts_file = f"MOONSHOT2/{patient_num}/bootstrap{bootstrap_num}/result.muts.json.gz"
        with gzip.open(summ_file, "r") as f:
            # with open(summ_file, "r") as f:
            j_summ = json.loads(f.read().decode('utf-8'))
            best_tree = None
            best_tree_llh = -np.infty
            for tree in j_summ['trees'].keys():
                if j_summ['trees'][tree]['llh'] > best_tree_llh:
                    best_tree_llh = j_summ['trees'][tree]['llh']
                    best_tree = tree
        mutass_file = f"MOONSHOT2/{patient_num}/bootstrap{bootstrap_num}/result.mutass.zip"
        with ZipFile(mutass_file, 'r') as zip:
            with zip.open(best_tree + ".json", ) as g:
                tree_detail = json.load(g)
        mut_assignments = tree_detail['mut_assignments']
        with gzip.open(muts_file, "r") as k:
            # with open(muts_file, "r") as k:
            j_muts = json.loads(k.read().decode('utf-8'))
        final_tree_cp = {}
        for p, c_list in j_summ['trees'][best_tree]['structure'].items():
            p_list = ['normal'] if p == '0' else [j_muts['ssms'][m]['name'] for m in
                                                  tree_detail['mut_assignments'][p]['ssms']]
            p_final = tuple(set(p_list))
            for c in c_list:
                c_final = tuple(
                    set([j_muts['ssms'][m]['name'] for m in tree_detail['mut_assignments'][str(c)]['ssms']]))
                final_tree_cp[c_final] = p_final
        #         if final_tree_cp not in tree_distribution.keys():
        #             tree_distribution[final_tree_cp] = []
        tree_structure = j_summ['trees'][best_tree]['structure']
        tree_dict = {}
        for parent, children in tree_structure.items():
            tree_dict.setdefault(int(parent), children)
        node_dict = {}
        node_dict_re = {}
        for p in tree_detail['mut_assignments'].keys():
            node_dict[p] = [j_muts['ssms'][m]['name'] for m in tree_detail['mut_assignments'][p]['ssms']]
            node_dict_re[tuple([j_muts['ssms'][m]['name'] for m in tree_detail['mut_assignments'][p]['ssms']])] = p

        #         g = render_tumor_tree(tree_structure, node_dict)
        #         g.render(filename=str(patient_num)+'_tree_bootstrap' + str(bootstrap_num))
        prev_mat = []
        population_dict = j_summ['trees'][best_tree]['populations']
        clonal_freq = {}
        for node, populations in population_dict.items():
            # prev_blood = populations['cellular_prevalence'][0] - sum(
            #     [population_dict[str(child)]['cellular_prevalence'][0] for child in
            #      tree_structure[str(node)]] if node in tree_structure.keys() else [0])
            # clonal_freq[node] = prev_blood
            clonal_freq[node] = populations['cellular_prevalence'][0]

        tree_list.append(tree_dict)
        node_list.append(node_dict)
        clonal_freq_list.append(clonal_freq)
        tree_freq_list.append(1)


gene2idx = {}
inter = pd.read_excel("patient_486.xlsx", sheet_name='common_blood_tissue_no_germline', index_col=0)
calls = inter
for i in range(inter.shape[0]):
    gene = calls.iloc[i, 0]
    if not isinstance(gene, str):
        gene = calls.iloc[i, 1] + '_' + str(calls.iloc[i, 2])
    gene_list.append(gene)
    gene2idx[gene] = i

#scrub node_list
node_list_scrub = []
for node_dict in node_list:
    temp = {}
    for key, values in node_dict.items():
        temp.setdefault(int(key), values)
    node_list_scrub.append(temp)

tree_list_sub = [tree_list[i] for i in range(5) ]
node_list_sub = [node_list_scrub[i] for i in range(5) ]
clonal_freq_list_sub = [clonal_freq_list[i] for i in range(5) ]
tree_freq_list_sub = [tree_freq_list[i] for i in range(5)]


n_markers = 3
# mh = MetropolisHasting(gene_list=gene_list, n_markers=n_markers, tree_list=tree_list, node_list=node_list,
#                        clonal_freq_list=clonal_freq_list, tree_freq_list=tree_freq_list, gene2idx=gene2idx)
# mh.initialization()
# mh.sampling(n_iter=500, read_depth=10000, annealing=False, const_max=1, const_min=1)

selected_markers = select_markers_fractions_gp(gene_list, n_markers, tree_list, node_list_scrub, gene2idx, tree_freq_list)
print(selected_markers)