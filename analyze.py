
def combine_tree(node_dict, node_dict_re, tree_structure, tree_cp, clonal_freq):
    global tree_distribution
    found_flag = False
    for idx in range(len(tree_distribution['cp_tree'])):
        if tree_cp == tree_distribution['cp_tree'][idx]:
            tree_distribution['freq'][idx] += 1
            match_dict = match_trees(node_dict_re, tree_distribution['node_dict_re'][idx])
            for node, freq in clonal_freq.items():
                 tree_distribution['clonal_freq'][idx][match_dict[node]].append(freq[0])
            found_flag = True
            break
    if not found_flag:
        tree_distribution['cp_tree'].append(tree_cp)
        tree_distribution['node_dict'].append(node_dict)
        tree_distribution['node_dict_re'].append(node_dict_re)
        tree_distribution['tree_structure'].append(tree_structure)
        tree_distribution['freq'].append(1)
        tree_distribution['clonal_freq'].append(clonal_freq)


def match_trees(node_dict_re, target_node_dict_re):
    match_dict = {'0': '0'}
    for muts, node in target_node_dict_re.items():
        match_dict[node_dict_re[muts]] = node
    print(match_dict, 'm')
    return match_dict