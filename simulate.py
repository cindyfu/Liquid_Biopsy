from collections import deque
import numpy as np

# simulation, read count

def simulate_clonal_tree(d=3, k=5, a=6, b=6):
    """
    input:
    d: maximum degree of each node
    k: number of nodes
    a, b: the parameter for the beta distribution

    output:
    tree: dict represents tree structure
    freq_obs: frequencies for each node
    """

    tree = {}
    root = 0
    nodes = [root]
    nodes_deque = deque([root])
    child_node = root
    freq_true = np.random.beta(a, b, k)
    freq_obs_tissue = np.random.dirichlet(freq_true)
    freq_obs_tissue_dict = {i: freq_obs_tissue[i] for i in range(len(freq_obs_tissue))}
    freq_obs_blood = np.random.dirichlet(freq_true)
    freq_obs_blood[0] = 0
    new_freq_obs_blood_normal = np.random.rand() / 10 + 0.9
    freq_obs_blood_dict = {i: freq_obs_blood[i] / np.sum(freq_obs_blood) * (
                1 - new_freq_obs_blood_normal) if i != 0 else new_freq_obs_blood_normal for i in
                           range(len(freq_obs_blood))}

    print(freq_true, freq_obs_tissue_dict, freq_obs_blood_dict)
    ### simulate the ground truth tumor tree
    end_loop = False
    while not end_loop:
        current_node = nodes_deque.popleft()
        while True:
            if current_node == root:
                child_num = np.random.choice(range(1, d + 1))
            else:
                child_num = np.random.choice(d + 1)
            if child_num + len(nodes) < k:
                break
            elif child_num + len(nodes) == k:
                end_loop = True
                break
        if current_node not in tree.keys() and child_num != 0:
            tree.setdefault(current_node, [])
        for child in range(child_num):
            child_node += 1
            nodes.append(child_node)
            nodes_deque.append(child_node)
            tree[current_node].append(child_node)
    return tree, freq_true, freq_obs_tissue_dict, freq_obs_blood_dict


def simulate_variant_reads(tree, freq_tissue, freq_blood, depth=50, mutation_rate=10):
    '''
    input:
    tree: dict represents tree structure
    freq_tissue: frequencies for each node in tissue sample
    freq_blood: frequencies for each node in blood sample
    depth: a poisson mean of sequencing depth
    mutation_rate: a poisson mean of number of mutations

    simulate a set of mutations for each node, the number of mutations holds a poisson distribution
    simulate the reference read counts which holds a poisson distribution
    and variant read counts for each mutation which holds a binomial distribution with the parameter of frequencies

    output:
    node_mutations: dict for tree node - mutation_list pair, i.e. {0: [0,1,2], 1:[3,4], 2:[5,6,7]}
    mutation_refer_count_tissue: dict for mutation - reference read counts for tissue sample
    mutation_variant_count_tissue: dict for mutation - variant read counts for tissue sample
    mutation_refer_count_blood: dict for mutation - reference read counts for blood sample
    mutation_variant_count_blood: dict for mutation - variant read counts for blood sample
    '''
    mutation_list = np.random.poisson(mutation_rate, (len(freq_tissue) - 1))
    mutation_list = np.insert(mutation_list, 0, 0)
    # Number of Mutations in each node;
    # Eg: [12,11,13,8,9,10,11]

    depth_list_T = np.random.poisson(depth, mutation_list.sum())
    depth_list_B = np.random.poisson(depth, mutation_list.sum())
    # Create Tissue and Blood list of depth with length equals to the sum of all mutations;

    variant_list_T = np.zeros(mutation_list.sum(), dtype=int)
    reference_list_T = np.zeros(mutation_list.sum(), dtype=int)

    variant_list_B = np.zeros(mutation_list.sum(), dtype=int)
    reference_list_B = np.zeros(mutation_list.sum(), dtype=int)

    tracker = 0
    node_mutation = {}
    mutation_variant_T = {}
    mutation_reference_T = {}
    mutation_variant_B = {}
    mutation_reference_B = {}
    # iterate over the mutation list for each node and choose the depth
    # and perform binomial distribution based on the value and number of mutations
    for key, value in freq_tissue.items():
        num_mutation = mutation_list[key]
        start = tracker
        tracker += num_mutation
        end = tracker
        node_mutation[key] = [i for i in range(start, end)]

        variant_list_T[start:end] = np.random.binomial(depth_list_T[start:end], value)
        reference_list_T[start:end] = np.subtract(depth_list_T[start:end], variant_list_T[start:end])

        variant_list_B[start:end] = np.random.binomial(depth_list_B[start:end], freq_blood[key])
        reference_list_B[start:end] = np.subtract(depth_list_B[start:end], variant_list_B[start:end])
        # Depth :  [50,51,53,46,47,49...]
        # Variant: [ 2, 3, 4, 5, 1, 2...]

    for i in range(mutation_list.sum()):
        # assign TISSUE and BLOOD variant and refer reads for each mutation dictionary;
        mutation_variant_T[i] = variant_list_T[i]
        mutation_reference_T[i] = reference_list_T[i]

        mutation_variant_B[i] = variant_list_B[i]
        mutation_reference_B[i] = reference_list_B[i]

    # Converting all dictionary's value into JSON serializable;
    mutation_variant_T = {k: int(v) for k, v in mutation_variant_T.items()}
    mutation_reference_T = {k: int(v) for k, v in mutation_reference_T.items()}
    mutation_variant_B = {k: int(v) for k, v in mutation_variant_B.items()}
    mutation_reference_B = {k: int(v) for k, v in mutation_reference_B.items()}

    return node_mutation, mutation_variant_T, mutation_reference_T, mutation_variant_B, mutation_reference_B


def simulation_plyWGS(v_blood, a_blood, v_tissue, a_tissue, output_path):
    """
        This function used the simulated output data of mutations
        to write them back to the plyWGS format can be ready to
        run in the plyWGS program;
    """
    d_blood = dict([(k, v_blood[k] + a_blood[k]) for k in set(v_blood) & set(a_blood)])
    d_tissue = dict([(k, v_tissue[k] + a_tissue[k]) for k in set(v_tissue) & set(a_tissue)])
    # The total number of read depth for blood and tissue samples;

    plyWGS_format = []
    count = 0
    for i in range(len(a_blood)):
        # Iterate over all simulated mutations
        plyWGS_format.append({'id': 's' + str(count), 'gene': f"mut_{i}", 'a': str(a_blood[i]) + ',' + str(a_tissue[i]),
                              'd': str(d_blood[i]) + ',' + str(d_tissue[i]), 'mu_r': 0.999, 'mu_v': 0.499})
        count += 1

    df_plyWGS = pd.DataFrame(plyWGS_format)
    df_plyWGS.to_csv(output_path / f'ssm_data.txt', index=False, sep='\t')


def simulation_excel(v_blood, a_blood, v_tissue, a_tissue, output_path):
    '''
        This function used the simulated output data of mutations
        to write them back to an excel that can be ready to perform
        further implementations;
    '''
    d_blood = dict([(k, v_blood[k] + a_blood[k]) for k in set(v_blood) & set(a_blood)])
    d_tissue = dict([(k, v_tissue[k] + a_tissue[k]) for k in set(v_tissue) & set(a_tissue)])
    freq_blood = dict([(k, v_blood[k] / d_blood[k]) for k in set(v_blood) & set(d_blood)])
    freq_tissue = dict([(k, v_tissue[k] / d_tissue[k]) for k in set(v_tissue) & set(d_tissue)])

    excel_format = []
    count = 0
    for i in range(len(a_blood)):
        # Iterate over all simulated mutations
        excel_format.append({'0': '',
                             'Gene': f"mut_{i}",
                             '1': '',
                             '2': '',
                             '3': '',
                             '4': '',
                             'Allele Frequency_x': str(freq_blood[i]),
                             'Depth_x': str(d_blood[i]),
                             '5': '',
                             '6': '',
                             '7': '',
                             '8': '',
                             '9': '',
                             'Allele Frequency_y': str(freq_tissue[i]), 'Depth_y': str(d_tissue[i]),
                             '10': '',
                             '11': '',
                             '12': ''})
        count += 1

    df_excel = pd.DataFrame(excel_format)
    writer = pd.ExcelWriter(output_path / "simulated_mut.xlsx", engine='openpyxl')
    df_excel.to_excel(writer, sheet_name='common_blood_tissue_no_germline', index=False, engine='openpyxl')
    writer.save()
    # The total number of read depth for blood and tissue samples;





