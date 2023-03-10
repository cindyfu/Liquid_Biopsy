### Simulation file for Zhicheng Luo

from collections import deque

# simulation, read count

def simulate_clonal_tree(d=3, k=5, a=0.3, b=0.3):
    """
    input:
    d: maximum degree of each node
    k: number of nodes
    a, b: the parameter for the beta distribution
    
    output:
    tree: dict represents tree structure
    freq_obs: frequencies for each node
    """
    
    k = len(tree.values()) + 1
    tree = {}
    root = 0
    nodes = [root]
    nodes_deque = deque([root])
    freq_true = np.random.beta(a, b, k)
    freq_obs = np.random.dirichlet(freq_true)
    freq_obs = {i: freq_obs[i] for i in range(len(freq_obs))}
    print(freq_true, freq_obs)
    ### simulate the ground truth tumor tree
    end_loop = False
    while not end_loop:
        current_node = nodes_deque.popleft()
        while True:
            if current_node == root:
                child_num = np.random.choice(range(1,d+1))
            else:
                child_num = np.random.choice(d+1)
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
    return tree, freq_obs

    
def simulate_variant_reads(tree, freq, depth=50, mutation_rate=10):
    '''
    input:
    tree: dict represents tree structure
    freq: frequencies for each node
    depth: a poisson mean of sequencing depth
    mutation_rate: a poisson mean of number of mutations
    
    simulate a set of mutations for each node, the number of mutations holds a poisson distribution
    simulate the reference read counts which holds a poisson distribution
    and variant read counts for each mutation which holds a binomial distribution with the parameter of frequencies

    output:
    node_mutations: dict for tree node - mutation_list pair, i.e. {0: [0,1,2], 1:[3,4], 2:[5,6,7]}
    mutation_refer_count: dict for mutation - reference read counts
    mutation_variant_count: dict for mutation - variant read counts
    '''
    #TODO
    pass
    