import pickle
import time

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Any
from tqdm import tqdm
import os
import json

import logging

from config import settings, dataset, filename
import psutil


support_percent = 80
file_created, frequent_file_created = False, False
size_path = 5
number_of_generated_candidate = 0
name_of_file_without_extension = filename.split('/')[-1].split('.')[0]
if not os.path.exists('cache'):
    os.mkdir('cache')
DATA_FILE = f"cache/raw_data_grite_{name_of_file_without_extension}.txt"
TIME_FILE = f"cache/raw_data_grite_{name_of_file_without_extension}_time.txt"

# logging.basicConfig(filename=f'grite_hcv_{support_percent}.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)



class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def visual_dict(d):
    lines = ['{']
    for i, (key, value) in enumerate(d.items()):
        line = f'   "{key}": {{"row": {value["row"]},"col": {value["col"]}, "sum": {value["sum"]}}}'
        if i < len(d) - 1:
            line += ','
        lines.append(line)
    lines.append('}')

    return '\n'.join(lines)


def from_order_index_to_binary_matrix(col: pd.Series):
    n = len(col)
    mat = pd.DataFrame(
        data=np.zeros((n, n), dtype=bool),
        columns=list(range(n)),
        index=list(range(n)),
    )
    for i_, i in enumerate(col.index):
        for j in col.index[i_ + 1:]:
            mat.loc[i, j] = True
            if col[i] == col[j]:
                mat.loc[j, i] = True
    return mat


@dataclass
class GradualPattern:
    comp_type = {"+": "-", "-": "+"}

    def __init__(self, name, type, bin_mat):
        self.col_name: List[str] = name
        self.variations: List[str] = type
        self.bin_mat: pd.DataFrame = bin_mat
        self._sons = None
        self.freq = 0

    def complement(self):
        c = ~self.bin_mat
        for i in range(len(c)):
            c.iloc[i, i] = False

        for i in range(len(c)):  # This loop is for each diagonal
            for j in range(i):
                if self.bin_mat.iloc[i, j] and self.bin_mat.iloc[j, i]:
                    c.iloc[i, j] = True
                    c.iloc[j, i] = True

        return GradualPattern(
            name=list(self.col_name), type=[self.comp_type[t] for t in self.variations], bin_mat=c
        )

    def init_memory(self):
        self.memory = {col: -1 for col in self.bin_mat.columns}

    def _compute_son(self):
        columns = self.bin_mat.columns
        self._sons = {col: [] for col in columns}
        self.init_memory()
        s = set()
        for i, col in enumerate(columns):
            for j in columns[i + 1:]:
                o = False
                if self.bin_mat.loc[col, j]:
                    self._sons[col].append(j)
                    s.add(j)
                    o = True
                if self.bin_mat.loc[j, col]:
                    if not o:
                        self._sons[j].append(col)
                        s.add(col)

        self.roots = list(set(columns).difference(s))

    @property
    def sons(self):
        if self._sons is None:
            self._compute_son()
        return self._sons

    @sons.setter
    def sons(self, sons):
        self._sons = sons

    def names(self) -> List[str | Any]:
        return list(map(lambda x: str(x[0]) + x[1], zip(sorted(self.col_name), self.variations)))

    def dynamic_covering(self, node, memory: dict):
        stack = [node]
        visited = set()

        while stack:
            current_node = stack[-1]

            if current_node not in visited:
                visited.add(current_node)
                # Process node and push unvisited children onto the stack
                for son in self.sons[current_node]:
                    if memory[son] == -1:
                        stack.append(son)

            else:
                # All children processed, update the depth for the current node
                max_child_depth = 0
                for son in self.sons[current_node]:
                    max_child_depth = max(max_child_depth, memory[son])
                memory[current_node] = max_child_depth + 1

                # Pop the current node from the stack
                stack.pop()
        return memory[node]

    def compute_frequency(self, item_number):
        self.sons
        for root in self.roots:
            self.init_memory()
            self.dynamic_covering(root, self.memory)
            freq = self.memory[max(self.memory, key=self.memory.get)] / item_number
            if freq > self.freq:
                self.freq = freq

    def __str__(self) -> str:
        s = str(self.names())
        s += "\n" + str(self.bin_mat.astype(np.int8))
        return s

    def __repr__(self) -> str:
        return str(self.names())


# je me permet ça car mes liste n'ont pas de doublons
def sublist(sub_list, liste):
    return set(sub_list).issubset(set(liste))


def join(
        item1: GradualPattern,
        item2: GradualPattern,
        already_generate,
        gradual_patterns: List[GradualPattern],
        k,
):
    name1, name2 = item1.names(), item2.names()
    diff = list(set(name1).difference(name2).union(set(name2).difference(name1)))

    if len(diff) != 2:
        return False, None

    if diff[0][:-1] == diff[1][:-1]:
        return False, None

    name_types = list(
        set(zip(item1.col_name, item1.variations)).union(zip(item2.col_name, item2.variations))
    )
    name_types = sorted(name_types, key=lambda x: x[0])
    name = list(map(lambda x: x[0], name_types))
    type_ = list(map(lambda x: x[1], name_types))

    candidate_name = list(map(lambda x: str(x[0]) + x[1], zip(name, type_)))
    if candidate_name in already_generate:
        return False, None

    result = pd.DataFrame(
        data=np.ones_like(item1.bin_mat, dtype=bool),
        columns=item1.bin_mat.columns,
        index=item1.bin_mat.index,
    )

    considered = 0

    for grad_p in gradual_patterns:
        if sublist(grad_p.names(), candidate_name):
            result = result & grad_p.bin_mat
            considered += 1

    if considered != k:
        return False, None
    global size_path
    result = result.fillna(False)
    columns = result.columns.copy()
    for col in columns:
        if (result.loc[:, col].sum() + result.loc[col, :].sum()) < (size_path - 1):
            result.drop(col, axis=0, inplace=True)
            result.drop(col, axis=1, inplace=True)

    if result.shape[0] < size_path:
        return False, None

    return True, GradualPattern(name=name, type=type_, bin_mat=result)


def generate_gradual_k(
        gradual_items: List[GradualPattern], item_number, min_freq, k
) -> List[GradualPattern]:
    if len(gradual_items) < k:
        return []

    gradual_k = []
    already_generated = []

    for i in tqdm(range(0, len(gradual_items), 2), f"Gradual patterns size={k}"):
        for j in range(i + 2, len(gradual_items)):
            is_posible, candidate = join(
                gradual_items[i], gradual_items[j], already_generated, gradual_items, k
            )
            # if candidate and candidate.names() in exclu_gp:
            #     with open("exclu_gp.txt", "a") as f:
            #         m = candidate.bin_mat.astype(np.int8)
            #         d = {col: {"row": m.loc[col, :].sum(), "col": m[col].sum()} for col in m.columns}
            #         for col in candidate.bin_mat.columns:
            #             d[col]["sum"] = d[col]["row"] + d[col]["col"]
            #         f.write(
            #             f"{candidate.names()}\n{visual_dict(d)}\n{candidate.bin_mat.astype(int).to_string(index=False)}\n")
            if not is_posible:
                continue
            global number_of_generated_candidate
            number_of_generated_candidate += 2
            t_ = time.time()
            candidate.compute_frequency(item_number)
            logger.info(f'check compute frequency {time.time() - t_}\t freq={candidate.freq}')
            m = candidate.bin_mat.astype(np.int8)
            if candidate.freq >= min_freq:
                global frequent_file_created
                if not frequent_file_created:
                    frequent_file_created = True
                    with open("col_frequent_pruning.txt", "w") as f:
                        pass
                s = list()

                with open("col_frequent_pruning.txt", "a") as f:
                    for col in candidate.bin_mat.columns:
                        s.append(candidate.bin_mat[col].sum())
                    d = {col: {"row": m.loc[col, :].sum(), "col": m[col].sum()} for col in m.columns}
                    for col in candidate.bin_mat.columns:
                        d[col]["sum"] = d[col]["row"] + d[col]["col"]
                    f.write(
                        f"{candidate.names()}\n{visual_dict(d)}\n{candidate.bin_mat.astype(int).to_string(index=False)}\n")
                gradual_k.append(candidate)
                already_generated.append(gradual_k[-1].names())
                gradual_k.append(candidate.complement())
                already_generated.append(gradual_k[-1].names())
                gradual_k[-1].freq = candidate.freq
            # else:
            #     global file_created
            #     if not file_created:
            #         file_created = True
            #         with open("matbin_non_freq.txt", "w") as f:
            #             pass
            #     with open("matbin_non_freq.txt", "a") as f:
            #         s = list()
            #         for col in candidate.bin_mat.columns:
            #             s.append(candidate.bin_mat[col].sum())
            #         f.write(f'{s} {s[size_path-1] >= size_path-1}\n')
            #         d = {col: {"row": m.loc[col, :].sum(), "col": m[col].sum()} for col in m.columns}
            #         for col in m.columns:
            #             d[col]["sum"] = d[col]["row"] + d[col]["col"]
            #         f.write(f"{candidate.names()}\n{visual_dict(d)}\n{candidate.bin_mat.astype(int).to_string(index=False)}\n")
    return gradual_k


def grite(dataset: pd.DataFrame, min_freq=0.7) -> dict:
    """Implementation of GRITE algorithm from Di-Jorio, L., Laurent, A., & Teisseire, M. (2009). Mining Frequent Gradual Itemsets from Large Databases. In N. M. Adams, C. Robardet, A. Siebes, & J.-F. Boulicaut (Éds.), Advances in Intelligent Data Analysis VIII (p. 297‑308). Springer. https://doi.org/10.1007/978-3-642-03915-7_26

    Note that we don't care about gradual pattern where we just have one item inside. And also gradual pattern of size 1.


    Args:
        dataset (pd.DataFrame): The dataset from which graduals rules should be extracted
        min_freq (float, optional): The minimum frequency Defaults to 0.7.

    Returns:
        dict: _description_
    """
    if min_freq >= 1:
        min_freq = min_freq / dataset.shape[0]
    columns = dataset.columns
    gradual_items = {1: []}
    global number_of_generated_candidate
    number_of_generated_candidate += 2 * len(columns)
    # generate 1-gradual-pattern
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as f:
            gradual_items = pickle.load(f)
    else:
        t = start_time = time.time()
        for col in tqdm(columns, f"Gradual patterns size=1"):
            s = from_order_index_to_binary_matrix(dataset.loc[:, col].sort_values())
            grad = GradualPattern(name=[col], type=["+"], bin_mat=s)
            grad.compute_frequency(dataset.shape[0])
            gradual_items[1].append(grad)
            gradual_items[1].append(grad.complement())
            gradual_items[1][-1].freq = grad.freq

        with open(DATA_FILE, "wb") as f:
            pickle.dump(gradual_items, f)

        with open(TIME_FILE, "wb") as f:
            pickle.dump((time.time() - start_time), f)

    # We don't really care about gradual pattern of size 1
    total = 0

    k = 2
    # We generate until the previous generation of gradual pattern is empty
    while gradual_items[k - 1]:
        gradual_items[k] = generate_gradual_k(
            gradual_items[k - 1], dataset.shape[0], min_freq, k
        )
        total += len(gradual_items[k])
        k += 1
    return {"total": total, "gradual_pattern": gradual_items}


def main():
    import time
    import sys
    import os

    # filename = "../../data/example.txt"
    # dataset = pd.read_csv(filename, header=0, sep="\s+")

    # filename = "../../data/medium_size_dataset_for_gp.csv"
    # dataset = pd.read_csv(filename, header=0, sep="\s+")
    # filename = "../../explainability/train_data.txt"
    # dataset = pd.read_csv(filename, header=0, sep="\s+")
        # filename = "../../data/hcv_data.csv"
        # dataset = pd.read_csv(filename, header=0, sep=";")
    # filename = "data_1.txt"
    # dataset = pd.read_csv(filename, header=0, sep="\s+")
    # if "0" in dataset.columns:
    #     dataset.drop(["0"], inplace=True)
    # filename = "data_grap.csv"
    # dataset = pd.read_csv(filename, header=0, sep="\s+")

    start_time = time.time()
    if os.path.exists(TIME_FILE):
        with open(TIME_FILE, "rb") as f:
            start_time += pickle.load(f)
    global size_path

    size_path = settings.size_path
    if len(sys.argv) > 1:
        if round(float(sys.argv[1])) == float(sys.argv[1]):
            size_path = int(float(sys.argv[1]))
        else:
            size_path = float(sys.argv[1])
    process = psutil.Process(os.getpid())
    base_memory_usage = process.memory_info().rss
    r = grite(dataset, min_freq=size_path)
    used_memory = process.memory_info().rss - base_memory_usage
    with open(f'memo_col_prunning{name_of_file_without_extension}', "a") as f:
        f.write(f"{size_path}, {used_memory}\n")
    print(dataset.shape)
    print("--- %s seconds ---" % (time.time() - start_time))
    logger.info("--- %s seconds ---" % (time.time() - start_time))
    l = []
    m = []
    # print(r)
    n = 0
    for i in range(1, len(r["gradual_pattern"])):
        n += len(r["gradual_pattern"][i])
        # for grad_pattern in r["gradual_pattern"][i]:
        #     m.append(grad_pattern.bin_mat)
        #     l.append((f'({", ".join(grad_pattern.names())})', grad_pattern.freq))
    # logger.info(f'Nombre de #####: {n}')
    # logger.info(f'{l}')
    # pd.DataFrame(l).to_excel(
    #     os.path.join("results", f"grite_{filename.split('.')[0]}.xlsx"), index=False
    # )
    # for g, m_b in zip(l, m):
    #     print(g)
        # print(m_b.astype(int))
    print(f"{n} Motifs Fréquents")
    print(f"{number_of_generated_candidate} nombre de candidats pour lesquels la fréquence a été calculé")


if __name__ == "__main__":
    main()
