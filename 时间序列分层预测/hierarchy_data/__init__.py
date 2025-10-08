import numpy as np
import pandas as pd


class TSNode(object):
    def __init__(self, idx, name, parent) -> None:
        self.idx = idx
        self.name = name
        self.parent = parent
        self.children = []


class LabourHierarchyData(object):
    """
    Defines a Hierarchical Dataset
    """

    def __init__(self, data_file="data/fault_count/tourism.csv") -> None:
        self.data_file = data_file
        self.data, self.idx_dict, self.nodes = self.get_hierarchy()

    def get_hierarchy(self):
        df = pd.read_csv(self.data_file)
        data = df.values[:, 1:]
        data = np.array([np.array(x, dtype=np.float32) for x in data])
        titles = df.columns.values[1:]
        titles = [x[2:-2] for x in titles]
        titles = [x.replace("'", "").split(",") for x in titles]
        titles = [[y.strip() for y in x] for x in titles]
        nodes = [TSNode(0, "Total", None)]
        idx_dict = {"Total": 0}
        for n, t in enumerate(titles[1:]):
            idx_dict["_".join(t)] = n + 1
            if len(t) > 1:
                parent_name = "_".join(t[1:])
            else:
                parent_name = "Total"
            curr_node = TSNode(n + 1, "_".join(t), nodes[idx_dict[parent_name]])
            nodes.append(curr_node)
            nodes[idx_dict[parent_name]].children.append(curr_node)
        return data.T, idx_dict, nodes


class ChargingHierarchyData(object):
    def __init__(self, data_file="data/fault_count/data.csv") -> None:
        self.data_file = data_file
        self.data, self.idx_dict, self.nodes = self.get_hierarchy()

    def get_hierarchy(self):
        df = pd.read_csv(self.data_file)
        data = df.values[:, 1:]
        data = np.array([np.array(x, dtype=np.float32) for x in data])
        titles = df.columns.values[1:]
        # print("titles:", titles)
        nodes = [TSNode(0, "长沙市", None)]
        idx_dict = {"长沙市": 0}

        for n, t in enumerate(titles[1:]):
            # print(len(t))
            idx_dict[t] = n + 1
            # print('idx_dict', idx_dict)
            # print('t', t)

            # 使用第一个 '-' 后的部分作为 parent_name
            if '-' in t:
                parent_name = t.split('-', 1)[1]  # 分割并取第一个 '-' 后的部分
                # print('t', t)
                # print('t.split("-", 1)[1]', parent_name)
            else:
                parent_name = "长沙市"

            curr_node = TSNode(n + 1, t, nodes[idx_dict[parent_name]])
            nodes.append(curr_node)
            nodes[idx_dict[parent_name]].children.append(curr_node)

        return data.T, idx_dict, nodes


class GetLabour(object):
    def __init__(self, data_file="data/fault_count/new_labour.csv") -> None:
        self.data_file = data_file
        self.data, self.idx_dict, self.nodes = self.get_hierarchy()

    def get_hierarchy(self):
        df = pd.read_csv(self.data_file)
        data = df.values[:, 1:]
        data = np.array([np.array(x, dtype=np.float32) for x in data])
        titles = df.columns.values[1:]
        # print("titles:", titles)
        nodes = [TSNode(0, "Total", None)]
        idx_dict = {"Total": 0}

        for n, t in enumerate(titles[1:]):
            # print(len(t))
            idx_dict[t] = n + 1
            # print('idx_dict', idx_dict)
            # print('t', t)

            # 使用第一个 '-' 后的部分作为 parent_name
            if '-' in t:
                parent_name = t.split('-', 1)[1]  # 分割并取第一个 '-' 后的部分
                # print('t', t)
                # print('t.split("-", 1)[1]', parent_name)
            else:
                parent_name = "Total"

            curr_node = TSNode(n + 1, t, nodes[idx_dict[parent_name]])
            nodes.append(curr_node)
            nodes[idx_dict[parent_name]].children.append(curr_node)

        return data.T, idx_dict, nodes


def normalize_data(dataset):
    for node in reversed(dataset.nodes):
        if len(node.children) > 0:
            dataset.data[node.idx, :] /= len(node.children)
    return dataset


def unnormalize_data(dataset):
    for node in reversed(dataset.nodes):
        if len(node.children) > 0:
            dataset.data[node.idx, :] *= len(node.children)
    return dataset




