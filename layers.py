import torch
from torch_sparse import spmm


class SparseNGCNLayer(torch.nn.Module):
    """
    Multi-scale Sparse Feature Matrix GCN layer.
    :param in_channels: Number of features.
    :param out_channels: Number of filters.
    :param iterations: Adjacency matrix power order.
    :param dropout_rate: Dropout value.
    """
    def __init__(self, in_channels, out_channels, iterations, dropout_rate, device):
        super(SparseNGCNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.iterations = iterations
        self.dropout_rate = dropout_rate
        self.device = device
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining the weight matrices.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels)).to(self.device)
        self.bias = torch.nn.Parameter(torch.Tensor(1, self.out_channels)).to(self.device)

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, normalized_adjacency_matrix, features):
        """
        Doing a forward pass.
        :param normalized_adjacency_matrix: Normalized adjacency matrix.
        :param features: Feature matrix.
        :return base_features: Convolved features.
        """
        feature_count, _ = torch.max(features["indices"], dim=1)
        feature_count = feature_count + 1
        base_features = spmm(features["indices"], features["values"], feature_count[0],
                             feature_count[1], self.weight_matrix)

        base_features = base_features + self.bias

        base_features = torch.nn.functional.dropout(base_features,
                                                    p=self.dropout_rate,
                                                    training=self.training)

        base_features = torch.nn.functional.relu(base_features)
        for _ in range(self.iterations-1):
            base_features = spmm(normalized_adjacency_matrix["indices"],
                                 normalized_adjacency_matrix["values"],
                                 base_features.shape[0],
                                 base_features.shape[0],
                                 base_features)
        return base_features

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.out_channels) + ')'

class DenseNGCNLayer(torch.nn.Module):
    """
    Multi-scale Dense Feature Matrix GCN layer.
    :param in_channels: Number of features.
    :param out_channels: Number of filters.
    :param iterations: Adjacency matrix power order.
    :param dropout_rate: Dropout value.
    """
    def __init__(self, in_channels, out_channels, iterations, dropout_rate, device):
        super(DenseNGCNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.iterations = iterations
        self.dropout_rate = dropout_rate
        self.device = device
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining the weight matrices.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels)).to(self.device)
        self.bias = torch.nn.Parameter(torch.Tensor(1, self.out_channels)).to(self.device)

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, normalized_adjacency_matrix, features):
        """
        Doing a forward pass.
        :param normalized_adjacency_matrix: Normalized adjacency matrix.
        :param features: Feature matrix.
        :return base_features: Convolved features.
        """
        base_features = torch.mm(features, self.weight_matrix)
        base_features = torch.nn.functional.dropout(base_features,
                                                    p=self.dropout_rate,
                                                    training=self.training)
        for _ in range(self.iterations-1):
            base_features = spmm(normalized_adjacency_matrix["indices"],
                                 normalized_adjacency_matrix["values"],
                                 base_features.shape[0],
                                 base_features.shape[0],
                                 base_features)
        base_features = base_features + self.bias
        return base_features

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.out_channels) + ')'
