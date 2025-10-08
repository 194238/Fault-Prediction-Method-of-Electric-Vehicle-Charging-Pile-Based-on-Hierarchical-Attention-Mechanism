import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from models.utils import (
    Normal,
    float_tensor,
    logitexp,
    sample_DAG,
    sample_Clique,
    sample_bipartite,
    Flatten,
    one_hot,
    device,
)
from typing import List, Tuple, TypeVar, Union


class LatentAtten(nn.Module):
    """
    Attention on latent representation
    """

    def __init__(self, h_dim, key_dim=None) -> None:
        super(LatentAtten, self).__init__()
        if key_dim is None:
            key_dim = h_dim
        self.key_dim = key_dim
        self.key_layer = nn.Linear(h_dim, key_dim)
        self.query_layer = nn.Linear(h_dim, key_dim)

    def forward(self, h_M, h_R):
        key = self.key_layer(h_M)
        query = self.query_layer(h_R)
        atten = (key @ query.transpose(0, 1)) / math.sqrt(self.key_dim)
        atten = torch.softmax(atten, 1)
        return atten


class TimeSeriesAttention(nn.Module):
    """
    Attention module designed specifically for time series data.
    """

    def __init__(self, dim_in=40, value_dim=40, key_dim=40):
        """
        param dim_in: Dimensionality of input sequence features
        param value_dim: Dimension of value transform
        param key_dim: Dimension of key transform
        """
        super(TimeSeriesAttention, self).__init__()
        self.value_layer = nn.Linear(dim_in, value_dim)
        self.query_layer = nn.Linear(dim_in, value_dim)
        self.key_layer = nn.Linear(dim_in, key_dim)

        # Optional normalization for scaling
        self.scale = math.sqrt(key_dim)

    def forward(self, seq):
        """
        Forward pass for self-attention.

        param seq: Input sequence of shape [Batch, Seq_len, Hidden_dim]
        """
        # Transform the input to value, key, and query
        value = self.value_layer(seq)  # [Batch, Seq_len, value_dim]
        query = self.query_layer(seq)  # [Batch, Seq_len, value_dim]
        key = self.key_layer(seq)      # [Batch, Seq_len, key_dim]

        # Compute attention weights
        weights = torch.matmul(query, key.transpose(1, 2)) / self.scale  # [Batch, Seq_len, Seq_len]
        weights = torch.softmax(weights, dim=-1)  # Normalize over sequence length

        # Compute weighted output
        attended_output = torch.matmul(weights, value)  # [Batch, Seq_len, value_dim]

        return attended_output

    def forward_with_mask(self, seq, mask):
        """
        Forward pass with a mask to handle padding or irregular time series lengths.

        param seq: Input sequence of shape [Batch, Seq_len, Hidden_dim]
        param mask: Binary mask of shape [Batch, Seq_len], where 1 indicates valid time steps
        """
        # Transform input to value, key, and query
        value = self.value_layer(seq)  # [Batch, Seq_len, value_dim]
        query = self.query_layer(seq)  # [Batch, Seq_len, value_dim]
        key = self.key_layer(seq)      # [Batch, Seq_len, key_dim]

        # Compute attention weights
        weights = torch.matmul(query, key.transpose(1, 2)) / self.scale  # [Batch, Seq_len, Seq_len]

        # Apply the mask
        mask = mask.unsqueeze(1)  # [Batch, 1, Seq_len] for broadcasting
        weights = weights.masked_fill(mask == 0, float('-inf'))  # Mask invalid positions
        weights = torch.softmax(weights, dim=-1)  # Normalize over sequence length

        # Compute weighted output
        attended_output = torch.matmul(weights, value)  # [Batch, Seq_len, value_dim]

        return attended_output


class SparseAttention(nn.Module):
    """
    Sparse Attention Module for Time Series Data, designed to efficiently handle sparse (0/1) time series data.
    """

    def __init__(self, dim_in=40, value_dim=40, key_dim=40):
        """
        Initialize the Sparse Attention Layer for time series.

        param dim_in: Dimensionality of input sequence features.
        param value_dim: Dimension of value transform.
        param key_dim: Dimension of key transform.
        """
        super(SparseAttention, self).__init__()

        # Linear layers for transforming input to query, key, and value
        self.value_layer = nn.Linear(dim_in, value_dim)
        self.query_layer = nn.Linear(dim_in, value_dim)
        self.key_layer = nn.Linear(dim_in, key_dim)

        # Optional normalization for scaling
        self.scale = math.sqrt(key_dim)

    def forward(self, seq, mask=None):
        """
        Forward pass for Sparse Self-Attention.

        param seq: Input sequence of shape [Batch, Seq_len, Hidden_dim]
        param mask: Optional binary mask of shape [Batch, Seq_len], where 1 indicates valid time steps.
        """
        # Transform input to value, query, and key
        value = self.value_layer(seq)  # [Batch, Seq_len, value_dim]
        query = self.query_layer(seq)  # [Batch, Seq_len, value_dim]
        key = self.key_layer(seq)  # [Batch, Seq_len, key_dim]

        # Compute attention scores
        attn_scores = torch.matmul(query, key.transpose(1, 2)) / self.scale  # [Batch, Seq_len, Seq_len]

        # Apply the mask: Mask out invalid positions (zero entries in sparse data)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))  # Mask out padding (0 entries)

        # Normalize attention scores (softmax) over sequence length
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Compute weighted sum of values based on attention weights
        attended_output = torch.matmul(attn_weights, value)  # [Batch, Seq_len, value_dim]

        return attended_output

    def forward_with_mask(self, seq, mask):
        """
        Forward pass with a mask to handle sparse (0/1) time series data.

        param seq: Input sequence of shape [Batch, Seq_len, Hidden_dim]
        param mask: Binary mask of shape [Batch, Seq_len], where 1 indicates valid time steps.
        """
        # Transform input to value, query, and key
        value = self.value_layer(seq)  # [Batch, Seq_len, value_dim]
        query = self.query_layer(seq)  # [Batch, Seq_len, value_dim]
        key = self.key_layer(seq)  # [Batch, Seq_len, key_dim]

        # Compute attention scores (query * key)
        attn_scores = torch.matmul(query, key.transpose(1, 2)) / self.scale  # [Batch, Seq_len, Seq_len]

        # Apply the mask to ignore the padding values
        mask = mask.unsqueeze(1)  # [Batch, 1, Seq_len] for broadcasting
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))  # Mask out invalid positions
        attn_weights = torch.softmax(attn_scores, dim=-1)  # Normalize over sequence length

        # Compute the weighted sum of the values based on the attention weights
        attended_output = torch.matmul(attn_weights, value)  # [Batch, Seq_len, value_dim]

        return attended_output


class EmbedMetaTSAttenSeq(nn.Module):
    """
    Module to embed a sequence. Adds Attention module to
    """

    def __init__(
        self,
        dim_seq_in: int = 5,
        num_metadata: int = 555,
        dim_metadata: int = 10,
        rnn_out: int = 40,
        dim_out: int = 50,
        n_layers: int = 1,
        bidirectional: bool = False,
        attn=TimeSeriesAttention,
        dropout=0.0,
    ) -> None:
        """
        param dim_seq_in: Dimensionality of input vector (no. of age groups)
        param dim_out: Dimensionality of output vector
        param dim_metadata: Dimensions of metadata for all sequences
        param rnn_out: output dimension for rnn
        """
        super(EmbedMetaTSAttenSeq, self).__init__()

        self.dim_seq_in = dim_seq_in
        self.dim_metadata = dim_metadata
        self.num_metadata = num_metadata
        self.rnn_out = rnn_out
        self.dim_out = dim_out
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(
            input_size=self.dim_seq_in,
            hidden_size=self.rnn_out // 2 if self.bidirectional else self.rnn_out,
            bidirectional=bidirectional,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_layer = attn(self.rnn_out, self.rnn_out, self.rnn_out)
        self.emb_layer = nn.Embedding(self.num_metadata, self.dim_metadata)
        self.out_layer = [
            nn.Linear(
                in_features=self.rnn_out + self.dim_metadata, out_features=self.dim_out
            ),
            nn.Tanh(),
            nn.Dropout(dropout),
        ]
        self.out_layer = nn.Sequential(*self.out_layer)

    def forward_mask(self, seqs, metadata, mask):
        # Take last output from GRU
        latent_seqs = self.rnn(seqs)[0]
        latent_seqs = latent_seqs
        latent_seqs = self.attn_layer.forward_mask(latent_seqs, mask)
        latent_seqs = latent_seqs.sum(0)
        out = self.out_layer(torch.cat([latent_seqs, metadata], dim=1))
        return out

    def forward(self, seqs, metadata):
        # Take last output from GRU
        latent_seqs = self.rnn(seqs)[0]
        latent_seqs = self.attn_layer(torch.transpose(latent_seqs, 0, 1)).sum(0)
        meta_emb = self.emb_layer(metadata)
        out = self.out_layer(torch.cat([latent_seqs, meta_emb], dim=1))

        return out


class EmbedMetaSpaAttenSeq(nn.Module):
    """
    Module to embed a sequence. Adds Attention module to
    """

    def __init__(
        self,
        dim_seq_in: int = 5,
        num_metadata: int = 555,
        dim_metadata: int = 10,
        rnn_out: int = 40,
        dim_out: int = 50,
        n_layers: int = 1,
        bidirectional: bool = False,
        attn=SparseAttention,
        dropout=0.0,
    ) -> None:
        """
        param dim_seq_in: Dimensionality of input vector (no. of age groups)
        param dim_out: Dimensionality of output vector
        param dim_metadata: Dimensions of metadata for all sequences
        param rnn_out: output dimension for rnn
        """
        super(EmbedMetaSpaAttenSeq, self).__init__()

        self.dim_seq_in = dim_seq_in
        self.dim_metadata = dim_metadata
        self.num_metadata = num_metadata
        self.rnn_out = rnn_out
        self.dim_out = dim_out
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(
            input_size=self.dim_seq_in,
            hidden_size=self.rnn_out // 2 if self.bidirectional else self.rnn_out,
            bidirectional=bidirectional,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_layer = attn(self.rnn_out, self.rnn_out, self.rnn_out)
        self.emb_layer = nn.Embedding(self.num_metadata, self.dim_metadata)
        self.out_layer = [
            nn.Linear(
                in_features=self.rnn_out + self.dim_metadata, out_features=self.dim_out
            ),
            nn.Tanh(),
            nn.Dropout(dropout),
        ]
        self.out_layer = nn.Sequential(*self.out_layer)

    def forward_mask(self, seqs, metadata, mask):
        # Take last output from GRU
        latent_seqs = self.rnn(seqs)[0]
        latent_seqs = latent_seqs
        latent_seqs = self.attn_layer.forward_mask(latent_seqs, mask)
        latent_seqs = latent_seqs.sum(0)
        out = self.out_layer(torch.cat([latent_seqs, metadata], dim=1))
        return out

    def forward(self, seqs, metadata):

        # import pdb
        # pdb.set_trace()

        # Take last output from GRU
        latent_seqs = self.rnn(seqs)[0]
        latent_seqs = self.attn_layer(torch.transpose(latent_seqs, 0, 1)).sum(0)
        meta_emb = self.emb_layer(metadata)
        out = self.out_layer(torch.cat([latent_seqs, meta_emb], dim=1))

        return out


class TSRegressionSepFNP(nn.Module):
    """
    Functional Neural Process for regression
    """

    def __init__(
        self,
        dim_x=1,        # 输入维度
        dim_y=1,        # 输出维度
        dim_h=50,       # 隐藏层维度
        transf_y=None,      # 输出数据的变换（如标准化）
        n_layers=1,     # 隐藏层的数量
        use_plus=True,      # 是否使用FNP+，即加入额外的潜变量u
        dim_u=1,        # u，z的维度
        dim_z=1,
        fb_z=0.0,
        use_ref_labels=False,
        use_DAG=False,
        add_atten=False,
        nodes=555,
    ):
        """
        :param dim_x: Dimensionality of the input
        :param dim_y: Dimensionality of the output
        :param dim_h: Dimensionality of the hidden layers
        :param transf_y: Transformation of the output (e.g. standardization)
        :param n_layers: How many hidden layers to use
        :param use_plus: Whether to use the FNP+
        :param num_M: How many points exist in the training set that are not part of the reference set
        :param dim_u: Dimensionality of the latents in the embedding space
        :param dim_z: Dimensionality of the  latents that summarize the parents
        :param fb_z: How many free bits do we allow for the latent variable z
        """
        super(TSRegressionSepFNP, self).__init__()

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_h = dim_h
        self.dim_u = dim_u
        self.dim_z = dim_z
        self.use_plus = use_plus
        self.fb_z = fb_z
        self.transf_y = transf_y
        self.use_ref_labels = use_ref_labels
        self.use_DAG = use_DAG
        self.add_atten = add_atten
        self.nodes = nodes
        # normalizes the graph such that inner products correspond to averages of the parents
        self.norm_graph = lambda x: x / (torch.sum(x, 1, keepdim=True) + 1e-8)

        self.register_buffer("lambda_z", float_tensor(1).fill_(1e-8))

        # function that assigns the edge probabilities in the graph
        self.pairwise_g_logscale = nn.Parameter(
            float_tensor(1).fill_(math.log(math.sqrt(self.dim_u)))
        )
        self.pairwise_g = lambda x: logitexp(
            -0.5
            * torch.sum(
                torch.pow(x[:, self.dim_u :] - x[:, 0 : self.dim_u], 2), 1, keepdim=True
            )
            / self.pairwise_g_logscale.exp()
        ).view(x.size(0), 1)
        # transformation of the input

        init = [nn.Linear(dim_x, self.dim_h), nn.ReLU()]
        for i in range(n_layers - 1):
            init += [nn.Linear(self.dim_h, self.dim_h), nn.ReLU()]
        self.cond_trans = nn.Sequential(*init)
        # p(u|x)
        self.p_u = nn.Linear(self.dim_h, 2 * self.dim_u)
        # q(z|x)
        self.q_z = nn.Linear(self.dim_h, 2 * self.dim_z)
        # for p(z|A, XR, yR)
        if use_ref_labels:
            self.trans_cond_y = nn.Linear(self.dim_y, 2 * self.dim_z)

        # p(y|z) or p(y|z, u)
        # TODO: Add for sR input
        self.atten_ref = TimeSeriesAttention(self.dim_x)
        self.output_params = [
            nn.Parameter(
                torch.randn(
                    self.nodes,
                    self.dim_z + self.dim_x
                    if not self.use_plus
                    else self.dim_z + self.dim_u + self.dim_x,
                    self.dim_h,
                ).to(device)
                / math.sqrt(self.dim_h)
            ),
            nn.Parameter(
                torch.randn(self.nodes, self.dim_h, self.dim_h).to(device)
                / math.sqrt(self.dim_h)
            ),
            nn.Parameter(
                torch.randn(self.nodes, self.dim_h, 2 * dim_y).to(device)
                / math.sqrt(self.dim_h)
            ),
        ]
        if self.add_atten:
            self.atten_layer = LatentAtten(self.dim_h)

    def forward(self, XR, XM, yM, kl_anneal=1.0):
        # sR = self.atten_ref(XR).mean(dim=0)
        sR = XR.mean(dim=0)
        X_all = torch.cat([XR, XM], dim=0)
        H_all = self.cond_trans(X_all)

        # get U
        pu_mean_all, pu_logscale_all = torch.split(self.p_u(H_all), self.dim_u, dim=1)
        pu = Normal(pu_mean_all, pu_logscale_all)
        u = pu.rsample()

        # get G
        if self.use_DAG:
            G = sample_DAG(u[0 : XR.size(0)], self.pairwise_g, training=self.training)
        else:
            G = sample_Clique(
                u[0 : XR.size(0)], self.pairwise_g, training=self.training
            )

        # get A
        A = sample_bipartite(
            u[XR.size(0) :], u[0 : XR.size(0)], self.pairwise_g, training=self.training
        )
        if self.add_atten:
            HR, HM = H_all[0 : XR.size(0)], H_all[XR.size(0) :]
            atten = self.atten_layer(HM, HR)
            A = A * atten

        # get Z
        qz_mean_all, qz_logscale_all = torch.split(self.q_z(H_all), self.dim_z, 1)
        qz = Normal(qz_mean_all, qz_logscale_all)
        z = qz.rsample()

        pz_mean_all = torch.mm(
            self.norm_graph(torch.cat([G, A], dim=0)), qz_mean_all[0 : XR.size(0)],
        )
        pz_logscale_all = torch.mm(
            self.norm_graph(torch.cat([G, A], dim=0)), qz_logscale_all[0 : XR.size(0)],
        )

        pz = Normal(pz_mean_all, pz_logscale_all)

        pqz_all = pz.log_prob(z) - qz.log_prob(z)

        # apply free bits for the latent z
        if self.fb_z > 0:
            log_qpz = -torch.sum(pqz_all)

            if self.training:
                if log_qpz.item() > self.fb_z * z.size(0) * z.size(1) * (1 + 0.05):
                    self.lambda_z = torch.clamp(
                        self.lambda_z * (1 + 0.1), min=1e-8, max=1.0
                    )
                elif log_qpz.item() < self.fb_z * z.size(0) * z.size(1):
                    self.lambda_z = torch.clamp(
                        self.lambda_z * (1 - 0.1), min=1e-8, max=1.0
                    )

            log_pqz_M = self.lambda_z * torch.sum(pqz_all[XR.size(0) :])

        else:
            log_pqz_M = torch.sum(pqz_all[XR.size(0) :])

        final_rep = z if not self.use_plus else torch.cat([z, u], dim=1)
        sR = sR.repeat(final_rep.shape[0], 1)
        final_rep = torch.cat([sR, final_rep], dim=-1)[XR.size(0) :]

        final_rep1 = torch.relu(
            torch.einsum("ij,ijk->ik", final_rep, self.output_params[0])
        )
        final_rep1 = torch.einsum("ij,ijk->ik", final_rep1, self.output_params[2])
        mean_y, logstd_y = torch.split(final_rep1, 1, dim=1)
        # mean_y, logstd_y = torch.split(self.output(final_rep), 1, dim=1)
        logstd_y = torch.log(0.1 + 0.9 * F.softplus(logstd_y))

        mean_yM = mean_y
        logstd_yM = logstd_y

        # logp(R)

        # logp(M|S)
        pyM = Normal(mean_yM, logstd_yM)
        log_pyM = torch.sum(pyM.log_prob(yM))

        # obj_M = (log_pyM + log_pqz_M) / float(XM.size(0))
        return mean_yM, logstd_yM, log_pyM, log_pqz_M, pyM
    # 与forward功能一样，只用于测试阶段，不涉及梯度计算
    def predict(self, XR, x_new, sample=True):
        # sR = self.atten_ref(XR).mean(dim=0)
        sR = XR.mean(dim=0)
        H_all = self.cond_trans(torch.cat([XR, x_new], 0))

        # get U
        pu_mean_all, pu_logscale_all = torch.split(self.p_u(H_all), self.dim_u, dim=1)
        pu = Normal(pu_mean_all, pu_logscale_all)
        u = pu.rsample()

        A = sample_bipartite(
            u[XR.size(0) :], u[0 : XR.size(0)], self.pairwise_g, training=False
        )

        if self.add_atten:
            HR, HM = H_all[0 : XR.size(0)], H_all[XR.size(0) :]
            atten = self.atten_layer(HM, HR)
            A = A * atten

        pz_mean_all, pz_logscale_all = torch.split(
            self.q_z(H_all[0 : XR.size(0)]), self.dim_z, 1
        )

        pz_mean_all = torch.mm(self.norm_graph(A), pz_mean_all)
        pz_logscale_all = torch.mm(self.norm_graph(A), pz_logscale_all)
        pz = Normal(pz_mean_all, pz_logscale_all)

        z = pz.rsample()
        final_rep = z if not self.use_plus else torch.cat([z, u[XR.size(0) :]], dim=1)
        sR = sR.repeat(final_rep.shape[0], 1)
        final_rep = torch.cat([sR, final_rep], dim=-1)
        final_rep1 = torch.relu(
            torch.einsum("ij,ijk->ik", final_rep, self.output_params[0])
        )
        final_rep1 = torch.einsum("ij,ijk->ik", final_rep1, self.output_params[2])
        mean_y, logstd_y = torch.split(final_rep1, 1, dim=1)

        # mean_y, logstd_y = torch.split(self.output(final_rep), 1, dim=1)
        logstd_y = torch.log(0.1 + 0.9 * F.softplus(logstd_y))

        init_y = Normal(mean_y, logstd_y)
        if sample:
            y_new_i = init_y.sample()
        else:
            y_new_i = mean_y

        y_pred = y_new_i

        if self.transf_y is not None:
            if torch.cuda.is_available():
                y_pred = self.transf_y.inverse_transform(y_pred.cpu().data.numpy())
            else:
                y_pred = self.transf_y.inverse_transform(y_pred.data.numpy())

        return y_pred, mean_y, logstd_y, init_y


class SpaRegressionSepFNP(nn.Module):
    """
    Functional Neural Process for regression
    """

    def __init__(
        self,
        dim_x=1,        # 输入维度
        dim_y=1,        # 输出维度
        dim_h=50,       # 隐藏层维度
        transf_y=None,      # 输出数据的变换（如标准化）
        n_layers=1,     # 隐藏层的数量
        use_plus=True,      # 是否使用FNP+，即加入额外的潜变量u
        dim_u=1,        # u，z的维度
        dim_z=1,
        fb_z=0.0,
        use_ref_labels=False,
        use_DAG=False,
        add_atten=False,
        nodes=555,
    ):
        """
        :param dim_x: Dimensionality of the input
        :param dim_y: Dimensionality of the output
        :param dim_h: Dimensionality of the hidden layers
        :param transf_y: Transformation of the output (e.g. standardization)
        :param n_layers: How many hidden layers to use
        :param use_plus: Whether to use the FNP+
        :param num_M: How many points exist in the training set that are not part of the reference set
        :param dim_u: Dimensionality of the latents in the embedding space
        :param dim_z: Dimensionality of the  latents that summarize the parents
        :param fb_z: How many free bits do we allow for the latent variable z
        """
        super(SpaRegressionSepFNP, self).__init__()

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_h = dim_h
        self.dim_u = dim_u
        self.dim_z = dim_z
        self.use_plus = use_plus
        self.fb_z = fb_z
        self.transf_y = transf_y
        self.use_ref_labels = use_ref_labels
        self.use_DAG = use_DAG
        self.add_atten = add_atten
        self.nodes = nodes
        # normalizes the graph such that inner products correspond to averages of the parents
        self.norm_graph = lambda x: x / (torch.sum(x, 1, keepdim=True) + 1e-8)

        self.register_buffer("lambda_z", float_tensor(1).fill_(1e-8))

        # function that assigns the edge probabilities in the graph
        self.pairwise_g_logscale = nn.Parameter(
            float_tensor(1).fill_(math.log(math.sqrt(self.dim_u)))
        )
        self.pairwise_g = lambda x: logitexp(
            -0.5
            * torch.sum(
                torch.pow(x[:, self.dim_u :] - x[:, 0 : self.dim_u], 2), 1, keepdim=True
            )
            / self.pairwise_g_logscale.exp()
        ).view(x.size(0), 1)
        # transformation of the input

        init = [nn.Linear(dim_x, self.dim_h), nn.ReLU()]
        for i in range(n_layers - 1):
            init += [nn.Linear(self.dim_h, self.dim_h), nn.ReLU()]
        self.cond_trans = nn.Sequential(*init)
        # p(u|x)
        self.p_u = nn.Linear(self.dim_h, 2 * self.dim_u)
        # q(z|x)
        self.q_z = nn.Linear(self.dim_h, 2 * self.dim_z)
        # for p(z|A, XR, yR)
        if use_ref_labels:
            self.trans_cond_y = nn.Linear(self.dim_y, 2 * self.dim_z)

        # p(y|z) or p(y|z, u)
        # TODO: Add for sR input
        # import pdb
        # pdb.set_trace()

        self.atten_ref = SparseAttention(self.dim_x)
        self.output_params = [
            nn.Parameter(
                torch.randn(
                    self.nodes,
                    self.dim_z + self.dim_x
                    if not self.use_plus
                    else self.dim_z + self.dim_u + self.dim_x,
                    self.dim_h,
                ).to(device)
                / math.sqrt(self.dim_h)
            ),
            nn.Parameter(
                torch.randn(self.nodes, self.dim_h, self.dim_h).to(device)
                / math.sqrt(self.dim_h)
            ),
            nn.Parameter(
                torch.randn(self.nodes, self.dim_h, 2 * dim_y).to(device)
                / math.sqrt(self.dim_h)
            ),
        ]
        if self.add_atten:
            self.atten_layer = LatentAtten(self.dim_h)

    def forward(self, XR, XM, yM, kl_anneal=1.0):

        # import pdb
        # pdb.set_trace()

        # Ensure proper handling of the query, key, and value in attention
        # Modify the attention call to pass XR as query, key, and value
        sR = XR.mean(dim=0)

        X_all = torch.cat([XR, XM], dim=0)
        H_all = self.cond_trans(X_all)

        # Get U (latent variable)
        pu_mean_all, pu_logscale_all = torch.split(self.p_u(H_all), self.dim_u, dim=1)
        pu = Normal(pu_mean_all, pu_logscale_all)
        u = pu.rsample()

        # Get G (graph structure)
        if self.use_DAG:
            G = sample_DAG(u[0 : XR.size(0)], self.pairwise_g, training=self.training)
        else:
            G = sample_Clique(u[0 : XR.size(0)], self.pairwise_g, training=self.training)

        # Get A (adjacency matrix for bipartite graph)
        A = sample_bipartite(
            u[XR.size(0) :], u[0 : XR.size(0)], self.pairwise_g, training=self.training
        )
        if self.add_atten:
            HR, HM = H_all[0 : XR.size(0)], H_all[XR.size(0) :]
            atten = self.atten_layer(HM, HR)
            A = A * atten

        # Get Z (latent variables from posterior)
        qz_mean_all, qz_logscale_all = torch.split(self.q_z(H_all), self.dim_z, 1)
        qz = Normal(qz_mean_all, qz_logscale_all)
        z = qz.rsample()

        # Calculate p(z|A, XR, yR)
        pz_mean_all = torch.mm(self.norm_graph(torch.cat([G, A], dim=0)), qz_mean_all[0 : XR.size(0)])
        pz_logscale_all = torch.mm(self.norm_graph(torch.cat([G, A], dim=0)), qz_logscale_all[0 : XR.size(0)])

        pz = Normal(pz_mean_all, pz_logscale_all)
        pqz_all = pz.log_prob(z) - qz.log_prob(z)

        # Apply free bits for the latent z (if enabled)
        if self.fb_z > 0:
            log_qpz = -torch.sum(pqz_all)
            if self.training:
                if log_qpz.item() > self.fb_z * z.size(0) * z.size(1) * (1 + 0.05):
                    self.lambda_z = torch.clamp(self.lambda_z * (1 + 0.1), min=1e-8, max=1.0)
                elif log_qpz.item() < self.fb_z * z.size(0) * z.size(1):
                    self.lambda_z = torch.clamp(self.lambda_z * (1 - 0.1), min=1e-8, max=1.0)
            log_pqz_M = self.lambda_z * torch.sum(pqz_all[XR.size(0) :])
        else:
            log_pqz_M = torch.sum(pqz_all[XR.size(0) :])

        final_rep = z if not self.use_plus else torch.cat([z, u], dim=1)
        sR = sR.repeat(final_rep.shape[0], 1)
        final_rep = torch.cat([sR, final_rep], dim=-1)[XR.size(0) :]

        final_rep1 = torch.relu(torch.einsum("ij,ijk->ik", final_rep, self.output_params[0]))
        final_rep1 = torch.einsum("ij,ijk->ik", final_rep1, self.output_params[2])
        mean_y, logstd_y = torch.split(final_rep1, 1, dim=1)

        logstd_y = torch.log(0.1 + 0.9 * F.softplus(logstd_y))

        mean_yM = mean_y
        logstd_yM = logstd_y

        # Log-likelihood of the observed values
        pyM = Normal(mean_yM, logstd_yM)
        log_pyM = torch.sum(pyM.log_prob(yM))

        return mean_yM, logstd_yM, log_pyM, log_pqz_M, pyM
    # 与forward功能一样，只用于测试阶段，不涉及梯度计算
    def predict(self, XR, x_new, sample=True):
        # Ensure proper handling of the query, key, and value in attention
        # Modify the attention call to pass XR as query, key, and value
        sR = XR.mean(dim=0)

        H_all = self.cond_trans(torch.cat([XR, x_new], 0))

        # get U
        pu_mean_all, pu_logscale_all = torch.split(self.p_u(H_all), self.dim_u, dim=1)
        pu = Normal(pu_mean_all, pu_logscale_all)
        u = pu.rsample()

        A = sample_bipartite(
            u[XR.size(0) :], u[0 : XR.size(0)], self.pairwise_g, training=False
        )

        if self.add_atten:
            HR, HM = H_all[0 : XR.size(0)], H_all[XR.size(0) :]
            atten = self.atten_layer(HM, HR)
            A = A * atten

        pz_mean_all, pz_logscale_all = torch.split(
            self.q_z(H_all[0 : XR.size(0)]), self.dim_z, 1
        )

        pz_mean_all = torch.mm(self.norm_graph(A), pz_mean_all)
        pz_logscale_all = torch.mm(self.norm_graph(A), pz_logscale_all)
        pz = Normal(pz_mean_all, pz_logscale_all)

        z = pz.rsample()
        final_rep = z if not self.use_plus else torch.cat([z, u[XR.size(0) :]], dim=1)
        sR = sR.repeat(final_rep.shape[0], 1)
        final_rep = torch.cat([sR, final_rep], dim=-1)
        final_rep1 = torch.relu(
            torch.einsum("ij,ijk->ik", final_rep, self.output_params[0])
        )
        final_rep1 = torch.einsum("ij,ijk->ik", final_rep1, self.output_params[2])
        mean_y, logstd_y = torch.split(final_rep1, 1, dim=1)

        # mean_y, logstd_y = torch.split(self.output(final_rep), 1, dim=1)
        logstd_y = torch.log(0.1 + 0.9 * F.softplus(logstd_y))

        init_y = Normal(mean_y, logstd_y)
        if sample:
            y_new_i = init_y.sample()
        else:
            y_new_i = mean_y

        y_pred = y_new_i

        if self.transf_y is not None:
            if torch.cuda.is_available():
                y_pred = self.transf_y.inverse_transform(y_pred.cpu().data.numpy())
            else:
                y_pred = self.transf_y.inverse_transform(y_pred.data.numpy())

        return y_pred, mean_y, logstd_y, init_y


class Corem(nn.Module):
    def __init__(self, nodes: int, c: float = 5.0) -> None:
        super(Corem, self).__init__()
        self.nodes = nodes
        self.c = c
        self.w_hat = nn.Parameter(
            torch.randn(self.nodes).to(device) / math.sqrt(self.nodes) + c
        )
        self.w = nn.Linear(self.nodes, self.nodes)
        self.b = nn.Parameter(
            torch.randn(self.nodes).to(device) / math.sqrt(self.nodes) + c
        )
        self.v1 = nn.Linear(self.nodes, self.nodes)
        self.v2 = nn.Linear(self.nodes, self.nodes)

    def forward(self, mu, logstd, y):
        gamma = torch.sigmoid(self.w_hat)
        mu_final = gamma * mu + (1 - gamma) * self.w(mu)
        logstd_final = torch.sigmoid(self.b) * logstd + (
            1.0 - torch.sigmoid(self.b)
        ) * (self.v1(mu) + self.v2(logstd))
        py = Normal(mu_final[:, None], logstd_final[:, None])
        log_pyM = torch.sum(py.log_prob(y))
        return mu_final[:, None], logstd_final[:, None], log_pyM, py

    def predict(self, mu, logstd, sample=True):
        gamma = torch.sigmoid(self.w_hat)
        mu_final = gamma * mu + (1 - gamma) * self.w(mu)
        logstd_final = torch.sigmoid(self.b) * logstd + (
            1.0 - torch.sigmoid(self.b)
        ) * (self.v1(mu) + self.v2(logstd))
        py = Normal(mu_final[:, None], logstd_final[:, None])
        if sample:
            y_new_i = py.sample()
        else:
            y_new_i = mu_final[:, None]
        return y_new_i, mu_final[:, None], logstd_final[:, None], py

