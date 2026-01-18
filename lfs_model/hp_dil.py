from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import einops
import torch_geometric.utils
from torch.nn import Embedding, BatchNorm1d, Linear, ModuleList, Sequential, ReLU
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GPSConv
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.utils import to_dense_adj, to_undirected, remove_self_loops, subgraph
from torch import Tensor

from lfs_model.point_net import PCModel


class RedrawProjection:
    """
    https://pytorch-geometric.readthedocs.io/en/latest/tutorial/graph_transformer.html?highlight=transformer#positional-and-structural-encodings
    """
    def __init__(self, model: torch.nn.Module,
                redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1


class HabitatInteraction(torch.nn.Module):
    def __init__(self, num_habitat, num_cls, channels: int, pe_dim: int, num_layers: int,
                attn_type: str, attn_kwargs: Dict[str, Any]):
        super().__init__()

        # self.node_emb = Embedding(28, channels - pe_dim)
        self.num_cls = num_cls
        self.pe_lin = Sequential(
            Embedding(num_habitat+1, 20),
            BatchNorm1d(20),
            Linear(20, pe_dim),
        )
        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            conv = GPSConv(channels, GINEConv(nn, edge_dim=1), heads=4,
                        attn_type=attn_type, attn_kwargs=attn_kwargs)
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(channels, channels // 2),
            ReLU(),
            Linear(channels // 2, channels // 4),
            ReLU(),
            Linear(channels // 4, self.num_cls),
        )
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)
        self.mse_loss = torch.nn.MSELoss()
        self.lin1 = Linear(channels, channels)
        self.lin2 = Linear(channels, self.num_cls)

    def entropy_regularizer(self, S, eps=1e-8):
        p = S.clamp(min=eps)
        ent = -(p * torch.log(p)).sum(dim=1)
        return ent.mean()

    # def aggregate(self, assignment, x, batch, edge_index):
    #     # add sparse loss
    #     row_sparsity_loss = assignment.abs().mean()  # 行稀疏
    #     size0_loss = (assignment[:, 0].mean() - 0.2).pow(2)  # 0号簇大小
    #
    #     max_id = torch.max(batch)
    #     if torch.cuda.is_available():
    #         EYE = torch.ones(2).cuda()
    #     else:
    #         EYE = torch.ones(2)
    #     all_adj = to_dense_adj(edge_index)[0]
    #     all_pos_penalty, total_ent = 0, 0
    #     all_graph_embedding = []
    #     all_pos_embedding = []
    #     st = 0
    #     end = 0
    #     for i in range(int(max_id + 1)):
    #         j = 0
    #         # while batch[st + j] == i and st + j <= len(batch) - 2:
    #         while st + j < len(batch) and batch[st + j] == i:
    #             j += 1
    #         end = st + j
    #         one_batch_x = x[st:end]
    #         one_batch_assignment = assignment[st:end]
    #         entropy_i = self.entropy_regularizer(one_batch_assignment)
    #         total_ent += entropy_i
    #         group_features = torch.mm(torch.t(one_batch_assignment), one_batch_x)
    #         pos_embedding = group_features[0].unsqueeze(dim=0)
    #         Adj = all_adj[st:end,st:end]
    #         new_adj = torch.mm(torch.t(one_batch_assignment), Adj)
    #         new_adj = torch.mm(new_adj, one_batch_assignment)
    #         normalize_new_adj = F.normalize(new_adj, p=1, dim=1, eps = 0.00001)
    #         norm_diag = torch.diag(normalize_new_adj)
    #         pos_penalty = self.mse_loss(norm_diag, EYE)
    #         graph_embedding = one_batch_x.mean(0, keepdim=True)
    #         all_pos_embedding.append(pos_embedding)
    #         all_graph_embedding.append(graph_embedding)
    #         all_pos_penalty = all_pos_penalty + pos_penalty
    #         st = end
    #     all_pos_embedding = torch.cat(tuple(all_pos_embedding), dim=0)
    #     all_graph_embedding = torch.cat(tuple(all_graph_embedding), dim=0)
    #     all_pos_penalty = all_pos_penalty / (max_id + 1)
    #     external_losses = total_ent / (max_id + 1) + row_sparsity_loss + size0_loss + all_pos_penalty
    #     return all_pos_embedding,all_graph_embedding, all_pos_penalty, external_losses
    def aggregate(self, assignment, x, batch, edge_index):
        row_sparsity_loss = assignment.abs().mean()
        size0_loss = (assignment[:, 0].mean() - 0.2).pow(2)
        x_dense, mask = torch_geometric.utils.to_dense_batch(x, batch)
        assign_dense, _ = torch_geometric.utils.to_dense_batch(assignment, batch)
        adj_dense = torch_geometric.utils.to_dense_adj(edge_index, batch)
        eps = 1e-8
        p = assign_dense.clamp(min=eps)
        ent_per_node = -(p * torch.log(p)).sum(dim=2)  # [B, N_max]
        total_ent = (ent_per_node * mask).sum() / mask.sum().clamp(min=1)
        group_features = torch.bmm(assign_dense.transpose(1, 2), x_dense)
        all_pos_embedding = group_features[:, 0, :]  # [B, C]
        all_graph_embedding = (x_dense * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        temp = torch.bmm(assign_dense.transpose(1, 2), adj_dense)
        new_adj = torch.bmm(temp, assign_dense)
        normalize_new_adj = F.normalize(new_adj, p=1, dim=1, eps=0.00001)
        norm_diag = torch.diagonal(normalize_new_adj, dim1=1, dim2=2)  # [B, K]
        if torch.cuda.is_available():
            EYE = torch.ones(2).to(x.device)
        else:
            EYE = torch.ones(2)
        pos_penalty = self.mse_loss(norm_diag[:, :2], EYE.expand(norm_diag.size(0), -1))
        external_losses = total_ent + row_sparsity_loss + size0_loss + pos_penalty
        return all_pos_embedding, all_graph_embedding, pos_penalty, external_losses


    def forward(self, x, pe, edge_index, edge_attr, batch, assignment):
        # pe = pe + 1    # -1（padding）-> 0，for valid Embedding
        x_pe = self.pe_lin(pe)
        x = torch.cat((x, x_pe), 1)
        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)    # [B * num_habitat, embedding_dim]
        all_pos_embedding, all_graph_embedding, all_pos_penalty, ent_penalty = self.aggregate(assignment, x, batch, edge_index)
        num_per_graph = torch.bincount(batch)
        assign_list = torch.split(assignment, num_per_graph.tolist())
        x = F.relu(self.lin1(all_pos_embedding))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x, all_pos_embedding, all_graph_embedding, all_pos_penalty, ent_penalty, assign_list    # [b,1], [b, 128], [b, 128], [losses,..]
        # pred, pos_embed, all_embed, loss_conn, loss_ent, node_select, node_choice


def batch_cosine_similarity(x):
    x_norm = F.normalize(x, p=2, dim=-1)
    sim_matrix = torch.bmm(x_norm, x_norm.transpose(1, 2))
    return sim_matrix


def weight_sim_dist(edge_index, dist_w, sim_mat, batch, habitat_order, eps=1e-6):
    src, dst = edge_index
    b = batch[src]
    h_src = habitat_order[src]
    h_dst = habitat_order[dst]
    sim = sim_mat[b, h_src, h_dst].clamp(min=0)
    sigma = dist_w.std().clamp_min(eps)
    weight = sim * torch.exp(-(dist_w ** 2) / (sigma ** 2 + eps))
    return weight


def build_graph(x_node, dist_mat : Tensor, top_k=None, x_add=None, mask=None):
    """
        Build a graph from the distance matrix and node features.
    """
    B, H, _ = dist_mat.shape
    device = dist_mat.device
    if mask is None:
        mask = (dist_mat.abs().sum(dim=2) > 1e-6)
    exist_mask = mask
    node_valid_pair = exist_mask.unsqueeze(2) & exist_mask.unsqueeze(1)
    diag_eye = torch.eye(H, dtype=torch.bool, device=device).unsqueeze(0)
    edge_mask = (dist_mat >= 0) & ~diag_eye & node_valid_pair  # [B,H,H], NO SELF-LOOP
    # filter top-k edges :-) (NO USE MAYBE)
    if top_k is not None:
        masked_dist = dist_mat.masked_fill(~edge_mask, 1e9)
        topk_idx = torch.topk(masked_dist, top_k, largest=False).indices  # (B,H,k)
        new_edge_mask = torch.zeros_like(edge_mask, dtype=torch.bool)
        b_arange = torch.arange(B, device=device)[:, None, None]
        r_arange = torch.arange(H, device=device)[None, :, None]
        new_edge_mask[b_arange, r_arange, topk_idx] = True
        edge_mask = new_edge_mask & node_valid_pair
    hab_mat = torch.arange(H, device=device).expand(B, H).clone()
    hab_mat[~exist_mask] = -1
    gid = torch.arange(B * H, device=device).view(B, H)
    src = gid[:, :, None].expand(-1, -1, H)[edge_mask]
    dst = gid[:, None, :].expand(-1, H, -1)[edge_mask]
    w = dist_mat[edge_mask]
    edge_index = torch.stack([src, dst], 0)
    edge_attr = w
    edge_index, edge_attr = to_undirected(edge_index, edge_attr, reduce='mean')
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    flat_exist = exist_mask.view(-1)
    batch_full = torch.arange(B, device=device).repeat_interleave(H)
    x_node = x_node[flat_exist]
    if x_add is not None:
        x_add = x_add[flat_exist]
    batch = batch_full[flat_exist]
    hab_mat = hab_mat.view(-1)[flat_exist]
    edge_index, edge_attr = subgraph(
        flat_exist, edge_index, edge_attr=edge_attr, relabel_nodes=True)
    return x_node, edge_index.long(), edge_attr, hab_mat, batch, x_add


class HabitatGraph(nn.Module):
    def __init__(self, feat_dim, num_habitat, pe_dim, num_classes=2, in_channel=7):
        super().__init__()
        self.inchannel = in_channel
        self.pe_dim = int(pe_dim)
        self.pc_hfe = PCModel(out_dim=feat_dim, num_add_cls=2, normal_channel=True, in_channel=in_channel)
        self.feat_dim = feat_dim
        self.edge_type = 2
        self.num_learning_path = 4
        self.inner_dim = 512
        self.num_cls = num_classes
        self.num_habitat = num_habitat
        self.num_layers = 3
        self.all_dim = int(self.feat_dim + self.pe_dim)
        self.gnn_model = HabitatInteraction(num_habitat=num_habitat, num_cls=num_classes,
                                            channels=self.all_dim, pe_dim=self.pe_dim, num_layers=2,
                                            attn_type='performer', attn_kwargs={})


    def forward(self, x_pcs, x_dist_mat, mask=None):
        """
        :param x_dist_mat: [batch, num_habitats, num_habitats]
        :param x_pcs: [batch, num_habitats, num_points, shallow_dims]
        :return:
        """
        # [b(num_p), num_clouds, num_point, channel] ---pc high-level FE---> [ num_p*num_clouds, num_features]
        # ----build graph----> graph: num_p * [num_clouds, num_features] --graph_pooling--> [num_p, num_features]
        # ----cls---> [num_p, num_cls] (yes or no)
        B, H = x_dist_mat.shape[:2]
        x_pcs, assignment = self.pc_hfe(x_pcs)  # [num_p*num_clouds, num_features]

        # make edge embeddings
        sim_mat = einops.rearrange(x_pcs, '(b h) n -> b h n', b=B, h=H, n=self.feat_dim)  # [B, H, num_features, num_points]
        sim_mat = batch_cosine_similarity(sim_mat)  # [B, num_h, num_h]
        x_pcs, edge_index, edge_w, habitat_order, batch, assignment  = build_graph(x_pcs, x_dist_mat, top_k=None, x_add=assignment, mask=mask)  # [num_p*num_clouds, num_features]
        edge_w = weight_sim_dist(
            edge_index=edge_index,
            dist_w=edge_w,
            sim_mat=sim_mat,
            batch=batch,
            habitat_order=habitat_order
        ).unsqueeze(-1)
        pred, pos_embed, all_embed, loss_conn, loss_ent, node_choice = self.gnn_model(x=x_pcs, pe=habitat_order, edge_index=edge_index, edge_attr=edge_w, batch=batch, assignment=assignment)
        return pred, pos_embed, all_embed, loss_conn, loss_ent, node_choice


class Discriminator(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super(Discriminator, self).__init__()

        self.input_size = 2 * hidden_size
        self.hidden_size = hidden_size
        self.lin1 = torch.nn.Linear(self.input_size,self.hidden_size)
        self.lin2 = torch.nn.Linear(self.hidden_size, 1)
        self.relu = torch.nn.ReLU()


    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()


    def forward(self, embeddings, positive):
        cat_embeddings = torch.cat((embeddings, positive),dim = -1)
        pre = self.relu(self.lin1(cat_embeddings))
        pre = self.relu(self.lin2(pre))
        return pre


def MI_Est(discriminator, embeddings, positive):

    batch_size = embeddings.shape[0]

    shuffle_embeddings = embeddings[torch.randperm(batch_size)]

    joint = discriminator(embeddings,positive)

    margin = discriminator(shuffle_embeddings,positive)

    #Donsker
    mi_est = torch.mean(joint) - torch.clamp(torch.log(torch.mean(torch.exp(margin))),-100000,100000)
    #JSD
    #mi_est = -torch.mean(F.softplus(-joint)) - torch.mean(F.softplus(-margin)+margin)
    #x^{2}
    #mi_est = torch.mean(joint**2) - 0.5* torch.mean((torch.sqrt(margin**2)+1.0)**2)
    return mi_est


if __name__ == "__main__":
    input = torch.randn(3, 15, 1024, 5).cuda()
    dist_mat = torch.randn(3, 15, 15).cuda()
    model = HabitatGraph(feat_dim=120, pe_dim=8, num_habitat=15, in_channel=2).cuda()
    outputs = model(input, dist_mat)
    pred = outputs[0]
    pos_embed = outputs[1]
    print("Pred shape:", pred.shape)
    print("Pos Embed shape:", pos_embed.shape)
    pass



