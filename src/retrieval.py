import torch.nn.functional as F
import numpy as np
import torch
from torch_geometric.data.data import Data
from collections import OrderedDict
import networkx as nx


def retrieval_func(questions_emb,SG_lists,KG_lists,SG_edges,des,device,topk_n=7,topk_e=3,intervals=0.40):
    questions_emb = torch.stack(questions_emb).to(device)
    SG_lists = torch.stack(SG_lists).to(device)
    inner_product = torch.matmul(questions_emb, SG_lists.T)
    l2_norm1 = torch.norm(questions_emb, p=2, dim=1, keepdim=True)
    l2_norm2 = torch.norm(SG_lists, p=2, dim=1, keepdim=True)
    sg_prizes = inner_product / (l2_norm1 * l2_norm2.T)

    # topk_sg_values, topk_sg_indices = torch.topk(sg_prizes, 5, largest=True)
    # neighbors = SG_edges.T[topk_sg_indices.cpu().numpy()]
    # dic = OrderedDict()
    # for i in range(len(neighbors)):
    #     for edge in neighbors[i].cpu().numpy():
    #         if edge[0] not in dic.keys():
    #             dic[edge[0]] = sg_prizes[i][edge[0]]
    #         else:
    #             dic[edge[0]] += sg_prizes[i][edge[0]]
    #
    #         if edge[1] not in dic.keys():
    #             dic[edge[1]] = sg_prizes[i][edge[1]] * 0.5
    #         else:
    #             dic[edge[1]] += sg_prizes[i][edge[1]] * 0.5
    # target = list(dic.items())[0][0]

    # topk_sg_values, topk_sg_indices = torch.topk(sg_prizes, 1, largest=True)
    # topk_sg_values, topk_sg_indices = torch.topk(sg_prizes, 3, largest=True)
    # topk_sg_values, topk_sg_indices = torch.topk(sg_prizes, 5, largest=True)
    topk_sg_values, topk_sg_indices = torch.topk(sg_prizes, 7, largest=True)
    # topk_sg_values, topk_sg_indices = torch.topk(sg_prizes, 10, largest=True)
    selected_SG = topk_sg_indices.cpu().tolist()

    neighbors = SG_edges.T[topk_sg_indices.cpu().numpy()]
    for i in range(len(neighbors)):
        for edge in neighbors[i]:
            if int(edge[1]) not in selected_SG[i]:
                selected_SG[i].append(int(edge[1]))

    ret_data = []
    ret_mask = []
    ret_des = []
    for i in range(len(selected_SG)):
        KGs_x = [KG_lists[j].x for j in selected_SG[i]]
        KGs_edge_attr = [KG_lists[j].edge_attr for j in selected_SG[i]]
        KGs_edge_index = [KG_lists[j].edge_index for j in selected_SG[i]]
        KGs_numnodes = [KG_lists[j].num_nodes for j in selected_SG[i]]

        inner_product = torch.matmul(questions_emb[i], torch.cat(KGs_x, dim=0).T.to(device))
        l2_norm1 = torch.norm(questions_emb[i], p=2, dim=0, keepdim=True)
        l2_norm2 = torch.norm(torch.cat(KGs_x, dim=0).to(device), p=2, dim=1, keepdim=True).squeeze()
        n_prizes = inner_product / (l2_norm1 * l2_norm2)
        topk_n = min(len(n_prizes), topk_n)
        topk_n_values, topk_n_indices = torch.topk(n_prizes,topk_n , largest=True)
        selected_node = []
        for j in range(len(topk_n_indices)):
            if topk_n_values[j] > intervals:
                selected_node.append(topk_n_indices[j])

        pedal = [0]
        for j in range(len(KGs_numnodes)-1):
            pedal.append(pedal[-1]+KGs_numnodes[j])

        e = []
        e_attr = []
        tmp_des = []
        for j in range(len(KGs_edge_index)):
            target_des = des[int(selected_SG[i][j])]
            if not target_des:
                target_des = des[int(selected_SG[i][j])-1]
            for key,values in enumerate(KGs_edge_index[j].T):
                if (values[0] + pedal[j] in selected_node or values[1] + pedal[j] in selected_node) and str(int(values[0]))+','+ str(int(values[1])) in target_des.keys():
                    e.append(values)
                    e_attr.append(KGs_edge_attr[j][key])
                    tmp_des.append(target_des[str(int(values[0]))+','+ str(int(values[1]))])
        ret_des.append(','.join(tmp_des))

    return ret_data, selected_SG,ret_mask,ret_des

def retrieval_func_two_stage(questions_emb,doc_embed_tensor,KG_lists,SG_edges,des,device,topk_n=7,topk_e=3,intervals=0.40):
    questions_emb = torch.stack(questions_emb).to(device)
    SG_lists = doc_embed_tensor.to(device)
    inner_product = torch.matmul(questions_emb, SG_lists.T)
    l2_norm1 = torch.norm(questions_emb, p=2, dim=1, keepdim=True)
    l2_norm2 = torch.norm(SG_lists, p=2, dim=1, keepdim=True)
    sg_prizes = inner_product / (l2_norm1 * l2_norm2.T)

    # topk_sg_values, topk_sg_indices = torch.topk(sg_prizes, 1, largest=True)
    topk_sg_values, topk_sg_indices = torch.topk(sg_prizes, 3, largest=True)
    # topk_sg_values, topk_sg_indices = torch.topk(sg_prizes, 5, largest=True)
    # topk_sg_values, topk_sg_indices = torch.topk(sg_prizes, 7, largest=True)
    # topk_sg_values, topk_sg_indices = torch.topk(sg_prizes, 10, largest=True)
    selected_SG = topk_sg_indices.cpu().tolist()

    neighbors = SG_edges.T[topk_sg_indices.cpu().numpy()]
    for i in range(len(neighbors)):
        for edge in neighbors[i]:
            for j in range(1,len(edge)):
                if int(edge[j]) not in selected_SG[i]:
                    selected_SG[i].append(int(edge[j]))

    ret_data = []
    ret_mask = []
    ret_des = []
    for i in range(len(selected_SG)):
        KGs_x = [KG_lists[j].x for j in selected_SG[i]]
        KGs_edge_attr = [KG_lists[j].edge_attr for j in selected_SG[i]]
        KGs_edge_index = [KG_lists[j].edge_index for j in selected_SG[i]]
        KGs_numnodes = [KG_lists[j].num_nodes for j in selected_SG[i]]

        inner_product = torch.matmul(questions_emb[i], torch.cat(KGs_x, dim=0).T.to(device))
        l2_norm1 = torch.norm(questions_emb[i], p=2, dim=0, keepdim=True)
        l2_norm2 = torch.norm(torch.cat(KGs_x, dim=0).to(device), p=2, dim=1, keepdim=True).squeeze()
        n_prizes = inner_product / (l2_norm1 * l2_norm2)
        topk_n = min(len(n_prizes), topk_n)
        topk_n_values, topk_n_indices = torch.topk(n_prizes, topk_n, largest=True)
        selected_node = []
        for j in range(len(topk_n_indices)):
            if topk_n_values[j] > intervals:
                selected_node.append(topk_n_indices[j])

        pedal = [0]
        for j in range(len(KGs_numnodes) - 1):
            pedal.append(pedal[-1] + KGs_numnodes[j])

        e = []
        e_attr = []
        tmp_des = []
        for j in range(len(KGs_edge_index)):
            target_des = des[int(selected_SG[i][j])]
            if not target_des:
                target_des = des[int(selected_SG[i][j]) - 1]
            for key, values in enumerate(KGs_edge_index[j].T):
                if (values[0] + pedal[j] in selected_node or values[1] + pedal[j] in selected_node) and str(
                        int(values[0])) + ',' + str(int(values[1])) in target_des.keys():
                    e.append(values)
                    e_attr.append(KGs_edge_attr[j][key])
                    tmp_des.append(target_des[str(int(values[0])) + ',' + str(int(values[1]))])
        ret_des.append(','.join(tmp_des))

    return ret_data, selected_SG, ret_mask, ret_des

def retrieval_func_with_att(questions_emb,doc_embed_tensor,KG_lists,SG,des,device,topk_n=7,topk_e=3,intervals=0.40):
    questions_emb = torch.stack(questions_emb).to(device)
    SG_lists = doc_embed_tensor.to(device)
    inner_product = torch.matmul(questions_emb, SG_lists.T)
    l2_norm1 = torch.norm(questions_emb, p=2, dim=1, keepdim=True)
    l2_norm2 = torch.norm(SG_lists, p=2, dim=1, keepdim=True)
    sg_prizes = inner_product / (l2_norm1 * l2_norm2.T)

    k = 4
    topk_sg_values, topk_sg_indices = torch.topk(sg_prizes, k, largest=True)

    selected_SG = topk_sg_indices.cpu().tolist()
    weight_list = [1]*k

    new_node = []
    for node in selected_SG[0]:
        for neib in list(nx.neighbors(SG, node)):  # find 1_th neighbors
            if neib not in selected_SG[0] and neib not in new_node:
                new_node.append(neib)
                weight_list.append(SG[node][neib]['weight'])
    new_new_node = new_node.copy()
    # for j in range(len(new_node)):
    #     for neib in list(nx.neighbors(SG, new_node[j])):  # find 2_th neighbors
    #         if neib not in selected_SG[0] and neib not in new_node and neib not in new_new_node:
    #             new_new_node.append(neib)
    #             weight_list.append(SG[new_node[j]][neib]['weight']*weight_list[k+j])
    selected_SG[0] += new_new_node

    # for i in range(1):
    #     new_node = []
    #     for node in selected_SG[0]:
    #         for j in dic[node]:
    #             if j not in selected_SG[0] and j not in new_node:
    #                 new_node.append(j)
    #     selected_SG[0] += new_node
    ret_data = []
    ret_mask = []
    ret_des = []
    for i in range(len(selected_SG)):
        KGs_x = [KG_lists[j].x for j in selected_SG[i]]
        KGs_edge_attr = [KG_lists[j].edge_attr for j in selected_SG[i]]
        KGs_edge_index = [KG_lists[j].edge_index for j in selected_SG[i]]
        KGs_numnodes = [KG_lists[j].num_nodes for j in selected_SG[i]]

        attention = []
        for j in range(len(weight_list)):
            attention += [weight_list[j]] * KGs_numnodes[j]

        inner_product = torch.matmul(questions_emb[i], torch.cat(KGs_x, dim=0).T.to(device))
        l2_norm1 = torch.norm(questions_emb[i], p=2, dim=0, keepdim=True)
        l2_norm2 = torch.norm(torch.cat(KGs_x, dim=0).to(device), p=2, dim=1, keepdim=True).squeeze()
        n_prizes = inner_product / (l2_norm1 * l2_norm2)
        topk_n = min(len(n_prizes), topk_n)
        topk_n_values, topk_n_indices = torch.topk(torch.stack([n_prizes[j] * attention[j] for j in range(len(attention))]), topk_n, largest=True)
        selected_node = []
        for j in range(len(topk_n_indices)):
            if topk_n_values[j] > intervals:
                selected_node.append(topk_n_indices[j])

        pedal = [0]
        for j in range(len(KGs_numnodes) - 1):
            pedal.append(pedal[-1] + KGs_numnodes[j])

        e = []
        e_attr = []
        tmp_des = []
        for j in range(len(KGs_edge_index)):
            target_des = des[int(selected_SG[i][j])]
            if not target_des:
                target_des = des[int(selected_SG[i][j]) - 1]
            for key, values in enumerate(KGs_edge_index[j].T):
                if (values[0] + pedal[j] in selected_node or values[1] + pedal[j] in selected_node) and str(
                        int(values[0])) + ',' + str(int(values[1])) in target_des.keys():
                    e.append(values)
                    e_attr.append(KGs_edge_attr[j][key])
                    tmp_des.append(target_des[str(int(values[0])) + ',' + str(int(values[1]))])
        ret_des.append(','.join(tmp_des))

    return ret_data, selected_SG, ret_mask, ret_des


def retrieval_func_via_doc(questions_emb,doc_embed_list,doc_text_list,device,topk_n=7,topk_e=3,intervals=0.44,SG_edges=None):
    # questions_emb 和 graph_emb之间有gap啊
    questions_emb = torch.stack(questions_emb).to(device)
    # doc_embed = torch.stack(doc_embed_list).to(device)
    doc_embed = doc_embed_list.to(device)
    inner_product = torch.matmul(questions_emb, doc_embed.T)
    l2_norm1 = torch.norm(questions_emb, p=2, dim=1, keepdim=True)
    l2_norm2 = torch.norm(doc_embed, p=2, dim=1, keepdim=True)
    sg_prizes = inner_product / (l2_norm1 * l2_norm2.T)

    topk_sg_values, topk_sg_indices = torch.topk(sg_prizes, 1, largest=True)
    # topk_sg_values, topk_sg_indices = torch.topk(sg_prizes, 3, largest=True)
    #topk_sg_values, topk_sg_indices = torch.topk(sg_prizes, 5, largest=True)
    selected_SG = topk_sg_indices.cpu().tolist()

    # neighbors = SG_edges.T[topk_sg_indices.cpu().numpy()]
    # for i in range(len(neighbors)):
    #     for edge in neighbors[i]:
    #         if int(edge[1]) not in selected_SG[i]:
    #             selected_SG[i].append(int(edge[1]))

    ret_data = []
    ret_mask = []
    ret_des = []
    for i in range(len(selected_SG)):
        e = []
        e_attr = []
        tmp_des = []
        for j in range(len(selected_SG[i])):
            target_des = doc_text_list[int(selected_SG[i][j])]
            tmp_des.append(target_des)
        ret_des.append(','.join(tmp_des))

    return ret_data, selected_SG,ret_mask,ret_des

