import re
import torch
from torch_geometric.data.data import Data
import pandas as pd
import os
from sentence_transformers import SentenceTransformer,util
import json
import pickle

path = '../CRUD'

nodeslist = []
edgeslist = []

def textualize_graph(i,line):
    triples = re.findall(r'\((.*?)\)', line)
    nodes = {}
    edges = []
    des = {}
    for tri in triples:
        tri = tri.split(';')

        while len(tri) < 3:
            tri.append(tri[0])
        if len(tri) > 3:
            tri[2] = ''.join(tri[2:len(tri)])

        src, edge_attr, dst = tri[0].strip(),tri[1].strip(),tri[2].strip()
        if src not in nodes:
            nodes[src] = len(nodes)
        if dst not in nodes:
            nodes[dst] = len(nodes)
        edges.append({'src': nodes[src], 'edge_attr': edge_attr.strip(), 'dst': nodes[dst], })
        des[str(nodes[src])+','+ str(nodes[dst])] = '(' + src +',' + edge_attr + ',' + dst + ')'

    with open(f'{path}/KG_QA2_vBGE/des/{i}.json', 'w') as file:
        json.dump(des, file, ensure_ascii=False)

    nodes = pd.DataFrame(nodes.items(), columns=['node_attr', 'node_id'])
    edges = pd.DataFrame(edges)
    return nodes,edges

def step_one():
    # generate knowledge graphs and des
    os.makedirs(f'{path}/KG_QA2_vBGE/des', exist_ok=True)

    with open(f'{path}/QA2_triples.txt', 'r') as file:
        for i,line in enumerate(file):
            nodes, edges = textualize_graph(i,line)
            nodeslist.append(nodes)
            edgeslist.append(edges)

def step_two():
    # encode KGs
    model = SentenceTransformer('../../../bge-base-zh-v1.5')

    os.makedirs(f'{path}/KG_QA2_vBGE', exist_ok=True)
    os.makedirs(f'{path}/KG_QA2_vBGE/graph', exist_ok=True)

    for i in range(len(nodeslist)):
        nodes = nodeslist[i]
        edges = edgeslist[i]
        if not len(nodes) or not len(edges):
            continue

        node_des = nodes.node_attr.tolist()
        edge_des = edges.edge_attr.tolist()

        x = model.encode(node_des,convert_to_tensor = True)
        e = model.encode(edge_des,convert_to_tensor = True)
        edge_index = torch.LongTensor([edges.src, edges.dst])

        data = Data(x=x, edge_index=edge_index, edge_attr=e, num_nodes=len(nodes))

        torch.save(data, f'{path}/KG_QA2_vBGE/graph/{i}.pt')

    # encode questions
    os.makedirs(f'{path}/questions_emb_QA2_vBGE', exist_ok=True)

    data_path = '../CRUD/QA2.json'
    with open(data_path, 'r') as file:
        data = json.load(file)
        data = data['results']

    for i in range(len(data)):
        question = data[i]["questions"]
        question_emb = model.encode(question, convert_to_tensor=True)
        torch.save(question_emb, f'{path}/questions_emb_QA2_vBGE/{i}.pt')

def encode_doc():
    data_path = '../CRUD/QA2.json'
    model = SentenceTransformer('../../../bge-base-zh-v1.5',device='cuda:1')

    os.makedirs(f'{path}/KG_QA2_vBGE', exist_ok=True)
    os.makedirs(f'{path}/KG_QA2_vBGE/doc_text', exist_ok=True)
    os.makedirs(f'{path}/KG_QA2_vBGE/doc_emb', exist_ok=True)

    with open(data_path, 'r') as file:
        data = json.load(file)
        data = data['results']

    doc_text_list = []
    doc_emb_list = []
    for i in range(len(data)):
        new = data[i]["news2"]
        doc_text_list.append(new)
        doc_emb_list.append(model.encode(new, convert_to_tensor=True))


    torch.save(doc_emb_list, f'{path}/KG_QA2_vBGE/doc_emb/{3000}.pt')
    with open(f'{path}/KG_QA2_vBGE/doc_text/{3000}.pkl', 'wb') as file:
        pickle.dump(doc_text_list, file)

if __name__ == '__main__':
    # step_one()
    # step_two()
    encode_doc()
