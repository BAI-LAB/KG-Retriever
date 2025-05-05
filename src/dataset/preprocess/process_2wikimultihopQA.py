import re
from src.utils.lm_modeling import load_model, load_text2embedding
import torch
from torch_geometric.data.data import Data
import pandas as pd
import os
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer,util
from src.dataset import load_dataset
from torch.utils.data import DataLoader
import json
import pickle
import ast
import pyarrow.parquet as pq

path = '../2wikimultihopQA'

nodeslist = []
edgeslist = []
deslist = []

def textualize_graph(i,line):
    triples = re.findall(r'\((.*?)\)', line)
    nodes = {}
    edges = []
    des = {}
    for tri in triples:
        tri = tri.split(';')
        # remove error triples
        if len(tri) != 3:
            if len(tri) >= 3:
                tri[2] = ''.join(tri[2:len(tri)])
            if len(tri) < 3:
                continue

        src, edge_attr, dst = tri[0].strip(),tri[1].strip(),tri[2].strip()
        if src not in nodes:
            nodes[src] = len(nodes)
        if dst not in nodes:
            nodes[dst] = len(nodes)
        edges.append({'src': nodes[src], 'edge_attr': edge_attr.strip(), 'dst': nodes[dst], })
        des[str(nodes[src])+','+ str(nodes[dst])] = '(' + src +',' + edge_attr + ',' + dst + ')'

    nodes = pd.DataFrame(nodes.items(), columns=['node_attr', 'node_id'])
    edges = pd.DataFrame(edges)
    return nodes,edges,des

def step_one():
    os.makedirs(f'{path}/KG_QA_vRobert/des100', exist_ok=True)
    cnt = 1
    tmp_nodeslist = []
    tmp_edgeslist = []
    tmp_deslist = []
    for i in range(100):
        with open(f'{path}/tempfile2/{i}.json') as file:
            triples = json.load(file)
            for j in range(len(triples)):
                nodes,edges,des = textualize_graph(j,triples[str(j)])
                tmp_nodeslist.append(nodes)
                tmp_edgeslist.append(edges)
                tmp_deslist.append(des)

            nodeslist.append(tmp_nodeslist)
            edgeslist.append(tmp_edgeslist)
            with open(f'{path}/KG_QA_vRobert/des100/{i}.pkl', 'wb') as file:
                pickle.dump(tmp_deslist, file)
            cnt += 1
            tmp_nodeslist = []
            tmp_edgeslist = []
            tmp_deslist = []

def step_two():
    # encode KGs
    model = SentenceTransformer('../../../all-roberta-large-v1',device='cuda:1')

    os.makedirs(f'{path}/KG_QA_vRobert', exist_ok=True)
    os.makedirs(f'{path}/KG_QA_vRobert/graph100', exist_ok=True)

    for i in range(len(nodeslist)):
        os.makedirs(f'{path}/KG_QA_vRobert/graph100/{i}', exist_ok=True)
        for j in range(len(nodeslist[i])):
            nodes = nodeslist[i][j]
            edges = edgeslist[i][j]
            if not len(nodes) or not len(edges):
                print(i, j)
                continue

            node_des = nodes.node_attr.tolist()
            edge_des = edges.edge_attr.tolist()

            x = model.encode(node_des,convert_to_tensor = True)
            e = model.encode(edge_des,convert_to_tensor = True)
            edge_index = torch.LongTensor([edges.src, edges.dst])

            data = Data(x=x, edge_index=edge_index, edge_attr=e, num_nodes=len(nodes))

            torch.save(data, f'{path}/KG_QA_vRobert/graph100/{i}/{j}.pt')

def encode_question():
    model = SentenceTransformer('../../../all-roberta-large-v1')
    os.makedirs(f'{path}/questions_emb_vRobert', exist_ok=True)
    # 读取Parquet文件
    parquet_file = f'{path}/dev.parquet'
    table = pq.read_table(parquet_file)
    df = table.to_pandas()

    # 将DataFrame转换成JSON字符串
    json_data = df.to_json(orient='records')
    json_data = json.loads(json_data)
    for i in range(100):
        question = json_data[i]['question']

        question_emb = model.encode(question, convert_to_tensor=True)
        torch.save(question_emb, f'{path}/questions_emb_vRobert/{i}.pt')

def encode_doc():
    model = SentenceTransformer('../../../all-roberta-large-v1',device='cuda:1')

    os.makedirs(f'{path}/KG_QA_vRobert', exist_ok=True)
    os.makedirs(f'{path}/KG_QA_vRobert/doc_text', exist_ok=True)
    os.makedirs(f'{path}/KG_QA_vRobert/doc_emb', exist_ok=True)
    index = 0

    doc_text_list = []
    doc_emb_list = []

    parquet_file = f'{path}/dev.parquet'
    table = pq.read_table(parquet_file)
    df = table.to_pandas()

    # 将DataFrame转换成JSON字符串
    json_data = df.to_json(orient='records')
    json_data = json.loads(json_data)

    for i in range(100):
        paragraphs = ast.literal_eval(json_data[i]['context'])

        for j in range(len(paragraphs)):
            text = ''.join(paragraphs[j][1])
            doc_emb = model.encode(text, convert_to_tensor=True)
            doc_emb_list.append(doc_emb)
            doc_text_list.append(text)

    torch.save(torch.stack(doc_emb_list), f'{path}/KG_QA_vRobert/doc_emb/{100}.pt')
    with open(f'{path}/KG_QA_vRobert/doc_text/{100}.pkl', 'wb') as file:
        pickle.dump(doc_text_list, file)

if __name__ == '__main__':
    step_one()
    step_two()
    # encode_question()
    # encode_doc()
