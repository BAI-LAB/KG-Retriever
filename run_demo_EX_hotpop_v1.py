from src.config import parse_args_llama
import torch
import gc
from src.model import load_model, llm_model_path
from src.dataset import load_dataset
from torch.utils.data import DataLoader
from src.retrieval import retrieval_func_two_stage,retrieval_func_via_doc,retrieval_func_with_att
import os
import pickle
import datetime
import networkx as nx

def main(args):
    batch_size = 1
    n = 0
    break_p = xxx

    # STEP 1: Data preprossion    Constructing KG and Encoding KG

    args.llm_model_path = llm_model_path[args.llm_model_name]
    model = load_model[args.model_name](args=args)

    test_dataset_path = './src/dataset/HOTPOPQA/hotpot_dev_fullwiki_v1.json'
    test_dataset = load_dataset['HOTPOPQA'](test_dataset_path,size1=n,size2=break_p)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, pin_memory=True, shuffle=False)

    # Load questions emb
    questions_emb_list = []
    for i in range(break_p):
        question_emb = torch.load(f"./src/dataset/HOTPOPQA/questions_emb_vRobert/{i}.pt")
        questions_emb_list.append(question_emb)


    # Calculate node emb of SG
    KG_lists = []

    test_folder_path = "./src/dataset/HOTPOPQA/KG_QA_vRobert"

    for i in range(n,break_p):
        KG_path = f'{test_folder_path}/graph/{i}/'
        if os.path.exists(KG_path):
            for j in range(20):
                if os.path.exists(KG_path + f'{j}.pt'):
                    KG = torch.load(KG_path + f'{j}.pt')
                    KG_lists.append(KG)
                else:
                    break
        else:
            KG_lists.append(KG)

    doc_embed_tensor = torch.load(f"./src/dataset/HOTPOPQA/KG_QA_vRobert/doc_emb/{xxx}.pt")
    doc_embed_tensor = doc_embed_tensor[:len(KG_lists)]


    SG_lists_tensor = doc_embed_tensor
    inner_product = torch.matmul(SG_lists_tensor, SG_lists_tensor.T)
    l2_norm = torch.norm(SG_lists_tensor, p=2, dim=1, keepdim=True)
    n_prizes = inner_product / (l2_norm * l2_norm.T)
    topk_n_values, topk_n_indices = torch.topk(n_prizes, 3, largest=True)
    SG_edges = topk_n_indices.T
    # edge_attr = [topk_n_values[i][1] for i in range(len(topk_n_indices))]

    SG = nx.Graph()
    for i in range(len(topk_n_indices)):
        for j in range(1,len(topk_n_indices[i])):
            SG.add_edge(int(topk_n_indices[i][0]),int(topk_n_indices[i][j]))
            SG[int(topk_n_indices[i][0])][int(topk_n_indices[i][j])]['weight'] = float(topk_n_values[i][j])

    des = []
    for i in range(n,break_p):
        des_path = f'{test_folder_path}/des/' + f'{i+1}.pkl'
        f = open(des_path, 'rb')
        tmp_des = pickle.load(f)
        for j in range(len(tmp_des)):
            des.append(tmp_des[j])

    # Eva
    predictions = []
    labels = []

    current_time = datetime.datetime.now()
    print('Times:', current_time)
    for step, batch in enumerate(test_loader):
        if step * batch_size < n:
            continue
        with torch.no_grad():
            ids = batch['id'].tolist()
            questions_emb = [questions_emb_list[i] for i in ids]
            test_extra_knowledges, _, masks,retri_des = retrieval_func_with_att(questions_emb, doc_embed_tensor, KG_lists, SG,des,device=model.device(),topk_n=20,intervals=0)
            output = model.inference_text(batch,retri_des)
            for j in range(len(output['questions'])):
                predictions.append(output['pred'][j])
                labels.append(output['answers'][j])

    with open('./answer_list/ours_hotpop_2att.pkl', 'wb') as file2:
        pickle.dump(predictions,file2)

    current_time = datetime.datetime.now()
    print('Times:', current_time)

    Truenums1 = 0
    for i in range(len(predictions)):
        if labels[i].lower() in predictions[i].lower():
            Truenums1 += 1
    print('EM: ',Truenums1/len(predictions))


if __name__ == "__main__":
    args = parse_args_llama()

    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()