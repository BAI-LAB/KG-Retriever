import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model.gnn import load_gnn_model
from torch_geometric.data.data import Data
import numpy as np
from torch_scatter import scatter
BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '[/s]'

IGNORE_INDEX = -100

class GraphRAGLLM(torch.nn.Module):

    def __init__(
        self,
        args,
        **kwargs
    ):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens

        print('Loading LLM')
        kwargs = {
            "max_memory": {1:'60GB'},
            "device_map": "auto",
            "revision": "main",
        }

        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False, revision=kwargs["revision"],trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            **kwargs
        )

        if args.llm_frozen == 'True':
            print("Freezing LLM!")
            for name, param in model.named_parameters():
                param.requires_grad = False
        else:
            print("Training LLM with LORA!")

        self.model = model
        print('Finish loading LLM!')

        self.graph_encoder = load_gnn_model[args.gnn_model_name](
            in_channels=args.gnn_in_dim,
            out_channels=args.gnn_out_dim,
            hidden_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            num_heads=args.gnn_num_heads,
        ).to(self.model.device)

        self.input_embeddings = self.model.get_input_embeddings()

    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.bfloat16):
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)

    def forward(self, samples, extra_knowledges,masks,des):
        # encode questions and answers
        questions = self.tokenizer(samples["questions"], add_special_tokens=False)
        answers = self.tokenizer(samples["answers"], add_special_tokens=False)
        prompting = self.tokenizer('请结合上述信息对接下来的问题做出回答，你的回答仅包含问题对应的答案，不包含其他无关信息。', add_special_tokens=False)
        des = self.tokenizer(des, add_special_tokens=False)

        # encode special tokens
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.input_embeddings(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.device()))
        pad_embeds = self.input_embeddings(torch.tensor(self.tokenizer.pad_token_id).to(self.device())).unsqueeze(0)

        # encode graphs
        glist = []
        for i in range(len(masks)):
            extra_knowledge = extra_knowledges[i]
            mask = masks[i]
            graph_embeds = self.encode_subgraphs(extra_knowledge,mask)
            graph_embeds = self.projector(graph_embeds)
            glist.append(graph_embeds)

        # Add bos & eos token
        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        for i in range(batch_size):
            # Add bos & eos token
            label_input_ids = answers.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids
            input_ids = des.input_ids[i] + prompting.input_ids + questions.input_ids[i] + eos_user_tokens.input_ids + label_input_ids
            #input_ids = prompting.input_ids + questions.input_ids[i] + eos_user_tokens.input_ids + label_input_ids
            inputs_embeds = self.input_embeddings(torch.tensor(input_ids).to(self.model.device))
            #inputs_embeds = torch.cat([bos_embeds, glist[i].unsqueeze(0), inputs_embeds], dim=0)
            inputs_embeds = torch.cat([bos_embeds, glist[i], inputs_embeds], dim=0)
            #inputs_embeds = torch.cat([bos_embeds, torch.cat(glist), inputs_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0] - len(label_input_ids)) + label_input_ids
            batch_label_input_ids.append(label_input_ids)

        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length+batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )

        return outputs.loss

    def inference(self, samples,extra_knowledges,masks):
        # encode questions
        questions = self.tokenizer(samples["questions"], add_special_tokens=False)
        prompting = self.tokenizer('请结合上述信息，用一句话对接下来的问题做出回答，你的回答仅包含问题对应的答案，不包含其他无关信息。',add_special_tokens=False)

        # encode special tokens
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.input_embeddings(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0])
        pad_embeds = self.input_embeddings(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0)

        # encode graphs
        glist = []

        for i in range(len(masks)):
            extra_knowledge = extra_knowledges[i]
            mask = masks[i]
            graph_embeds = self.encode_subgraphs(extra_knowledge, mask.unique())
            graph_embeds = self.projector(graph_embeds)
            glist.append(graph_embeds)

        # Add bos & eos token
        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            # Add bos & eos token
            input_ids = prompting.input_ids + questions.input_ids[i] + eos_user_tokens.input_ids
            inputs_embeds = self.input_embeddings(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, glist[i], inputs_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                return_dict=True,
            )

        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {'id': samples['id'],
                'pred': pred,
                'answers': samples['answers'],
                'questions': samples['questions'],
                }

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    def inference_text(self, samples, extra_knowledges):
        questions = self.tokenizer(samples["questions"], add_special_tokens=False)
        # For CHINESE DATASETS
        # prompting = self.tokenizer('请整理以上三元组中的信息，对接下来的问题做出回答，你的回答应为一句完整的话，不要输出选项，回答示例：“启明行动是为了防控儿童青少年的近视问题，并发布了《防控儿童青少年近视核心知识十条》“。你需要回答的问题是:',add_special_tokens=False)

        prompting = self.tokenizer(
              'Please combine the information in the above triples and your own knowledge to answer the following questions, Answer the question in short',
           add_special_tokens=False)

        extra_knowledges_emb = self.tokenizer(extra_knowledges, add_special_tokens=False)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            input_ids = extra_knowledges_emb.input_ids[i] + prompting.input_ids + questions.input_ids[i]
            inputs_embeds = self.input_embeddings(torch.tensor(input_ids).to(self.model.device))

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )

        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {'id': samples['id'],
                'pred': pred,
                'answers': samples['answers'],
                'questions': samples['questions'],
                }