import json
from torch.utils.data import Dataset
from data_provider.mol_dataset import smiles2graph, get_unimol_data
from collections import defaultdict
from torch_geometric.data import Data, Batch
from data_provider.tokenization_utils import batch_tokenize_messages_list
from data_provider.collaters import Mol3DCollater
from data_provider.mol_dataset import MolDataset_cid

import numpy as np
import re
import os


class JudgeDataset(Dataset):
    def __init__(self, 
                json_paths, 
                unimol_dictionary,
                encoder_types,
                prompts,
                root,
                ):
        super().__init__()

        # self.data_list = self.pampa_data['data_list']
        # self.split = self.pampa_data['split']
        # self.data_list = [self.data_list[i] for i in self.split[split]]

        print('Loading molecule data...')
        molecules_data_list = json.load(open(root + 'pubchem-molecules.json'))
        self.mol_dataset = MolDataset_cid(molecules_data_list, unimol_dictionary, encoder_types)

        self.data_list = defaultdict(list)
        for j, json_path in enumerate(json_paths):
            with open(json_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    self.data_list[j].append(data)
            print(json_path, len(self.data_list[j]))

        min_len = min([len(self.data_list[i]) for i in range(len(self.data_list))])
        for i in range(len(self.data_list)):
            self.data_list[i] = self.data_list[i][:min_len]
            print(json_paths[i], len(self.data_list[i]))

        self.mol_prompt = "<mol><mol><mol><mol><mol><mol><mol><mol>"
        self.prompts = prompts
        self.unimol_dictionary = unimol_dictionary

        # self.responses = []
        # if response_path is not None:
        #     with open(response_path, 'r', encoding='utf-8') as f:
        #         for line in f:
        #             data = json.loads(line)
        #             self.responses.append(data)
        #     assert len(self.responses) == len(self.data_list)

    def __len__(self):
        return len(self.data_list[0])

    def __getitem__(self, idx):
        data_0 = self.data_list[0][idx]
        data_1 = self.data_list[1][idx]
        assert data_0['cid'] == data_1['cid']
        cid = data_0['cid']
        data_graphs, data_others = self.mol_dataset[cid]

        message = data_0['messages'][-1]
        response_0, response_1 = data_0['response'], data_1['response']

        assert message['role'] == 'user'
        user_question = message.get('content', "")

        system_prompt = self.prompts['system']
        user_prompt_temp = self.prompts['user'].replace('<<QUESTION>>', user_question)

        user_prompt_forward = user_prompt_temp.replace('<<ANSWER1>>', response_0).replace('<<ANSWER2>>', response_1)
        user_prompt_reverse = user_prompt_temp.replace('<<ANSWER1>>', response_1).replace('<<ANSWER2>>', response_0)
        
        other_infos = {
            'cid': cid,
            'names': data_0['names'],
            'task_type': data_0['task_type'],
            'messages': data_0['messages'],
            'model_response': [response_0, response_1],
        }
        
        messages_forward = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_forward}
        ]
        messages_reverse = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_reverse}
        ]
        return data_graphs, messages_forward, messages_reverse, other_infos

class JudgeCollater():
    def __init__(self, tokenizer, unimol_dictionary, llama_version):
        self.tokenizer = tokenizer
        self.llama_version = llama_version
        self.d3_collater = Mol3DCollater(unimol_dictionary.pad())
        
    def __call__(self, batch):
        data_graphs, messages_forward_list, messages_reverse_list, other_infos = zip(*batch)

        graph_batch = {}
        data_unimol = []
        for data in data_graphs:
            data_unimol.extend(data['unimol'])
        graph_batch['unimol'] = self.d3_collater(data_unimol)
        data_moleculestm = []
        for data in data_graphs:
            data_moleculestm.extend(data['moleculestm'])
        graph_batch['moleculestm'] = Batch.from_data_list(data_moleculestm)

        tokenized_forward = batch_tokenize_messages_list(messages_forward_list, self.tokenizer, 
                                                self.llama_version, padding_side='left', think=True)
        tokenized_reverse = batch_tokenize_messages_list(messages_reverse_list, self.tokenizer, 
                                                self.llama_version, padding_side='left', think=True)
        
        return graph_batch, tokenized_forward, tokenized_reverse, other_infos
