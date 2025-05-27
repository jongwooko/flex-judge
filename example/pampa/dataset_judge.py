import json
from torch.utils.data import Dataset
from data_provider.mol_dataset import smiles2graph, get_unimol_data
from collections import defaultdict
from torch_geometric.data import Data, Batch
from data_provider.tokenization_utils import batch_tokenize_messages_list
from data_provider.collaters import Mol3DCollater
import numpy as np
import re

def truncate_at_final_answer(text):
    match = re.search(r"[Ff]inal [Aa]nswer:", text)
    if match:
        return text[:match.start()].strip(), text[match.end():].strip()
    return text, ""


class PAMPADataset(Dataset):
    def __init__(self, json_path, split, prompt_type, 
                unimol_dictionary,
                prompts,
                response_path=None,
                ):
        super().__init__()

        self.pampa_data = json.load(open(json_path, 'r'))
        self.prompts = prompts

        self.data_list = self.pampa_data['data_list']
        self.split = self.pampa_data['split']

        self.data_list = [self.data_list[i] for i in self.split[split]]

        self.mol_prompt = "<mol><mol><mol><mol><mol><mol><mol><mol>"

        self.unimol_dictionary = unimol_dictionary

        self.responses = []
        if response_path is not None:
            with open(response_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    self.responses.append(data)
            assert len(self.responses) == len(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        if self.responses:
            response = self.responses[idx]
        else:
            raise NotImplementedError
        
        assert response['SMILES'] == data['smiles']
        
        data_graphs = defaultdict(list)
        data_graphs['unimol'].append(
            get_unimol_data(data['atoms'], np.array(data['coordinates'][0]), self.unimol_dictionary))

        graphs = smiles2graph(data['smiles'])
        data_graphs['moleculestm'].append(
            Data(x=graphs['node_feat'], 
                edge_index=graphs['edge_index'], 
                edge_attr=graphs['edge_feat'])
        )

        # Prepare the prompt
        system_prompt = self.prompts['system']
        user_prompt = self.prompts['user'].replace('<mol>', self.mol_prompt)
        
        assistant_response = response.get("Response", "")
        assistant_response, assistant_answer = truncate_at_final_answer(assistant_response)
        user_prompt = user_prompt + assistant_response

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return data_graphs, messages, data['answer'], data['smiles'], response

class PAMPACollater():
    def __init__(self, tokenizer, unimol_dictionary, llama_version):
        self.tokenizer = tokenizer
        self.llama_version = llama_version
        self.d3_collater = Mol3DCollater(unimol_dictionary.pad())
        
    def __call__(self, batch):
        data_graphs, messages_list, answers, smiles, response = zip(*batch)

        graph_batch = {}
        data_unimol = []
        for data in data_graphs:
            data_unimol.extend(data['unimol'])
        graph_batch['unimol'] = self.d3_collater(data_unimol)
        data_moleculestm = []
        for data in data_graphs:
            data_moleculestm.extend(data['moleculestm'])
        graph_batch['moleculestm'] = Batch.from_data_list(data_moleculestm)

        tokenized = batch_tokenize_messages_list(messages_list, self.tokenizer, 
                                                self.llama_version, padding_side='left')
        text_batch = tokenized

        return graph_batch, text_batch, answers, smiles, response

    