import json
import pandas as pd
import torch
from torch.utils.data import Dataset


class HOTPOPDataset(Dataset):
    def __init__(self,path,size1,size2):
        super().__init__()

        with open(path, 'r') as file:
            raw_data = json.load(file)

        self.questions = [raw_data[i]['question'] for i in range(min(len(raw_data),size2))]
        self.answers = [raw_data[i]['answer'] for i in range(min(len(raw_data),size2))]
        self.contexts = [raw_data[i]['context'] for i in range(min(len(raw_data),size2))]


    def __len__(self):
        """Return the len of the dataset."""
        return len(self.questions)

    def __getitem__(self, index):

        questions = self.questions[index]
        answers = self.answers[index]
        contexts = self.contexts[index]

        return {
            'id': index,
            'questions': questions,
            'answers': answers,
            'contexts': contexts,
        }