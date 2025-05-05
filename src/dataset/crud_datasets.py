import json
import pandas as pd
import torch
from torch.utils.data import Dataset


class CRUDDataset(Dataset):
    def __init__(self,path,size1,size2):
        super().__init__()

        with open(path, 'r') as file:
            raw_data = json.load(file)['results']
        self.questions = [raw_data[i]['questions'] for i in range(min(len(raw_data),size2))]
        self.answers = [raw_data[i]['answers'] for i in range(min(len(raw_data),size2))]
        self.news = [raw_data[i]['news1'] for i in range(min(len(raw_data),size2))]
        self.event = [raw_data[i]['event'] for i in range(min(len(raw_data),size2))]

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.questions)

    def __getitem__(self, index):

        questions = self.questions[index]
        answers = self.answers[index]
        news = self.news[index]
        event = self.event[index]

        return {
            'id': index,
            'questions': questions,
            'answers': answers,
            'news': news,
            'event': event,
        }