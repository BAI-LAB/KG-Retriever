import json
import pandas as pd
import torch
from torch.utils.data import Dataset


class MUSIQUEDataset(Dataset):
    def __init__(self,path,size1,size2):
        super().__init__()
        questions =[]
        answers = []
        with open(path, 'r') as file:
            index = 0
            for line in file:
                index += 1
                if index > size2:
                    break
                answers.append(json.loads(line)['answer'])
                questions.append(json.loads(line)['question'])

        self.questions = questions
        self.answers = answers


    def __len__(self):
        """Return the len of the dataset."""
        return len(self.questions)

    def __getitem__(self, index):

        questions = self.questions[index]
        answers = self.answers[index]

        return {
            'id': index,
            'questions': questions,
            'answers': answers,
        }