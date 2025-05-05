import json
import pandas as pd
import torch
from torch.utils.data import Dataset
import pyarrow.parquet as pq

class WIKIDataset(Dataset):
    def __init__(self,path,size1,size2):
        super().__init__()
        questions =[]
        answers = []

        table = pq.read_table(path)
        df = table.to_pandas()

        # 将DataFrame转换成JSON字符串
        json_data = df.to_json(orient='records')
        json_data = json.loads(json_data)
        for i in range(1,100):
            answers.append(json_data[i]['answer'])
            questions.append(json_data[i]['question'])

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