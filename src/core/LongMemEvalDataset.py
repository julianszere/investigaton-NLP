import json
import pandas as pd


class LongMemEvalDataset:
    def __init__(self, dataset_type: str):
        paths = {
            "oracle": "data/longmemeval/longmemeval_oracle.json",
            "short": "data/longmemeval/longmemeval_s_cleaned.json",
            "long": "data/longmemeval/longmemeval_m_cleaned.json",
        }

        with open(paths[dataset_type], "r", encoding="utf-8") as f:
            self.dataset = pd.DataFrame(json.load(f)).reset_index(drop=True)

        self.current_index = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        sliced_data = self.dataset.iloc[key]
        if isinstance(key, slice):
            return [self.instance_from_row(row) for _, row in sliced_data.iterrows()]
        else:
            return self.instance_from_row(sliced_data.iloc[key])

    def instance_from_row(self, row):
        question = row["question"]
        sessions = row["haystack_sessions"]
        t_question = row["question_date"]
        answer = row["answer"]
        return question, sessions, t_question, answer
