import torch, torchaudio
from torch.utils import data
import pandas


class VoxforgeDataset(data.Dataset):

    def __init__(self, csv_file: str):
        super().__init__()
        self.data_index = pandas.read_csv(csv_file)
        langs_str = self.data_index.columns[-1]
        self.langs = sorted(langs_str.split())
        counts = self.data_index[langs_str].value_counts()
        self.weights = [ counts.max()/counts[l] for l in self.langs ]
        self.lang_index = { lang: i for i, lang in enumerate(self.langs) }      

    def __len__(self):
        return len(self.data_index.index)

    def __getitem__(self, index: int):
        row = self.data_index.iloc[index]
        audio, sr = torchaudio.load(row[1])
        assert sr == 8000 # Resampling done by organize_data script
        return audio, self.lang_index[row[-1]]

