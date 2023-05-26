from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

class ContDataset(Dataset):

    def __init__(self, df_cont):

        """
        df_cont : only cont columns
        """
        x = df_cont.values.reshape(-1, df_cont.shape[1]).astype('float32')
        self.standardizer = StandardScaler()
        self.x = self.standardizer.fit_transform(x)

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx]