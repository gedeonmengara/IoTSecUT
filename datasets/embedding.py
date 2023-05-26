import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
import numpy as np

import torch
from torch.utils.data import Dataset

from dimensionality_reduction import Reduce
from .ordinal_encoder import OrdinalEncoder

logger = logging.getLogger(__name__)

class EmbeddingDataset(Dataset):

    def __init__(self, df_path, data_type, drop_cols, cat_cols, cont_cols, target, latent_dim, is_reduce=False, reduced_weight_path=None):

        self.data_type = data_type
        self.drop_cols = drop_cols
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols
        self.target = target
        self.latent_dim = latent_dim
        self.is_reduce = is_reduce
        self.reduced_weight_path = reduced_weight_path

        self.categorical_encoder = OrdinalEncoder(cat_cols)
        self.continuous_transform = PowerTransformer(method="yeo-johnson", standardize=False)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        self.data= self.prepare_data(df_path)
        self.y = self.data[self.target].astype(np.int64).values

        if len(self.cat_cols) > 0:
            self.categorical_X = self.data[self.cat_cols].astype(np.int64).values
        # if len(self.cont_cols) > 0:
        #     self.continuous_X = self.data[self.cont_cols].astype(np.float32).values

        # reduce Dim


    def prepare_data(self, df_path):

        data = pd.read_csv(df_path)
        data.drop(self.drop_cols, axis=1, inplace=True)

        data = self.categorical_encoder.fit_transform(data)
        self.categorical_cardinality = [self.categorical_encoder._mapping[a].shape[0]+1 for a in self.cat_cols]

        # Continuous

        if self.is_reduce:
            dim_reduce = Reduce(
                data[self.cont_cols],
                data_type=self.data_type,
                latent_dim=self.latent_dim,
            )

            reduced_mu, _ = dim_reduce.batch_embedding(self.reduced_weight_path)
            self.continuous_X = reduced_mu.cpu().numpy()

        else:
            data.loc[:, self.cont_cols] = self.continuous_transform.fit_transform(
                data.loc[:, self.cont_cols]
            )
            data.loc[:, self.cont_cols] = self.scaler.fit_transform(
                data.loc[:, self.cont_cols]
            )

            self.continuous_X = data[self.cont_cols].astype(np.float32).values

        data[self.target] = self.label_encoder.fit_transform(
            data[self.target]
        )

        return data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        sample_data = {
            "target": self.y[idx],
            "continuous": self.continuous_X[idx] if len(self.cont_cols) > 0 else torch.Tensor(),
            "categorical": self.categorical_X[idx] if len(self.cat_cols) > 0 else torch.Tensor(),
        }

        return sample_data