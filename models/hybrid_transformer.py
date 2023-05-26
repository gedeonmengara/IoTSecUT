from .blocks import SharedEmbeddings, TransformerEncoderBlock, _linear_dropout_bn, _initialize_layers

import torch
import torch.nn as nn
from einops import rearrange

from collections import OrderedDict

class TableHybridTransformer(nn.Module):

    def __init__(
        self, 
        cfg,
        categorical_dim=None,
        categorical_cardinality=None,
        continuous_dim=None,
        output_dim=1
    ):

        super().__init__()

        self.cfg = cfg
        self.categorical_dim = categorical_dim
        self.continuous_dim = continuous_dim

        if categorical_dim>0:
            self.cat_embedding_layers = nn.ModuleList([
                SharedEmbeddings(
                    cardinality,
                    cfg.MODEL.INPUT_EMBED_DIM,
                    add_shared_embed=cfg.MODEL.SHARE_EMBEDDING_STRATEGY=='add',
                    frac_shared_embed=cfg.MODEL.SHARED_EMBEDDING_FRACTION
                )
                for cardinality in categorical_cardinality
            ])

            self.embed_dropout = nn.Dropout(cfg.MODEL.EMBEDDING_DROPOUT)

        self.transformer_blocks = OrderedDict()
        for i in range(cfg.MODEL.NUM_ATT_BLOCKS):
            self.transformer_blocks[f"mha_block_{i}"] = TransformerEncoderBlock(
                input_embed_dim=cfg.MODEL.INPUT_EMBED_DIM,
                num_heads=cfg.MODEL.NUM_HEADS,
                ff_hidden_multiplier=cfg.MODEL.FF_HIDDEN_MULTIPLIER,
                ff_activation=cfg.MODEL.TRANSFORMER_ACTIVATION,
                attn_dropout=cfg.MODEL.ATTN_DROPOUT,
                ff_dropout=cfg.MODEL.FF_DROPOUT,
                add_norm_dropout=cfg.MODEL.ADD_NORM_DROPOUT,
                keep_attn=False,  # No easy way to convert TabTransformer Attn Weights to Feature Importance
            )

        self.transformer_blocks = nn.Sequential(self.transformer_blocks)
        self.attention_weights = [None] * cfg.MODEL.NUM_ATT_BLOCKS

        if self.cfg.MODEL.BATCH_NORM_CONTINUOUS_INPUT:
            self.normalizing_batch_norm = nn.BatchNorm1d(continuous_dim)

        _curr_units = cfg.MODEL.INPUT_EMBED_DIM * categorical_dim + continuous_dim
        layers = []

        in_units = _curr_units
        for units in cfg.MODEL.OUT_FF_LAYERS:
            out_units = units
            layers.extend(
                _linear_dropout_bn(
                    cfg.MODEL.OUT_FF_ACTIVATION,
                    cfg.MODEL.OUT_FF_INITIALIZATION,
                    cfg.MODEL.USE_BATCH_NORM,
                    in_units,
                    out_units,
                    cfg.MODEL.OUT_FF_DROPOUT,
                )
            )

            in_units = units

        self.linear_layers = nn.Sequential(*layers)
        # self.output_dim = _curr_units

        self.head = nn.Sequential(
            nn.Dropout(cfg.MODEL.OUT_FF_DROPOUT),
            nn.Linear(out_units, output_dim),
        )
        self.certain_attention = nn.Sequential(nn.Linear(out_units, 1), nn.Sigmoid())

        _initialize_layers(
            cfg.MODEL.OUT_FF_ACTIVATION,
            cfg.MODEL.OUT_FF_INITIALIZATION,
            self.head,
        )
    
    def forward(self, x):
        continuous_data, categorical_data = x["continuous"], x["categorical"]

        x = None
        if self.categorical_dim > 0:
            x_cat = [
                embedding_layer(categorical_data[:, i]).unsqueeze(1)
                for i, embedding_layer in enumerate(self.cat_embedding_layers)
            ]
            # (B, N, E)
            x = torch.cat(x_cat, 1)
            if self.cfg.MODEL.EMBEDDING_DROPOUT != 0:
                x = self.embed_dropout(x)
            for i, block in enumerate(self.transformer_blocks):
                x = block(x)
            # Flatten (Batch, N_Categorical, Hidden) --> (Batch, N_CategoricalxHidden)
            x = rearrange(x, "b n h -> b (n h)")
        if self.continuous_dim > 0:
            if self.cfg.MODEL.BATCH_NORM_CONTINUOUS_INPUT:
                x_cont = self.normalizing_batch_norm(continuous_data)
            else:
                x_cont = continuous_data
            # (B, N, E)
            x = x_cont if x is None else torch.cat([x, x_cont], 1)
        
        x = self.linear_layers(x)

        if self.cfg.TRAIN.SCHEME == 0:
            attention_weights = 1.0
        else:
            attention_weights = self.certain_attention(x)

        x = self.head(x)

        return attention_weights, x * attention_weights