# from dimensionality_reduction import Reduce
# import pandas as pd

# df = pd.read_csv("data/syntesized_data/unsw_nb15.csv")
# drop_cols = ["id"]
# categorical_cols = ["proto", "service", "state", "attack_cat"]
# numerical_cols = list(set(df.columns) - set(categorical_cols + drop_cols))

# print(df.shape)
# df.drop(drop_cols, axis=1, inplace=True)
# print(df.shape)

# df_cont = df[numerical_cols]
# print(df_cont.shape)

# dim_reduce = Reduce(
#     df_cont,
#     data_type="unsw_nb15",
#     latent_dim=3,
#     epochs=10000,
#     batch_size=512,
#     lr=1e-4
# )

# dim_reduce.train()

# from dimensionality_reduction import Reduce
# import pandas as pd

# df = pd.read_csv("data/syntesized_data/bot_iot.csv")
# categorical_cols = ["proto", "state", "category"]
# numerical_cols = list(set(df.columns) - set(categorical_cols))

# print(df.shape)
# # df.drop(drop_cols, axis=1, inplace=True)
# print(df.shape)

# df_cont = df[numerical_cols]
# print(df_cont.shape)

# dim_reduce = Reduce(
#     df_cont,
#     data_type="bot_iot",
#     H1=64, 
#     H2=6,
#     latent_dim=3,
#     epochs=2000,
#     batch_size=8192,
#     lr=1e-4,
#     save_postfix="8192_64_6"
# )

# dim_reduce.train()

# from dimensionality_reduction import Reduce
# import pandas as pd

# df = pd.read_csv("data/syntesized_data/cicids2018.csv")
# categorical_cols = ["Protocol", "Label"]
# numerical_cols = list(set(df.columns) - set(categorical_cols))

# print(df.shape)
# # df.drop(drop_cols, axis=1, inplace=True)
# print(df.shape)

# df_cont = df[numerical_cols]
# print(df_cont.shape)

# dim_reduce = Reduce(
#     df_cont,
#     data_type="cicids",
#     H1=128, 
#     H2=24,
#     latent_dim=3,
#     epochs=2000,
#     batch_size=8192,
#     lr=1e-4,
#     save_postfix="8192_128_24"
# )

# dim_reduce.train()


from dimensionality_reduction import Reduce
import pandas as pd

df = pd.read_csv("data/syntesized_data/nsl_kdd.csv")
categorical_cols = ['protocol_type', 'service', 'flag', "attack"]
numerical_cols = list(set(df.columns) - set(categorical_cols))

print(df.shape)
# df.drop(drop_cols, axis=1, inplace=True)
print(df.shape)

df_cont = df[numerical_cols]
print(df_cont.shape)

dim_reduce = Reduce(
    df_cont,
    data_type="nsl_kdd",
    H1=64, 
    H2=6,
    latent_dim=3,
    epochs=2000,
    batch_size=8192,
    lr=1e-4,
    save_postfix="8192_64_6"
)

dim_reduce.train()