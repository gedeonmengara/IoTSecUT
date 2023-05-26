import pandas as pd
import numpy as np

from synthesizer.cgan import CGANSynthesizer
from datasets.ordinal_encoder import OrdinalEncoder

# data = pd.read_csv("data/bot_iot.csv")
# drop_cols = ["flgs", "saddr", "sport", "daddr", "dport", "attack", "subcategory"]
# categorical_cols = ["proto", "state"]
# target_cols = "category"

# data.drop(drop_cols, axis=1, inplace=True)
# model = CGANSynthesizer(verbose=True, batch_size=500)

# # print(data)
# model.train(data, categorical_columns=categorical_cols)
# model.save("weights/synthesizer/bot_iot.pt")


# data = pd.read_csv("data/cicids2018.csv")
# categorical_cols = ["Protocol"]
# model = CGANSynthesizer(verbose=True, batch_size=500)
# model.train(data, categorical_columns=categorical_cols)
# model.save("weights/synthesizer/cicids2018.pt")



# data = pd.read_csv("data/bot_iot.csv")
# categorical_cols = ["proto", "state"] + ["category"]
# cat_encoder = OrdinalEncoder(categorical_cols)
# data = cat_encoder.fit_transform(data)
# cat_encoder.save_mapping("weights/odinal_mappings/bot_iot.pkl")
# model = CGANSynthesizer(verbose=True, batch_size=500)
# model.train(data, categorical_columns=categorical_cols)
# model.save("weights/synthesizer/bot_iot.pt")



# data = pd.read_csv("data/cicids2018.csv")
# categorical_cols = ["Protocol"]
# target_cols = "Label"
# data.fillna(0, inplace=True)

# # we will use these for plotting below
# attack_labels = ['Normal','BruteForce','DDOS','DoS', "Infilteration", "SQLInjection", "Bot"]

# # helper function to pass to data frame mapping
# def map_attack(attack):
#     if attack.startswith("Brute") or attack.startswith("FTP") or attack.startswith("SSH"):
#         # dos_attacks map to 1
#         attack_type = 1
#     elif attack.startswith("DDOS"):
#         # probe_attacks mapt to 2
#         attack_type = 2
#     elif attack.startswith("DoS"):
#         # privilege escalation attacks map to 3
#         attack_type = 3
#     elif attack.startswith("Infilteration"):
#         # remote access attacks map to 4
#         attack_type = 4
#     elif attack.startswith("SQL"):
#         attack_type = 5
#     elif attack.startswith("Bot"):
#         attack_type = 6
#     else:
#         # normal maps to 0
#         attack_type = 0
        
#     return attack_type

# data[target_cols] = data[target_cols].apply(lambda x: map_attack(x))
# data = data[np.isfinite(data).all(1)]

# categorical_cols = ["Protocol"] + ["Label"]
# cat_encoder = OrdinalEncoder(categorical_cols)
# data = cat_encoder.fit_transform(data)
# cat_encoder.save_mapping("weights/odinal_mappings/cicids2018.pkl")
# model = CGANSynthesizer(verbose=True, batch_size=500)
# model.train(data, categorical_columns=categorical_cols)
# model.save("weights/synthesizer/cicids2018.pt")

# data = pd.read_csv("data/UNSW_NB15_training-set.csv")
# categorical_cols = ["proto", "service", "state"] + ["attack_cat"]
# cat_encoder = OrdinalEncoder(categorical_cols)
# data = cat_encoder.fit_transform(data)
# cat_encoder.save_mapping("weights/odinal_mappings/unsw_nb15.pkl")
# model = CGANSynthesizer(verbose=True, batch_size=500)
# model.train(data, categorical_columns=categorical_cols)
# model.save("weights/synthesizer/unsw_nb15.pt")




########################################################
###############      Generate      #####################
########################################################

data = pd.read_csv("data/UNSW_NB15_training-set.csv")
categorical_cols = ["proto", "service", "state"] + ["attack_cat"]
cat_encoder = OrdinalEncoder(categorical_cols)
cat_encoder.load_mapping("weights/odinal_mappings/unsw_nb15.pkl")
data = cat_encoder.transform(data)
model = CGANSynthesizer(verbose=True, batch_size=500, epochs=1)
model.train(data, categorical_columns=categorical_cols)
model.load("weights/synthesizer/unsw_nb15.pt")

# print(cat_encoder._mapping["attack_cat"])
data_cls = model.generate(10, condition_column="attack_cat", condition_value=1)

print(data_cls)