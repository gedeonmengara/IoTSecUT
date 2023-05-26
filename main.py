import argparse
import pprint
import numpy as np
import os
from tqdm import tqdm
from omegaconf import OmegaConf
import nvidia_smi
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset, DataLoader

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.metrics import geometric_mean_score

from models import TableHybridTransformer
from datasets.embedding import EmbeddingDataset
from utils.logger import create_logger

def parse_args():

    parser = argparse.ArgumentParser("Intrusion Detection System")

    # Dataset
    parser.add_argument('--cfg_file', type=str, default="configs/hybrid_transformer.yaml", help="config File Path", )
    parser.add_argument('--gpu_id', type=int, default=1, help="GPU I'D")
    args = parser.parse_args()
    return args

def get_gpu_used(gpu_id):
    # GPU consumption (in GB)
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_id)  
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    gpu_used = info.used / (1024 ** 3)
    nvidia_smi.nvmlShutdown()

    return gpu_used

def main():
    args = parse_args()

    start_time = time.time()
    sleep_gpu = get_gpu_used(args.gpu_id)

    cfg = OmegaConf.load(args.cfg_file)

    device = torch.device(f"cuda:{args.gpu_id}") if torch.cuda.is_available() else torch.device("cpu")

    logger, tb_dir = create_logger("logs", cfg.DATASETS.DATA_TYPE)
    logger.info(pprint.pformat(cfg))

    # create tensorboard
    writer = SummaryWriter(log_dir=tb_dir)

    # Dataset settings
    if cfg.DATASETS.DATA_TYPE == "unsw_nb15":
        drop_cols=["id"]
        cat_cols=["proto", "service", "state", "label"]
        cont_cols = [
            'dur', 'spkts', 'dpkts', 'sbytes',
            'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss',
            'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin',
            'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
            'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm',
            'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
            'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm',
            'ct_srv_dst', 'is_sm_ips_ports', 'label'
        ]
        target="attack_cat"

    elif cfg.DATASETS.DATA_TYPE == "bot_iot":
        drop_cols=[]
        cat_cols=["proto", "state"]
        cont_cols = [
            'AR_P_Proto_P_Dport', 'AR_P_Proto_P_DstIP', 
            'AR_P_Proto_P_Sport', 'AR_P_Proto_P_SrcIP', 
            'N_IN_Conn_P_DstIP', 'N_IN_Conn_P_SrcIP', 
            'Pkts_P_State_P_Protocol_P_DestIP', 
            'Pkts_P_State_P_Protocol_P_SrcIP', 'TnBPDstIP', 'TnBPSrcIP', 
            'TnP_PDstIP', 'TnP_PSrcIP', 'TnP_PerProto', 'TnP_Per_Dport', 
            'bytes', 'dbytes', 'dpkts', 'drate', 'dur', 'flgs_number', 
            'ltime', 'max', 'mean', 'min', 'pkSeqID', 'pkts', 
            'proto_number', 'rate', 'sbytes', 'seq', 'spkts', 'srate', 
            'state_number', 'stddev', 'stime', 'sum'
        ]
        target="category"
    
    elif cfg.DATASETS.DATA_TYPE == "cicids":
        drop_cols=[]
        cat_cols=["Protocol"]
        cont_cols = [
            'ACK Flag Cnt', 'Active Max', 'Active Mean', 'Active Min', 
            'Active Std', 'Bwd Blk Rate Avg', 'Bwd Byts/b Avg', 
            'Bwd Header Len', 'Bwd IAT Max', 'Bwd IAT Mean', 
            'Bwd IAT Min', 'Bwd IAT Std', 'Bwd IAT Tot', 'Bwd PSH Flags', 
            'Bwd Pkt Len Max', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Min', 
            'Bwd Pkt Len Std', 'Bwd Pkts/b Avg', 'Bwd Pkts/s', 
            'Bwd Seg Size Avg', 'Bwd URG Flags', 'CWE Flag Count', 
            'Down/Up Ratio', 'Dst Port', 'ECE Flag Cnt', 'FIN Flag Cnt', 
            'Flow Byts/s', 'Flow Duration', 'Flow IAT Max', 'Flow IAT Mean', 
            'Flow IAT Min', 'Flow IAT Std', 'Flow Pkts/s', 'Fwd Act Data Pkts', 
            'Fwd Blk Rate Avg', 'Fwd Byts/b Avg', 'Fwd Header Len', 'Fwd IAT Max', 
            'Fwd IAT Mean', 'Fwd IAT Min', 'Fwd IAT Std', 'Fwd IAT Tot', 'Fwd PSH Flags', 
            'Fwd Pkt Len Max', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Min', 
            'Fwd Pkt Len Std', 'Fwd Pkts/b Avg', 'Fwd Pkts/s', 'Fwd Seg Size Avg', 
            'Fwd Seg Size Min', 'Fwd URG Flags', 'Idle Max', 'Idle Mean', 
            'Idle Min', 'Idle Std', 'Init Bwd Win Byts', 'Init Fwd Win Byts', 
            'PSH Flag Cnt', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Min', 
            'Pkt Len Std', 'Pkt Len Var', 'Pkt Size Avg', 'RST Flag Cnt', 
            'SYN Flag Cnt', 'Subflow Bwd Byts', 'Subflow Bwd Pkts', 
            'Subflow Fwd Byts', 'Subflow Fwd Pkts', 'Tot Bwd Pkts', 'Tot Fwd Pkts', 
            'TotLen Bwd Pkts', 'TotLen Fwd Pkts', 'URG Flag Cnt'
        ]
        target="Label"

    elif cfg.DATASETS.DATA_TYPE == "nsl_kdd":
        drop_cols=[]
        cat_cols=['protocol_type', 'service', 'flag']
        cont_cols = [
            'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
            'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 
            'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'level'
        ]
        target = "attack"

    else:
        raise NotImplementedError

    data_set = EmbeddingDataset(
        df_path=cfg.DATASETS.PATH,
        data_type=cfg.DATASETS.DATA_TYPE,
        drop_cols=drop_cols,
        cat_cols=cat_cols,
        cont_cols=cont_cols,
        target=target,
        latent_dim=cfg.DATASETS.LATENT_DIM,
        is_reduce=cfg.DATASETS.REDUCE,
        reduced_weight_path=cfg.DATASETS.REDUCE_WEIGHT_PATH
    )

    # Split Dataset

    # Random
    # train_indices, val_indices = train_test_split(list(range(len(data_set))), test_size=0.2)

    # Equal per class
    label_indices = [[] for _ in range(cfg.DATASETS.N_CLASSES)]
    for idx, batch in enumerate(data_set):
        label_indices[batch["target"]].append(idx)

    # Split each label's indices into train and validation sets
    train_indices, val_indices = [], []
    for label_idx in label_indices:
        train_idx, val_idx = train_test_split(label_idx, test_size=0.2, random_state=42)
        train_indices.extend(train_idx)
        val_indices.extend(val_idx)

    
    train_set = Subset(data_set, train_indices)
    val_set = Subset(data_set, val_indices)

    train_loader = DataLoader(train_set, batch_size=cfg.DATASETS.BATCH_SIZE, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=cfg.DATASETS.BATCH_SIZE, num_workers=8, pin_memory=True, shuffle=True)

    if cfg.DATASETS.REDUCE:
        continuous_dim = cfg.DATASETS.LATENT_DIM
    else:
        continuous_dim = len(cont_cols)
        
    model = TableHybridTransformer(
        cfg=cfg,
        categorical_dim=len(cat_cols),
        continuous_dim=continuous_dim,
        categorical_cardinality=data_set.categorical_cardinality,
        output_dim=cfg.DATASETS.N_CLASSES
    )

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, cfg.TRAIN.LR, epochs=cfg.TRAIN.EPOCHS, steps_per_epoch=len(train_loader))

    val_acc_track = 0

    for epoch in range(cfg.TRAIN.EPOCHS):

        model.train()
        losses = 0

        with tqdm(train_loader, unit="batch") as tloader:
            for batch in tloader:
                tloader.set_description("Epoch {}".format(epoch))

                batch = {k: v.to(device=device, non_blocking=True) for k, v in batch.items()}

                optimizer.zero_grad()
                attention_weights, pred = model(batch)

                if cfg.TRAIN.SCHEME == 2:
                    n_top = int(cfg.DATASETS.BATCH_SIZE * cfg.TRAIN.BETA)
                    n_down = cfg.DATASETS.BATCH_SIZE - n_top
                    _, top_idx = torch.topk(attention_weights.squeeze(), n_top)
                    _, down_idx = torch.topk(attention_weights.squeeze(), n_down, largest = False)

                    high_mean = torch.mean(attention_weights[top_idx])
                    low_mean = torch.mean(attention_weights[down_idx])

                    diff  = low_mean - high_mean + cfg.TRAIN.MARGIN
                    diff = max(torch.tensor(0), diff)
                else:
                    diff = torch.tensor(0)
                                        
                ce_loss = criterion(pred, batch["target"])
                loss = ce_loss + diff

                loss.backward()
                optimizer.step()
                sched.step()

                losses += loss.item()

                tloader.set_postfix(loss=loss.item())
        
        train_loss = losses/len(train_loader)
        msg = 'Train Epoch : {}\t Training Loss : {}'.format(epoch, train_loss)
        logger.info(msg)

        model.eval()
        tar_list = []
        pred_list = []
        losses = 0

        with torch.no_grad():
            with tqdm(val_loader, unit="batch") as tloader:
                for batch in tloader:
                    tloader.set_description("Epoch {}".format(epoch))

                    batch = {k: v.to(device=device, non_blocking=True) for k, v in batch.items()}

                    _, pred = model(batch)
                    _, predicted = torch.max(pred.data, 1)

                    tar_list.append(batch["target"].cpu().numpy())
                    pred_list.append(predicted.cpu().numpy())

                    loss = criterion(pred, batch["target"])

                    losses += loss.item()

                    tloader.set_postfix(loss=loss.item())

        val_loss = losses/len(val_loader)
        tar_list_np = np.concatenate(tar_list)
        pred_list_np = np.concatenate(pred_list)

        cm = confusion_matrix(tar_list_np, pred_list_np)
        class_acc = cm.diagonal()/cm.sum(axis=1)
        class_precision = precision_score(pred_list_np, tar_list_np, average=None)
        class_f1 = f1_score(pred_list_np, tar_list_np, average=None)
        class_recall = recall_score(pred_list_np, tar_list_np, average=None)
        class_g_mean = geometric_mean_score(tar_list_np, pred_list_np, average=None)

        accuracy = accuracy_score(pred_list_np, tar_list_np)
        precision = precision_score(pred_list_np, tar_list_np, average='macro')
        f1 = f1_score(pred_list_np, tar_list_np, average='macro')
        recall = recall_score(pred_list_np, tar_list_np, average='macro')
        g_mean = geometric_mean_score(tar_list_np, pred_list_np)

        gpu_mem = get_gpu_used(args.gpu_id) - sleep_gpu

        logger.info('Class Accuracy')
        logger.info(pprint.pformat(class_acc))
        logger.info('Class Precision')
        logger.info(pprint.pformat(class_precision))
        logger.info('Class F1 Score')
        logger.info(pprint.pformat(class_f1))
        logger.info('Class Recall')
        logger.info(pprint.pformat(class_recall))
        logger.info("Class G Mean")
        logger.info(pprint.pformat(class_g_mean))

        msg = 'Eval Epoch : {}\t Validation Loss {}\t Accuracy : {}\t Precision : {}\t F1_score : {}\t Recall : {}\t G Mean : {}'.format(
            epoch, val_loss, accuracy, precision, f1, recall, g_mean
        )
        logger.info(msg)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Metrics/Accuracy", accuracy, epoch)
        writer.add_scalar("Metrics/Precision", precision, epoch)
        writer.add_scalar("Metrics/Recall", recall, epoch)
        writer.add_scalar("Metrics/F1", f1, epoch)
        writer.add_scalar("Metrics/GMean", f1, epoch)
        writer.add_scalar("Utils/GPU", gpu_mem, epoch)

        if accuracy > val_acc_track:

            ckpt = {
                "model_dict": model.state_dict(),
                'val_loss' : val_loss,
                'epoch': epoch,
                'accuracy': accuracy,
                'precision' : precision,
                'f1_score' : f1,
                'recall' : recall,
                "gmean": g_mean,
                "cm": cm,
                'class_acc' : class_acc,
                'class_precision' : class_precision,
                'class_f1' : class_f1,
                'class_recall' : class_recall,
                "class_g_mean": class_g_mean,
            }

            save_path = os.path.join(cfg.TRAIN.WEIGHT_PATH, f'best_model_{cfg.DATASETS.DATA_TYPE}_scheme_{cfg.TRAIN.SCHEME}_{cfg.DATASETS.REDUCE}.pt')
            torch.save(ckpt, save_path)
            logger.info(f'Best Model saved with accuracy {accuracy}')
            val_acc_track = accuracy

    logger.info("Training finished")
    msg = "accuracy: {} precision: {} f1: {} recall: {}".format(
        ckpt["accuracy"], ckpt["precision"], ckpt["f1_score"], ckpt["recall"]
    )
    logger.info(msg)

    end_time = time.time()

    logger.info(f"finished. Time taken {end_time - start_time}")

if __name__ == "__main__":
    main()