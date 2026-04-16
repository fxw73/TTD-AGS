import argparse
import time
from copy import deepcopy
from PIL import Image
import numpy as np
import json
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.nn.functional import softmax
from sklearn.decomposition import PCA
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models
import matplotlib.pyplot as plt
from clip.custom_clip import get_coop
from clip.clip import tokenize
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
import os
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import torchattacks
import random
from collections import defaultdict
from typing import Dict, List
import torchvision.transforms as T
from scipy.signal import savgol_filter
from torchvision.datasets import *
import torchvision
from autoattack import AutoAttack
import seaborn as sns
from scipy import stats
import os


tinyimagenet_root = '/media/cqu/D/FXV/PSSR_master/TINY-IMAGE/tiny-imagenet-200'
# ---- 辅助：稳定协方差计算 ----
def safe_cov_from_features(feats: torch.Tensor, mu: torch.Tensor = None, eps: float = 1e-6) -> torch.Tensor:
    """
    稳定地从多个增强特征计算协方差（中心化到 mu，如果 mu=None 则使用样本均值）
    输入:
      feats: [K, D] tensor
      mu: [D] tensor or None
    返回:
      cov: [D, D] tensor (在 feats.device, feats.dtype)
    """
    if feats.ndim != 2:
        feats = feats.view(-1, feats.size(-1))
    device = feats.device
    dtype = feats.dtype
    K, D = feats.shape

    if mu is None:
        mu = feats.mean(dim=0)
    else:
        mu = mu.to(device=device, dtype=dtype)

    # 中心化

    diff = feats - mu.unsqueeze(0)

    if K < 2:
        # 少样本退化：仅用逐维方差（biased var 避免 N-1=0），添加 eps 保证非零
        var = feats.var(dim=0, unbiased=False).clamp_min(eps)
        cov = torch.diag(var).to(device=device, dtype=dtype)
    else:
        cov = (diff.T @ diff) / (K - 1)
        cov = cov + eps * torch.eye(D, device=device, dtype=dtype)

    # 把可能的 NaN/Inf 替换掉（保险）
    cov = torch.nan_to_num(cov, nan=0.0, posinf=1e6, neginf=-1e6)
    return cov

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def get_top_sim(sim_matrix):
    k = 20 # use 20 neighbor
    sim_matrix[sim_matrix>=1.0] = float('-inf')
    top_k_values, _ = sim_matrix.topk(k, dim=-1)
    top_k_mean = top_k_values.mean(dim=-1)
    return top_k_mean

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

def entropy_avg(outputs):
    batch_entropy = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)
    return batch_entropy.mean()
def tikhonov_shrinkage_preserve_type(sigma_dict, eps_ratio=1e-6, use_torch_batch=False):
    """
    对每个协方差矩阵做 Tikhonov / Ridge 收缩，同时保持原数据类型
    优化点:
    - 使用谱范数 (||Sigma||_2) 代替全特征值分解，加速最大特征值计算
    - 支持 torch 批量计算
    """
    shrink_sigma_dict = {}

    if use_torch_batch and all(isinstance(Sigma, torch.Tensor) for Sigma in sigma_dict.values()):
        # 批量 torch 模式
        keys = list(sigma_dict.keys())
        Sigma_batch = torch.stack([sigma_dict[k] for k in keys], dim=0)  # [C,D,D]
        Sigma_batch = (Sigma_batch + Sigma_batch.transpose(-1, -2)) / 2
        # 最大特征值 ≈ 谱范数
        max_eigs = torch.linalg.norm(Sigma_batch.double(), 2, dim=(-2, -1))
        eps_vals = eps_ratio * max_eigs
        eye = torch.eye(Sigma_batch.size(-1), device=Sigma_batch.device, dtype=Sigma_batch.dtype)
        Sigma_shrink_batch = Sigma_batch + eps_vals.view(-1, 1, 1) * eye
        shrink_sigma_dict = {k: Sigma_shrink_batch[i] for i, k in enumerate(keys)}
        return shrink_sigma_dict

    # 普通逐类版本
    for cls_name, Sigma in sigma_dict.items():
        if isinstance(Sigma, torch.Tensor):
            Sigma = (Sigma + Sigma.T) / 2
            max_eig = torch.linalg.norm(Sigma.double(), 2).item()
            eps = eps_ratio * max_eig
            Sigma_shrink = Sigma + eps * torch.eye(Sigma.shape[0], dtype=Sigma.dtype, device=Sigma.device)
        else:
            Sigma = (Sigma + Sigma.T) / 2
            max_eig = np.linalg.norm(Sigma, 2)
            eps = eps_ratio * max_eig
            Sigma_shrink = Sigma + eps * np.eye(Sigma.shape[0], dtype=Sigma.dtype)
        shrink_sigma_dict[cls_name] = Sigma_shrink

    return shrink_sigma_dict


def tikhonov_shrinkage_preserve_type_val(Sigma, eps_ratio=1e-6):
    """
    对每个协方差矩阵做 Tikhonov / Ridge 收缩，同时保持原数据类型
    sigma_dict: dict
        key: 类别名称
        value: 协方差矩阵 (np.ndarray 或 torch.Tensor)
    eps_ratio: 正则化比例，eps = eps_ratio * max_eigval
    返回:
        shrink_sigma_dict: dict, 数据类型与输入一致
    """
    is_torch = isinstance(Sigma, torch.Tensor)
        
    if is_torch:
        Sigma = Sigma.double()  # 提升精度计算
        Sigma = (Sigma + Sigma.T) / 2
        # 计算最大特征值
        eigvals = torch.linalg.eigvalsh(Sigma)
        max_eig = torch.max(eigvals).item()
        eps = eps_ratio * max_eig
        Sigma_shrink = Sigma + eps * torch.eye(Sigma.shape[0], dtype=Sigma.dtype, device=Sigma.device)
        Sigma_shrink = Sigma_shrink.to(dtype=Sigma.dtype)  # 保持原 dtype
    else:
        Sigma = np.asarray(Sigma, dtype=np.float64)
        Sigma = (Sigma + Sigma.T) / 2
        eigvals = np.linalg.eigvalsh(Sigma)
        max_eig = np.max(eigvals)
        eps = eps_ratio * max_eig
        Sigma_shrink = Sigma + eps * np.eye(Sigma.shape[0])
        Sigma_shrink = Sigma_shrink.astype(Sigma.dtype)  # 保持原 dtype

    
    return Sigma_shrink

def scale_sigma_dict_txt2img_congruence4_noclip(
    sigma_txt_dict,      # {classname -> [D,D] Tensor} 文本协方差字典
    Sigma_img,           # [D,D] Tensor，参考类别图像协方差
    ref_class=None,      # str，可选，指定参考类别文本协方差
    mode="diag",         # 目前仅支持 "diag"
    beta=0.0,            # 类间平滑权重
    eps=1e-12,
    weak_dim_ratio=0.1,  # 对极端维度比例 top-k 弱化
    weak_slope=0.2       # 平滑函数斜率
):
    """
    文本协方差 → 图像域（对角线映射 + alpha + r_i + 极端维度平滑 + 类归一化）
    去掉了方差比率裁剪（clipping）。
    
    极端维度平滑：解决 局部维度方差失衡，防止个别异常维度主导距离度量；

    类归一化：解决 类别整体方差规模差异，保证马氏距离在跨类对比中具备一致性。
    """
    device = Sigma_img.device
    D = Sigma_img.size(0)

    if ref_class is None:
        ref_class = next(iter(sigma_txt_dict))
    Sigma_txt_ref = sigma_txt_dict[ref_class].to(device)

    if mode != "diag":
        raise NotImplementedError("仅支持 diag")

    # 1. alpha 全局缩放
    diag_txt_ref = torch.clamp(torch.diag(Sigma_txt_ref), min=eps)
    diag_img_ref = torch.clamp(torch.diag(Sigma_img), min=eps)
    alpha = diag_img_ref.mean() / diag_txt_ref.mean()

    # 2. 文本域相对缩放 r_i
    r_dict = {}
    for c, Sigma_txt in sigma_txt_dict.items():
        diag_txt = torch.clamp(torch.diag(Sigma_txt.to(device)), min=eps)
        r_dict[c] = diag_txt / diag_txt_ref

    # 3. 初步映射
    diag_mapped_dict = {}
    for c in sigma_txt_dict:
        diag_mapped = alpha * r_dict[c] * diag_img_ref
        diag_mapped_dict[c] = diag_mapped

    # 4. 极端维度平滑 + 类归一化
    all_ratios = torch.stack([diag_mapped_dict[c] / diag_img_ref for c in diag_mapped_dict], dim=0)
    mean_ratio = all_ratios.mean(dim=0)
    k_extreme = max(1, int(D * weak_dim_ratio))
    extreme_dims = torch.topk(torch.abs(torch.log(mean_ratio + eps)), k=k_extreme).indices

    for c in diag_mapped_dict:
        diag_vec = diag_mapped_dict[c]
        ratio = diag_vec / diag_img_ref
        # --- 去掉裁剪 ---
        # 平滑弱化极端维度
        diag_vec[extreme_dims] = diag_vec[extreme_dims] / (
            1.0 + weak_slope * torch.abs(torch.log(ratio[extreme_dims] + eps))
        )
        # 类归一化
        diag_vec = diag_vec / diag_vec.mean() * diag_img_ref.mean()
        diag_mapped_dict[c] = diag_vec

    # 5. 构造对角矩阵输出
    sigma_txt_adjusted = {c: torch.diag(diag_mapped_dict[c]) for c in diag_mapped_dict}
    return sigma_txt_adjusted

def compute_mahalanobis_loss_for_class(
    mu_all, sigma_dict, classnames, cls_name, temperature=1.0
):
    """
    只计算指定类别 cls_name 与其他类别的马氏距离损失（对角协方差）。
    
    输入:
        mu_all: [C,D] Tensor，每行是类别均值 (需要梯度)
        sigma_dict: {classname: [D,D]} 协方差矩阵 (不参与梯度)
        classnames: list[str] 类别名，顺序与 mu_all 一致
        cls_name: str，只计算该类别与其他类别的损失
        temperature: softmax 温度
    
    输出:
        loss: 标量 Tensor，该类别的损失
    """
    C, D = mu_all.shape
    device = mu_all.device

    # --- 找到该类别的索引 ---
    if cls_name not in classnames:
        raise ValueError(f"{cls_name} not in classnames.")
    idx = classnames.index(cls_name)

    # --- 取该类别的对角协方差 ---
    sigma_diag = torch.diag(sigma_dict[cls_name]).detach().to(device)  # [D]

    # --- 计算该类别与所有类别的马氏距离 ---
    diff = mu_all[idx].unsqueeze(0) - mu_all   # [C,D]
    inv_var = 1.0 / (sigma_diag + 1e-12)       # [D]
    dists = torch.sum(diff**2 * inv_var, dim=1)  # [C]

    # 去掉自身
    mask = torch.ones(C, dtype=torch.bool, device=device)
    mask[idx] = False
    dists = dists[mask]   # [C-1]

    # --- softmax 权重 ---
    weights = torch.softmax(-dists / temperature, dim=0)
    loss = torch.sum(dists)  # 标量

    return loss
def compute_mean_mahalanobis_loss_diag_softmax(
    mu_all, sigma_dict, classnames, temperature=1.0
):
    """
    用对角协方差计算 Mahalanobis 距离，协方差不参与梯度。
    输入:
        mu_all: [C,D] Tensor，每行是类别均值 (需要梯度)
        sigma_dict: {classname: [D,D]} 协方差矩阵 (不参与梯度)
        classnames: list[str] 类别名
        temperature: softmax 温度
    输出:
        loss: [C] Tensor
    """
    C, D = mu_all.shape
    device = mu_all.device


    # --- 取对角协方差并 detach ---
    sigma_diag_dict = {}
    for c in classnames:
        diag = torch.diag(sigma_dict[c]).detach()   # 切断梯度
        sigma_diag_dict[c] = diag.to(device)


    # --- 构造马氏距离 ---
    loss_list = []
    for i, ci in enumerate(classnames):
        diff = mu_all[i].unsqueeze(0) - mu_all      # [C,D]
        inv_var = 1.0 / (sigma_diag_dict[ci] + 1e-12)  # [D]，已无梯度
        dists = torch.sum(diff**2 * inv_var, dim=1)    # [C]

        # 去掉自身
        mask = torch.ones(C, dtype=torch.bool, device=device)
        mask[i] = False
        dists = dists[mask]  # [C-1]

        # softmax 权重（距离越小权重大）
        weights = torch.softmax(-dists / temperature, dim=0).detach()
        loss_list.append(torch.sum(dists))
#         loss_list.append(torch.sum(dists))

    loss = torch.stack(loss_list)  # [C]
    return loss


def compute_mean_mahalanobis_loss_diag_rank(
    mu_all, sigma_dict, classnames, alpha=1.0
):
    """
    用对角协方差计算 Mahalanobis 距离，协方差不参与梯度。
    使用 rank-based 权重代替 softmax。
    
    输入:
        mu_all: [C,D] Tensor，每行是类别均值 (需要梯度)
        sigma_dict: {classname: [D,D]} 协方差矩阵 (不参与梯度)
        classnames: list[str] 类别名
        alpha: float，rank 权重调节参数，越大越强调最近邻
    
    输出:
        loss: [C] Tensor
    """
    C, D = mu_all.shape
    device = mu_all.device

    # --- 取对角协方差并 detach ---
    sigma_diag_dict = {c: torch.diag(sigma_dict[c]).detach().to(device) for c in classnames}

    loss_list = []
    for i, ci in enumerate(classnames):
        diff = mu_all[i].unsqueeze(0) - mu_all      # [C,D]
        inv_var = 1.0 / (sigma_diag_dict[ci] + 1e-12)  # [D]
        dists = torch.sum(diff**2 * inv_var, dim=1)    # [C]

        # 去掉自身
        mask = torch.ones(C, dtype=torch.bool, device=device)
        mask[i] = False
        dists = dists[mask]  # [C-1]

        # --- rank-based 权重 ---
        ranks = torch.argsort(torch.argsort(dists))  # rank: 0 最近，C-2 最远
        weights = (C-1 - ranks).float() ** alpha      # 离得近的权重大
        weights = weights / weights.sum()             # 归一化

        loss_list.append(torch.sum(weights * dists))

    loss = torch.stack(loss_list)  # [C]
    return loss
def class_stats(mu_all_dict, sigma_diag_dict):
    """
    计算每个类别的类间最小距离、类间平均距离和类内方差。
    
    输入:
        mu_all_dict: dict {classname: [D] Tensor} 每类均值
        sigma_diag_dict: dict {classname: [D] Tensor} 每类对角协方差
    输出:
        mins: np.array [C] 每类与其他类的最小距离
        means: np.array [C] 每类与其他类的平均距离
        intra: np.array [C] 每类自身的平均方差
    """
    classnames = list(mu_all_dict.keys())
    C = len(classnames)
    
    mins = []
    means = []
    intra = []
    
    for i, ci in enumerate(classnames):
        mu_i = mu_all_dict[ci]           # [D]
        sigma_i = sigma_diag_dict[ci]    # [D]
        
        dists = []
        for j, cj in enumerate(classnames):
            if i == j:
                continue
            mu_j = mu_all_dict[cj]
            # 对角马氏距离
            inv_var = 1.0 / (sigma_i + 1e-6)
            dist = torch.sum((mu_i.cpu() - mu_j.cpu())**2 * inv_var.cpu())
            dists.append(dist.item())
        
        dists = np.array(dists)
        mins.append(dists.min())
        means.append(dists.mean())
        intra.append(sigma_i.mean().item())
    
    return np.array(mins), np.array(means), np.array(intra)
def compute_mean_euclidean_loss_softmax(
    mu_all, classnames, temperature=1.0
):
    """
    计算类别中心之间的欧式距离 (L2)，并做 softmax 加权损失。
    输入:
        mu_all: [C,D] Tensor，每行是类别均值 (需要梯度)
        classnames: list[str] 类别名
        temperature: softmax 温度
    输出:
        loss: [C] Tensor
    """
    C, D = mu_all.shape
    device = mu_all.device

    loss_list = []
    for i in range(C):
        diff = mu_all[i].unsqueeze(0) - mu_all      # [C,D]
        dists = torch.sum(diff**2, dim=1)           # [C] 欧式距离平方

        # 去掉自身
        mask = torch.ones(C, dtype=torch.bool, device=device)
        mask[i] = False
        dists = dists[mask]  # [C-1]

        # softmax 权重（距离越小权重大）
        weights = torch.softmax(-dists / temperature, dim=0)

        # 👉 这里有两种选择：
        # 1. 加权和: torch.sum(weights * dists)
        # 2. 简单平均: torch.mean(dists)
        loss_list.append(torch.sum(weights * dists))

    loss = torch.stack(loss_list)  # [C]
    return loss

import torch



import torch.nn.functional as F

def test_time_tuning(mu_dict, clip_output, clip_outputs, clip_features, classnames, sigma_dict, model, inputs, optimizer, scaler, args):
    selected_idx = None
    
    
    for j in range(args.tta_steps):
        if True:
            output, txt_feat = model(inputs, txtlabel=True)

    
            
            pseudo_idx = int(clip_output.argmax().item())
            cls_name = classnames[pseudo_idx]
            conv = safe_cov_from_features(clip_features, mu=mu_dict[cls_name], eps=1e-6)
            
#             sigma_img = tikhonov_shrinkage_preserve_type_val(conv)
#             sigma_txt = tikhonov_shrinkage_preserve_type(sigma_dict)
            sigma_img = conv
            sigma_txt = sigma_dict
    
   
            out = scale_sigma_dict_txt2img_congruence4_noclip(sigma_txt, sigma_img, ref_class=cls_name)
            
            loss_dis = compute_mean_mahalanobis_loss_diag_softmax(txt_feat, out, classnames).sum()


            
   

            
            
            #loss_dis = compute_mean_euclidean_loss_softmax(txt_feat, classnames).sum()
            C = len(classnames)





            if selected_idx is not None:
                output = output[selected_idx]
            else:
                output, selected_idx = select_confident_samples(output, args.selection_p)
                #print(torch.max(F.softmax(output, dim=1)))

#             loss =  entropy_avg(output) - 1  * loss_dis / torch.log(torch.tensor(C, dtype=loss_dis.dtype, device=loss_dis.device) + 1)
                probs = F.softmax(output, dim=1)  # [N, C]
    
                # 平均置信度
                confs = probs.max(dim=1).values  # [N]
           

  

         
#                 loss = (1 - (1 - torch.max(confs)).detach() ** 2) * entropy_avg(output) - (1 - torch.max(confs)).detach() ** 2 * loss_dis / #(C * torch.log2(torch.tensor(C, dtype=loss_dis.dtype, device=loss_dis.device)))
#                 loss = (1 - (1 - torch.max(confs)).detach() ** 2) * entropy_avg(output) - 0.005 * (1 - torch.max(confs)).detach() ** 2 * loss_dis / (1024 * (torch.log(torch.tensor(C, dtype=loss_dis.dtype, device=loss_dis.device)) +1))
                loss = entropy_avg(output)
                
                
                
                
                
                # O-TPT
                lambda_ = 18

                number_of_class = len(classnames)           
                #------------------------------------------------- Householder Transform--------
                text_feature = txt_feat
            #print("text feature shape model:",text_feature.shape)
            #computing orthogonal constrained  SVD
                Wwt  =  torch.matmul(text_feature,text_feature.T)
                wwt_norm_col_HT = torch.linalg.norm(Wwt,dim=-1)
                Wwt_val_HT = wwt_norm_col_HT.mean()
            #wtW  =  torch.matmul(text_feature.T,text_feature)
                e = torch.eye(Wwt.shape[1], device=args.gpu)
                M_norm = torch.linalg.norm(Wwt, dim=0,keepdim=True)
                scaled_e = e * M_norm
            # Subtract the scaled identity matrix from Wwt
                u = Wwt - scaled_e
                u_norm = torch.linalg.norm(u, dim=-1,keepdim=True)
            #u_norm = u_norm ** 2
            # We need to expand u_norm to shape (47, 47, 1) for broadcasting
            #u_norm_exp = u_norm.unsqueeze(2)  # Shape: (1, 47, 1)
            
            #Transposing the u for batch element column and coresponding column transpose matrix multiplication
          
                v = u/u_norm
                normalized_matrix_exp = v.unsqueeze(2)  # Shape: (47, 47, 1)
                normalized_matrix_T_exp = v.unsqueeze(1)  # Shape: (47, 1, 47)
            
            # This will create a batch of 3 matrices, each of shape (47, 47)
                outer_products = normalized_matrix_exp @ normalized_matrix_T_exp  # Shape: (47, 47, 47)
            
            # Perform element-wise division of each outer product by the corresponding u_norm value
                divided_matrix = outer_products #/ u_norm_exp  # Shape: (47, 47, 47)

            # Multiply the result by 2
                scaled_matrix = 2 * divided_matrix  # Shape: (47, 47, 47)
            # Subtract the scaled result from the corresponding identity matrix for each batch
                identity_matrix_dim = e.unsqueeze(0).expand(Wwt.shape[1], -1, -1)  # Shape: (47, 47, 47)
            # Subtract from identity matrix
                transformed_matrix = identity_matrix_dim - scaled_matrix  # Shape: (47, 47, 47)
            # Reshape M so that its columns are aligned for batch multiplication
                Wwt_exp = Wwt.unsqueeze(2)  # Shape: (47, 47, 1)

            # Perform batched matrix multiplication between transformed matrix and M_exp
                Hx = torch.bmm(transformed_matrix, Wwt_exp)  # Shape: (47, 47, 1)
            
            # Reshape back the result to (3, 3) by removing the last singleton dimension
                Hx = Hx.squeeze(2)  # Shape: (47, 47)
            #print("shape of Hx:",Hx.shape)
            #print("Hx:",Hx)
            #normalizing Column wise
            #Hx_norm =torch.linalg.norm(Hx, dim=0,keepdim=True)
            #Hx = Hx/Hx_norm
                Ht_ortho = Hx - e  
                Ht_ortho_norm = torch.linalg.norm(Ht_ortho, dim=-1)
                Ht_ortho_norm_val = Ht_ortho_norm.mean()
                loss += (+(lambda_ * Ht_ortho_norm_val))
                



        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return



def compute_ece(logits, labels, n_bins=15):
    """
    logits: [N, C] 模型输出（未经过 softmax）
    labels: [N]    真实标签
    n_bins: 分桶数量
    """
    probs = torch.softmax(logits, dim=1)          # [N, C]
    conf, preds = torch.max(probs, dim=1)         # [N] 最大置信度和预测类别

    ece = torch.zeros(1, device=logits.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (conf > bin_lower) & (conf <= bin_upper)

        if mask.any():
            acc_in_bin = (preds[mask] == labels[mask]).float().mean()
            avg_conf_in_bin = conf[mask].mean()
            prop_in_bin = mask.float().mean()

            ece += torch.abs(avg_conf_in_bin - acc_in_bin) * prop_in_bin

    return ece.item()


templates = [
    "a photo of a {}.",
    "a picture of a {}.",
    "this is a photo of a {}.",
    "an image of a {}.",
    "a photo showing a {}.",
    "a photo of a {} object.",
    "a cropped photo of a {}.",
    "a close-up of a {}.",
    "the photo of a {}.",
    "a photo of a single {}.",
    "a typical photo of a {}.",
    "a photo of a {} in the dataset.",
    "a natural photo of a {}.",
    "a good photo of a {}.",
    "a real photo of a {}.",
    "a digital photo of a {}.",
    "a simple photo of a {}.",
    "a standard photo of a {}.",
    "an image showing a {}.",
    "a clear photo of a {}.",
    "a centered photo of a {}.",
    "a focused photo of a {}.",
    "a photo capturing a {}.",
    "a close-up picture of a {}.",
    "a cropped picture of a {}.",
    "a high quality photo of a {}.",
    "a bright picture of a {}.",
    "a dark picture of a {}.",
    "a photo illustrating a {}.",
    "a photo highlighting a {}.",
    "a photo representing a {}.",
    "a photo depicting a {}.",
    "an image depicting a {}.",
    "a picture showing a {}.",
    "a picture of the {}.",
    "this is an image of a {}.",
    "an image capturing a {}.",
    "a real image of a {}.",
    "a simple image of a {}.",
    "a focused image of a {}.",
]

import pandas as pd
def covariance_similarity(sigma_img, sigma_txt, mode="diag"):
    """
    输入:
        sigma_img: dict {classname -> [D,D] Tensor}
        sigma_txt: dict {classname -> [D,D] Tensor}
        mode: "diag" 只比较对角线（方差）
              "full" 比较整个协方差矩阵
              "frobenius" Frobenius 距离
              "logdet" Log-Det Divergence

    输出:
        DataFrame: 每个类别的相关性或距离值
    """
    results = []
    for cls in sigma_img:
        if cls not in sigma_txt:
            continue

        A = sigma_img[cls]
        B = sigma_txt[cls]

        if mode == "diag":
            a = torch.diag(A)
            b = torch.diag(B)
            corr = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
        elif mode == "full":
            corr = torch.corrcoef(torch.stack([A.flatten(), B.flatten()]))[0, 1].item()
        elif mode == "frobenius":
            diff = A - B
            corr = -torch.norm(diff, p="fro").item()  # 距离取负号表示越大越相似
        elif mode == "logdet":
            # 对称正定矩阵时可用
            M = 0.5 * (A + B)
            try:
                corr = torch.logdet(M) - 0.5 * (torch.logdet(A) + torch.logdet(B))
                corr = -corr.item()  # 越小越相似
            except RuntimeError:
                corr = float("nan")
        else:
            raise ValueError(f"Unknown mode: {mode}")

        results.append({"class": cls, "score": corr})

    return pd.DataFrame(results)

def get_mu_sigma_batched_mu(model, classnames, templates=templates, default_template="a photo of a {}"):
    device = model.device
    clip_model = model.clip
    text_encoder = model.text_encoder
    d = text_encoder.text_projection.shape[1]
    model.eval()

    # ----- mu (默认模板) -----
    default_texts = [default_template.format(c.replace("_", " ")) for c in classnames]

    toks = tokenize(default_texts).to(device)
    with torch.no_grad():
        emb = clip_model.token_embedding(toks).type(text_encoder.dtype)
        x = emb + text_encoder.positional_embedding.type(text_encoder.dtype)
        x = x.permute(1, 0, 2)
        x = text_encoder.transformer(x)
        x = x.permute(1, 0, 2)
        x = text_encoder.ln_final(x).type(text_encoder.dtype)
        feats = x[torch.arange(x.shape[0]), toks.argmax(dim=-1)] @ text_encoder.text_projection
        feats = feats / feats.norm(dim=-1, keepdim=True)


    #mu_dict = {c: feats[i].cpu().numpy() for i, c in enumerate(classnames)}
    mu_dict_tensor = {c: feats[i] for i, c in enumerate(classnames)}


    return mu_dict_tensor

def get_mu_sigma_batched_sigma(
    model,
    classnames,
    prompt_json_path="/media/cqu/D/FXV/R-TPT-main6/dtd_prompts.json",
    default_template="a photo of a {}"
):
    """
    读取 dtd_prompts.json 中每个类别的文本模板（并加入默认模板），
    通过 CLIP 文本编码得到特征后为每个类别估计协方差矩阵。
    返回值形式与原函数一致：sigma_dict = compute_ensemble_covariance(features_per_class)

    依赖：
      - tokenize: 需与原有代码一致（例如 from clip import tokenize 或 open_clip.tokenize）
      - compute_ensemble_covariance(features_per_class): 外部提供
    """
    device = model.device
    model.eval()
    clip_model = model.clip
    text_encoder = model.text_encoder
    d = text_encoder.text_projection.shape[1]

    # ---- 读取 JSON ----
    with open(prompt_json_path, "r", encoding="utf-8") as f:
        prompt_dict = json.load(f)


    def _find_key_for_class(cls: str):
        """兼容 'banded' 或 'DTD_banded' 两种键名"""
        if cls in prompt_dict:
            return cls
        alt = f"DTD_{cls}"
        return alt if alt in prompt_dict else None

    def _clean_and_filter(prompts, cls: str):
        """过滤无效行：必须以 'A photo of a' 开头且包含类别名；去重并保序"""
        if not isinstance(prompts, list):
            return []
        cls_l = cls.lower()
        seen = set()
        kept = []
        for s in prompts:
            if not isinstance(s, str):
                continue
            t = s.strip()
            if not t:
                continue
            kept.append(t)
#             tl = t.lower()
#             if (tl.startswith("a photo of a")
#                 and cls_l in tl
#                 and not tl.endswith("a photo of a")):
#                 if t not in seen:
#                     seen.add(t)
                    
        return kept

    features_per_class = {}

    for cls in classnames:
        cls_name = cls.replace("_", " ")

        # 1) 默认模板放首位（作为中心模板参与协方差估计）
        merged_templates = [default_template.format(cls_name)]

        # 2) 读取 JSON 中的该类模板并清洗
        key = _find_key_for_class(cls)
        if key is not None:
            cleaned = _clean_and_filter(prompt_dict.get(key, []), cls)
            # 去重（保留顺序）
            for t in cleaned:
                if t not in merged_templates:
                    merged_templates.append(t)
        else:
            print(f"[Warning] '{cls}' not found in {prompt_json_path}; only using default template.")

        # 若清洗后只有默认模板，也允许继续；无需强行兜底固定句式
      

        # ---- 编码 ----
        toks = tokenize(merged_templates).to(device)
        with torch.no_grad():
            emb = clip_model.token_embedding(toks).type(text_encoder.dtype)
            x = emb + text_encoder.positional_embedding.type(text_encoder.dtype)
            x = x.permute(1, 0, 2)
            x = text_encoder.transformer(x)
            x = x.permute(1, 0, 2)
            x = text_encoder.ln_final(x).type(text_encoder.dtype)
            # 取 EOT 位置的表示并乘投影
            feats = x[torch.arange(x.shape[0]), toks.argmax(dim=-1)] @ text_encoder.text_projection
            feats = feats / feats.norm(dim=-1, keepdim=True)   # 归一化 [T, D]
            if torch.isnan(feats).any() or torch.isinf(feats).any():
                print(f"NaN/Inf detected in features for class {cls}")


        # 存为 {cls: [T, D]}
        features_per_class[cls] = feats
    print(feats.size())
    # ---- 计算协方差 ----
    sigma_dict = compute_ensemble_covariance_new(features_per_class)
    return sigma_dict
# def get_mu_sigma_batched_sigma(model, classnames, templates=templates, default_template="a photo of a {}"):
# 
#     # ----- sigma (多模板) -----
#     device = model.device
#     clip_model = model.clip
#     text_encoder = model.text_encoder
#     d = text_encoder.text_projection.shape[1]
#     all_texts = []
#     for c in classnames:
#         all_texts.extend([temp.format(c.replace("_", " ")) for temp in templates])
#     toks = tokenize(all_texts).to(device)
# 
#     with torch.no_grad():
#         emb = clip_model.token_embedding(toks).type(text_encoder.dtype)
#         x = emb + text_encoder.positional_embedding.type(text_encoder.dtype)
#         x = x.permute(1, 0, 2)
#         x = text_encoder.transformer(x)
#         x = x.permute(1, 0, 2)
#         x = text_encoder.ln_final(x).type(text_encoder.dtype)
#         feats = x[torch.arange(x.shape[0]), toks.argmax(dim=-1)] @ text_encoder.text_projection
#         feats = feats / feats.norm(dim=-1, keepdim=True)
# 
#     num_templates = len(templates)
#     feats = feats.view(len(classnames), num_templates, d)  # [C, T, D]
#     # ----- 构造 dict -----
#     features_per_class = {}
#     for i, cls in enumerate(classnames):
#         features_per_class[cls] = feats[i]  # [T, D]
#     sigma_dict = (features_per_class)
# 
#     return sigma_dict

# ---------- Step 2: 马氏距离 ----------
def mahalanobis_distance(mu_i, mu_j, sigma_i):
    diff = mu_i - mu_j
    inv_sigma = np.linalg.pinv(sigma_i)
    return np.sqrt(diff.T @ inv_sigma @ diff)

# def compute_mean_mahalanobis(mu_dict, sigma_dict, eps=1e-6):
#     """
#     向量化计算每个类别的平均 Mahalanobis 距离 (PyTorch 版)
#     - mu_dict: {class -> Tensor[d]}，有梯度
#     - sigma_dict: {class -> Tensor[d, d]}，已固定 (detach)
#     - return: {class -> Tensor}，每个类别的平均距离 (可做 loss)
#     """
#     classes = list(mu_dict.keys())
#     C = len(classes)
#     device = next(iter(mu_dict.values())).device
# 
#     # 堆叠所有 mu: [C, D]
#     mus = torch.stack([mu_dict[c] for c in classes], dim=0)  
# 
#     dist_dict = {}
#     for i, ci in enumerate(classes):
#         mu_i = mus[i]  # [D]
#         sigma_i = sigma_dict[ci] + eps * torch.eye(sigma_dict[ci].size(0), device=device)
#         inv_sigma_i = torch.inverse(sigma_i)  # [D, D]
# 
#         # 批量计算 diff: [C, D]
#         diff = mus - mu_i
# 
#         # einsum 实现: diff @ inv_sigma_i @ diff^T -> [C]
#         dists_sq = torch.einsum("nd,de,ne->n", diff, inv_sigma_i, diff)  # [C]
#         dists = torch.sqrt(dists_sq + eps)  # [C]
# 
#         # 去掉自身
#         dists = torch.cat([dists[:i], dists[i+1:]])
#         dist_dict[ci] = dists.mean()  # scalar tensor
# 
#     return dist_dict


def compute_mean_mahalanobis(mu_dict, sigma_dict, eps=1e-6, mode="full"):
    """
    向量化计算每个类别的平均 Mahalanobis 距离 (PyTorch 版)

    参数:
    - mu_dict: {class -> Tensor[d]}，有梯度
    - sigma_dict: {class -> Tensor[d, d]}，已固定 (detach)
    - eps: float, 防止数值不稳定
    - mode: "full" 使用完整协方差, "diag" 只用对角线方差

    返回:
    - {class -> Tensor}，每个类别的平均 Mahalanobis 距离
    """
    classes = list(mu_dict.keys())
    C = len(classes)

    # 堆叠所有 mu: [C, D]（保持原有 device）
    mus = torch.stack([mu_dict[c] for c in classes], dim=0)  

    dist_dict = {}
    for i, ci in enumerate(classes):
        mu_i = mus[i]  # [D]
        sigma_i = sigma_dict[ci]

        # 确保 mu 和 sigma 在同一设备
        device = sigma_i.device
        mu_i = mu_i.to(device)
        mus_local = mus.to(device)

        if mode == "diag":
            # 只保留对角线，构造对角矩阵
            diag_var = torch.diag(torch.diag(sigma_i))
            sigma_i = diag_var

        # 加上 eps I 避免奇异
        sigma_i = sigma_i + eps * torch.eye(sigma_i.size(0), device=device)

        inv_sigma_i = torch.inverse(sigma_i)  # [D, D]

        # 批量计算 diff: [C, D]
        diff = mus_local - mu_i

        # einsum 实现: diff @ inv_sigma_i @ diff^T -> [C]
        dists_sq = torch.einsum("nd,de,ne->n", diff, inv_sigma_i, diff)  # [C]
        dists = torch.sqrt(dists_sq + eps)  # [C]

        # 去掉自身
        dists = torch.cat([dists[:i], dists[i+1:]])
        dist_dict[ci] = dists.mean()  # scalar tensor

    return dist_dict


# ---------- Step 3: ECE (真实标签分布) ----------
def compute_classwise_ece_true(logits, labels, n_bins=15):
    probs = softmax(logits, dim=1)
    conf, preds = torch.max(probs, dim=1)

    classes = torch.unique(labels)
    ece_dict = {}

    for c in classes:
        mask = labels == c
        if mask.sum() == 0:
            continue
        conf_c, preds_c, labels_c = conf[mask], preds[mask], labels[mask]

        ece_c = torch.zeros(1, device=logits.device)
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)

        for i in range(n_bins):
            bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i+1]
            bin_mask = (conf_c > bin_lower) & (conf_c <= bin_upper)
            if bin_mask.any():
                acc_bin = (preds_c[bin_mask] == labels_c[bin_mask]).float().mean()
                conf_bin = conf_c[bin_mask].mean()
                prop_bin = bin_mask.float().mean()
                ece_c += torch.abs(conf_bin - acc_bin) * prop_bin

        ece_dict[int(c.item())] = ece_c.item()

    return ece_dict

# ---------- Step 4: 整合 (批量版) ---------


  
    
def plot_mahalanobis_vs_ece(results, save_path=None, annotate_topk=1, add_regression=True):
    """
    绘制 Mahalanobis 距离 vs ECE 散点图（带回归线 & 相关系数）
    - results: analyze_clip_calibration 的输出
    - annotate_topk: 标注 ECE 最大/最小的类别，避免过乱
    - add_regression: 是否加回归线
    """
    xs, ys, labels = [], [], []
    for r in results:
        if r["mahalanobis"] is not None and r["ece"] is not None:
            # 兼容 tensor / numpy / float
            x = r["mahalanobis"]
            y = r["ece"]
            if hasattr(x, "detach"):  # tensor
                x = x.detach().cpu().numpy()
            if hasattr(y, "detach"):
                y = y.detach().cpu().numpy()
            xs.append(float(x))
            ys.append(float(y))
            labels.append(r["class"])

    xs, ys = np.array(xs), np.array(ys)

    # 计算相关性
    pearson_corr, pearson_p = pearsonr(xs, ys)
    spearman_corr, spearman_p = spearmanr(xs, ys)

    # 绘制散点图
    plt.figure(figsize=(7.5, 6))
    plt.scatter(xs, ys, alpha=0.7, s=40, c="tab:blue", edgecolor="k", linewidth=0.5)

    # 添加回归线
    if add_regression and len(xs) > 1:
        coeffs = np.polyfit(xs, ys, 1)   # 一阶拟合
        x_fit = np.linspace(xs.min(), xs.max(), 200)
        y_fit = np.polyval(coeffs, x_fit)
        plt.plot(x_fit, y_fit, "r--", linewidth=2, label="Linear Fit")
        plt.legend(fontsize=10)

    # 标注 top-k 类别（ECE 最大和最小）
    if annotate_topk > 0 and len(xs) > 2 * annotate_topk:
        idx_sorted = np.argsort(ys)
        highlight_idx = list(idx_sorted[:annotate_topk]) + list(idx_sorted[-annotate_topk:])
        for i in highlight_idx:
            plt.text(xs[i], ys[i], labels[i], fontsize=11, ha="right", va="bottom")

    # 添加相关性信息
    textstr = "\n".join((
        rf"$\mathrm{{Pearson}}\ r = {pearson_corr:.3f}$",
        rf"$\mathrm{{Spearman}}\ \rho = {spearman_corr:.3f}$"
    ))
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                   fontsize=11, verticalalignment="top",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    # 设置标签和样式
    plt.xlabel("Mean Mahalanobis Distance", fontsize=13)
    plt.ylabel("Class-wise ECE", fontsize=13)
    plt.title("Calibration vs Separation", fontsize=14, weight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        



def compute_ensemble_covarianceimg(mu_dict,  features_per_class, eps=1e-6):
    """
    features_per_class: dict {cls -> [N,D] Tensor} 或 list of [D] Tensor
    mu_dict: dict {cls -> [D] Tensor}，可选，如果提供，用作中心
    返回: sigma_dict {cls -> [D,D]}
    """
    sigma_dict = {}
    center_dict = {}

    for cls, feats_list in features_per_class.items():
        # list -> [N,D]
        feats = torch.stack(feats_list, dim=0) if isinstance(feats_list, list) else feats_list
        N, D = feats.shape


        # 检查 NaN / Inf
        if torch.isnan(feats).any() or torch.isinf(feats).any():
            feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        if N < 2:
            sigma = torch.eye(D) * eps
        else:
            # 使用提供的 mu 作为中心，否则用自身均值
            if mu_dict is not None and cls in mu_dict:
                center = mu_dict[cls].to(feats.device)
                #center = feats.mean(dim=0)
            else:
                center = feats.mean(dim=0)

            feats_centered = feats - center
            
            sigma = (feats_centered.T @ feats_centered) / (N - 1)
            sigma = sigma + eps * torch.eye(D)

        sigma_dict[cls] = sigma
        center_dict[cls] = center

    return sigma_dict, center_dict

def compute_ensemble_covarianceimg1(mu_dict,  features_per_class, eps=1e-6):
    """
    输入:
        features_per_class: dict {classname -> list of [D] Tensor}
    输出:
        sigma_dict: dict {classname -> [D,D] 协方差矩阵}
    """
    sigma_dict = {}

    for cls, feats_list in features_per_class.items():
        # 把 list 堆叠成 [N,D]
        feats = torch.stack(feats_list, dim=0)  # [N, D]
        N, D = feats.shape

        # 检查 NaN / Inf
        if torch.isnan(feats).any() or torch.isinf(feats).any():
            print(f"[Warning] NaN/Inf detected in features of class {cls}, replacing with zeros.")
            feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        if N < 2:
            # 样本不足 -> 给单位阵
            sigma = torch.eye(D) * eps
        else:
            # 中心化
            #feats_centered = feats - feats.mean(dim=0, keepdim=True)
            feats_centered = mu_dict
            # 协方差 (N-1 归一化)
            sigma = (feats_centered.T @ feats_centered) / (N - 1)
            # 数值正则，避免奇异
            sigma = sigma + eps * torch.eye(D)

        sigma_dict[cls] = sigma

    return sigma_dict


def compute_ensemble_covariance(features_per_class, eps_scale=0.5, device='cpu'):
    """
    按类别计算协方差矩阵，不再划分子集。
    - features_per_class: dict {classname -> [N,D] Tensor}
    - eps_scale: 缩放正则强度
    - return: sigma_dict {classname -> [D,D]}
    """
    sigma_dict = {}
    for cls, feats in features_per_class.items():
        feats = feats.to(device)
        N, D = feats.shape

        # 协方差矩阵 [D,D]
        cov = torch.cov(feats.T)
        

        # 自动 eps，避免奇异
        avg_var = torch.diag(cov).mean()
        eps = max(float(eps_scale * avg_var), 1e-6)
        cov_reg = cov + eps * torch.eye(D, device=device)

        sigma_dict[cls] = cov_reg

    return sigma_dict

def compute_ensemble_covariance_new(features_per_class, eps_scale=0.5, device='cpu'):
    """
    按类别计算协方差矩阵，不再划分子集。
    - features_per_class: dict {classname -> [N,D] Tensor}
    - eps_scale: 缩放正则强度
    - return: sigma_dict {classname -> [D,D]}
    """
    sigma_dict = {}
    for cls, feats in features_per_class.items():
        feats = feats.to(device)
        N, D = feats.shape

        # 以第一个特征作为类别中心
        center = feats[0:1]   # shape [1, D]


        # 去中心化 (基于第一个特征)
        centered = feats - center

        # 手动计算协方差矩阵 [D,D]
        cov = (centered.T @ centered) / (N - 1)

        # 自动 eps，避免奇异
        avg_var = torch.diag(cov).mean()
        eps = max(float(eps_scale * avg_var), 1e-6)
        cov_reg = cov + eps * torch.eye(D, device=device)

        sigma_dict[cls] = cov_reg

    return sigma_dict


def covariance_correlations_histogram(sigma_img, sigma_txt, out, save_path=None, bins=20):
    """
    按类别计算图像协方差矩阵与文本协方差矩阵的相关性，
    包括整体协方差和对角线相关性，并绘制频率直方图，显示平均值。

    参数：
    - sigma_img: dict {class_name -> Tensor[D, D]}
    - sigma_txt: dict {class_name -> Tensor[D, D]}
    - save_path: str, 可选，保存路径
    - bins: 直方图分箱数量
    """
    corr_full_list = []
    corr_diag_list = []
    corr_diag_list2 = []

    for cls in sigma_img:
        if cls not in sigma_txt:
            continue

        # 提取矩阵并统一到 CPU float
        A = sigma_img[cls].detach().cpu().float()
        B = sigma_txt[cls].detach().cpu().float()
        C = out[cls].detach().cpu().float()

        # 整体相关性
        A_flat = A.flatten()
        B_flat = B.flatten()
        C_flat = C.flatten()
        if torch.isnan(A_flat).any() or torch.isnan(B_flat).any() or torch.isnan(C_flat).any():
            corr_full = float('nan')
        else:
            corr_full = torch.corrcoef(torch.stack([A_flat, B_flat]))[0, 1].item()
        corr_full_list.append(corr_full)

        # 对角线相关性
        A_diag = torch.diag(A)
        B_diag = torch.diag(B)
        C_diag = torch.diag(C)
        if torch.isnan(A_diag).any() or torch.isnan(B_diag).any() or torch.isnan(C_diag).any():
            corr_diag = float('nan')
        else:
            corr_diag = torch.corrcoef(torch.stack([A_diag, B_diag]))[0, 1].item()
            corr_diag2 = torch.corrcoef(torch.stack([A_diag, C_diag]))[0, 1].item()
        corr_diag_list.append(corr_diag)
        corr_diag_list2.append(corr_diag2)

    # 转 numpy
    corr_full_list = np.array(corr_full_list)
    corr_diag_list = np.array(corr_diag_list)
    corr_diag_list2 = np.array(corr_diag_list2)

    # 计算均值
    mean_full = np.nanmean(corr_full_list)
    mean_diag = np.nanmean(corr_diag_list)
    mean_diag2 = np.nanmean(corr_diag_list2)

    # 绘制频率直方图
    plt.figure(figsize=(8,6))

# 原始版本的频率直方图
    plt.hist(
    corr_diag_list, 
    bins=bins, 
    weights=np.ones_like(corr_diag_list)/len(corr_diag_list),  # 频率
    alpha=0.6, 
    color='tab:blue', 
    label=f'Original Diagonal Covariance (mean={mean_diag:.3f})'
)

# 转换后版本的频率直方图
    plt.hist(
    corr_diag_list2, 
    bins=bins, 
    weights=np.ones_like(corr_diag_list2)/len(corr_diag_list2),  # 频率
    alpha=0.6, 
    color='tab:orange', 
    label=f'Diagonal Covariance (mean={mean_diag2:.3f})'
)

    plt.xlabel("Correlation between Image & Text Covariance", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Covariance Correlation Histogram", fontsize=14, weight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def compute_mean_mahalanobis_loss_auto_eps_dist_softmax(
    mu_all, sigma_dict, classnames, lambda_mix=0.0, eps_scale=0.5, temperature=1.0):
    """
    自动计算 eps + 混合协方差 + Mahalanobis 距离
    使用距离 softmax 权重（距离越小权重大），数值稳定
    输入:
        mu_all: [C,D] Tensor，每行是一个类别均值
        sigma_dict: {classname: [D,D]} 每类协方差矩阵
        classnames: list[str] 类别名，长度 C
        lambda_mix: float, 协方差混合比例
        eps_scale: float, eps = eps_scale * 平均方差
        temperature: float, softmax 温度
        rescale_log: bool, 是否对 loss 做 /log(C+1) 缩放
    输出:
        loss: [C] Tensor，每个类别的加权 Mahalanobis 距离
    """
    C, D = mu_all.shape
    device = mu_all.device

    # -----------------------------
    # 1. 自动计算 eps
    # -----------------------------
    # 堆叠每个类别协方差矩阵
    all_sigma = torch.stack([sigma_dict[c].to(device) for c in classnames], dim=0)  # [C,D,D]
    # 计算全局平均协方差
    Sigma_global = all_sigma.mean(0)  # [D,D]
    # 取对角线平均得到全局平均方差
    avg_var = torch.diag(Sigma_global).mean()
    # 自适应 eps，确保数值稳定
    eps = (eps_scale * avg_var).clamp_min(1e-12)

    # -----------------------------
    # 2. 混合协方差并求逆
    # -----------------------------
    sigma_mix_batch = torch.stack([
        lambda_mix * Sigma_global + (1 - lambda_mix) * sigma_dict[c].to(device)
        for c in classnames
    ], dim=0)  # [C,D,D]
    # 加上 eps 保证矩阵正定
    sigma_mix_batch = sigma_mix_batch + eps * torch.eye(D, device=device, dtype=mu_all.dtype)
    # batch 求逆
    M_batch = torch.linalg.inv(sigma_mix_batch)  # [C,D,D]

    # -----------------------------
    # 3. 计算 Mahalanobis 距离矩阵
    # -----------------------------
    diffs = mu_all.unsqueeze(0) - mu_all.unsqueeze(1)  # [C,C,D]
    temp = torch.bmm(diffs, M_batch)                  # [C,C,D]
    dists = torch.sum(temp * diffs, dim=-1)           # [C,C]

    # -----------------------------
    # 4. 去掉自身
    # -----------------------------
    mask = ~torch.eye(C, dtype=torch.bool, device=device)
    dists_masked = dists[mask].view(C, C-1)          # [C,C-1]

    # -----------------------------
    # 5. 距离 softmax 权重
    # -----------------------------
    weights = torch.softmax(-dists_masked / temperature, dim=1)  # [C,C-1]
    # -----------------------------
#     # 5. 统一权重 (均匀平均)
#     weights = torch.ones_like(dists_masked) / (C - 1)
    # -----------------------------
    # 6. 加权求和
    # -----------------------------
    loss = torch.sum(weights * dists_masked, dim=1)  # [C]


    return loss




# ---- 更新 sigma_img_dict（每类只写一次） ----
def update_sigma_img_dict(
    clip_features: torch.Tensor,
    clip_outputs: torch.Tensor,
    classnames: List[str],
    mu_dict: Dict[str, torch.Tensor],
    sigma_img_dict: Dict[str, torch.Tensor],
    *,
    diag_store: bool = False,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """
    将当前测试样本（及其增强）的协方差写入 sigma_img_dict（如果该类尚无值）。
    参数:
      clip_features: [K, D] Tensor，K 个增强的特征（或 [D] 单向量）
      clip_outputs: logits/probs，若为 [K, C] 则先对 K 平均 -> [C]，若为 [C] 则直接使用
      classnames: 类别名列表（索引->类名）
      mu_dict: {classname -> Tensor[D]}，文本类别中心（用于中心化）
      sigma_img_dict: 现有字典（会被修改并返回），value 存为 CPU Tensor
      diag_store: 若 True，只保存对角线（diag 矩阵）
      eps: 数值稳定项
    返回:
      更新后的 sigma_img_dict（value 为 cpu().detach() 的 Tensor）
    """
    # 1) 标准化输入形状、device
    if clip_features.ndim == 1:
        clip_features = clip_features.unsqueeze(0)  # [1, D]
    feats = clip_features  # [K, D]
    device = feats.device
    dtype = feats.dtype

    # 2) 聚合输出得到单一伪标签
    if clip_outputs.ndim == 2:
        agg_logits = clip_outputs.mean(dim=0)
    elif clip_outputs.ndim == 1:
        agg_logits = clip_outputs
    else:
        raise ValueError("clip_outputs must be 1D ([C]) or 2D ([K, C])")

    pseudo_idx = int(agg_logits.argmax().item())
    cls_name = classnames[pseudo_idx]

    # 3) 如果已经存在则跳过
    if cls_name in sigma_img_dict:
        return sigma_img_dict

    # 4) 取 mu（若没有 mu，则用 feats 的均值作为中心）
    if cls_name in mu_dict:
        mu = mu_dict[cls_name].to(device=device, dtype=dtype)
    else:
        # 备选：若没有 mu，使用增强的样本均值
        mu = feats.mean(dim=0)

    # 5) 计算协方差（数值稳定）
    cov = safe_cov_from_features(feats, mu=mu, eps=eps)

    # 6) 如果要求只存对角，则转为对角矩阵
    if diag_store:
        cov = torch.diag(torch.diag(cov))

    # 7) 存为 CPU, detach，避免占用显存；但保持 dtype=float32
    sigma_img_dict[cls_name] = cov.detach().cpu().float()

    # 可选日志
    # print(f"[sigma_img_dict] stored class {cls_name} cov (diag_store={diag_store})")

    return sigma_img_dict

# ---- 用图像协方差对文本协方差做 shrink（支持只 shrink diagonal） ----
def shrink_text_covariance(
    sigma_text_dict: Dict[str, torch.Tensor],
    sigma_img_dict: Dict[str, torch.Tensor],
    *,
    alpha: float = 0.5,
    diag_shrink: bool = False,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """
    对 sigma_text_dict 中的每个类，用 sigma_img_dict 中对应的协方差做收缩：
      sigma_out = (1-alpha) * sigma_text + alpha * sigma_img
    参数:
      sigma_text_dict: {class -> [D,D] Tensor} （通常为 CPU 或 GPU）
      sigma_img_dict: {class -> [D,D] Tensor} （通常为 CPU）
      alpha: 收缩强度
      diag_shrink: 若 True，仅用 sigma_img 的对角线去替换 sigma_text 的对角线（其余非对角保留 sigma_text）
      eps: 数值稳定
    返回:
      shrinked_dict: {class -> [D,D] Tensor} （与 sigma_text_dict 使用相同 device 和 dtype）
    """
    shrinked = {}
    for cls, sigma_text in sigma_text_dict.items():
        # ensure sigma_text is tensor
        if not isinstance(sigma_text, torch.Tensor):
            sigma_text = torch.tensor(sigma_text)

        device = sigma_text.device
        dtype = sigma_text.dtype

        sigma_t = sigma_text.to(device=device, dtype=dtype)

        if cls in sigma_img_dict:
            sigma_i_cpu = sigma_img_dict[cls]  # stored as CPU per update function
            sigma_i = sigma_i_cpu.to(device=device, dtype=dtype)

            if diag_shrink:
                # keep text's off-diagonals, replace diagonals with convex combo
                diag_t = torch.diag(sigma_t)
                diag_i = torch.diag(sigma_i)
                diag_new = (1 - alpha) * diag_t + alpha * diag_i
                sigma_s = sigma_t.clone()
                sigma_s.fill_diagonal_(0.0)
                sigma_s = sigma_s + torch.diag(diag_new)
            else:
                sigma_s = (1 - alpha) * sigma_t + alpha * sigma_i
        else:
            # no image info -> keep original
            sigma_s = sigma_t.clone()

        # guard and regularize
        D = sigma_s.size(0)
        sigma_s = sigma_s + eps * torch.eye(D, device=device, dtype=dtype)
        sigma_s = torch.nan_to_num(sigma_s, nan=0.0, posinf=1e6, neginf=-1e6)

        shrinked[cls] = sigma_s

    return shrinked
def diag_scale_from_pair(Sigma_txt, Sigma_img, eps=1e-12):
    """
    从一个类的 (Sigma_txt, Sigma_img) 学到对角缩放因子
    """
    var_txt = torch.clamp(torch.diag(Sigma_txt), min=eps)  # 文本方差
    var_img = torch.clamp(torch.diag(Sigma_img), min=eps)  # 图像方差
    s = torch.sqrt(var_img / var_txt)                      # 每个维度的缩放系数
    return s  # shape [D]



def scale_sigma_dict(sigma_txt_dict, Sigma_img_ref, mode="diag", eps=1e-12):
    """
    将文本协方差矩阵字典缩放到图像协方差矩阵参考尺度
    参数:
        sigma_txt_dict: dict {classname -> Tensor[D,D]} 文本协方差矩阵
        Sigma_img_ref: Tensor[D,D] 某个类别的图像协方差矩阵，作为缩放参考
        mode: str, "diag" 或 "full"
            - "diag": 只缩放对角线元素
            - "full": 全矩阵缩放
        eps: float, 防止数值不稳定
    返回:
        dict {classname -> Tensor[D,D]} 缩放后的协方差矩阵字典
    """
    D = Sigma_img_ref.size(0)
    Sigma_img_ref = Sigma_img_ref + eps * torch.eye(D, device=Sigma_img_ref.device)

    out = {}
    for cname, Sigma_txt in sigma_txt_dict.items():
        Sigma_txt = Sigma_txt + eps * torch.eye(D, device=Sigma_txt.device)

        if mode == "diag":
            # 对角缩放
            s = torch.sqrt(torch.clamp(torch.diag(Sigma_img_ref), min=eps) / 
                           torch.clamp(torch.diag(Sigma_txt), min=eps))
            S = torch.diag(s)
            Sigma_scaled = S @ Sigma_txt @ S

        elif mode == "full":
            # 全矩阵缩放
            try:
                sqrt_txt = torch.linalg.cholesky(Sigma_txt)
                sqrt_img = torch.linalg.cholesky(Sigma_img_ref)
                A = sqrt_img @ torch.linalg.inv(sqrt_txt)
                Sigma_scaled = A @ Sigma_txt @ A.T
            except RuntimeError:
                # 若奇异或非正定，则退化为对角缩放
                s = torch.sqrt(torch.clamp(torch.diag(Sigma_img_ref), min=eps) / 
                               torch.clamp(torch.diag(Sigma_txt), min=eps))
                S = torch.diag(s)
                Sigma_scaled = S @ Sigma_txt @ S
        else:
            raise ValueError(f"Unknown mode: {mode}")

        out[cname] = Sigma_scaled

    return out


def diag_or_full_scale_dict(sigma_txt_dict, s, mode="full", alpha=2.0, eps=1e-12):
    """
    根据单个类别的 (Sigma_txt, Sigma_img) 学到的缩放系数 s
    对 sigma_txt_dict 中的每个协方差矩阵进行缩放。

    参数:
    - sigma_txt_dict: dict{classname -> covariance matrix}，文本协方差矩阵
    - s: [D] 缩放系数向量
    - mode: "diag" -> 只缩放对角线
            "full" -> 全元素缩放
    - alpha: 非对角线缩放比例（mode="full" 时有效）
             alpha=1.0 -> 全元素完全缩放
             alpha=0.0 -> 非对角线保持不变
    - eps: 防止除零或数值过小

    返回:
    - dict{classname -> scaled covariance matrix}
    """
    s = torch.clamp(s, min=eps)
    out = {}
    S = torch.diag(s)  # 对角矩阵

    for c, Sigma_txt in sigma_txt_dict.items():
        Sigma_txt = Sigma_txt.to(s.device)
        if mode == "diag":
            # 只缩放对角线
            Sigma_scaled = S @ Sigma_txt @ S
            # 保留原非对角线
            Sigma_scaled = torch.diag(torch.diag(Sigma_scaled)) + (Sigma_txt - torch.diag(torch.diag(Sigma_txt)))
        elif mode == "full":
            # 对所有元素缩放
            # 方案 1: 按 s_i * s_j 缩放全矩阵
            Sigma_scaled = Sigma_txt * (s.view(-1, 1) @ s.view(1, -1))
            # 可选 alpha 调整非对角线缩放强度
            off_diag = Sigma_scaled - torch.diag(torch.diag(Sigma_scaled))
            Sigma_scaled = torch.diag(torch.diag(Sigma_scaled)) + alpha * off_diag
        else:
            raise ValueError(f"Unknown mode: {mode}")

        out[c] = Sigma_scaled

    return out


def compare_diag_offdiag(sigma_img, sigma_txt, save_path=None):
    """
    比较图像和文本协方差矩阵的对角线与非对角线元素的数值分布
    - sigma_img: dict {classname -> [D, D] Tensor}
    - sigma_txt: dict {classname -> [D, D] Tensor}
    """
    diag_vals_img, offdiag_vals_img = [], []
    diag_vals_txt, offdiag_vals_txt = [], []

    for cls in sigma_img:
        if cls not in sigma_txt:
            continue
        for sigma, diag_vals, offdiag_vals in [
            (sigma_img[cls], diag_vals_img, offdiag_vals_img),
            (sigma_txt[cls], diag_vals_txt, offdiag_vals_txt),
        ]:
            # 确保在 CPU 上
            sigma = sigma.detach().cpu()
            diag = torch.diag(sigma).numpy()
            offdiag = sigma.numpy()[~np.eye(sigma.shape[0], dtype=bool)]

            diag_vals.extend(diag.tolist())
            offdiag_vals.extend(offdiag.tolist())

    # 转 numpy
    diag_vals_img, offdiag_vals_img = np.array(diag_vals_img), np.array(offdiag_vals_img)
    diag_vals_txt, offdiag_vals_txt = np.array(diag_vals_txt), np.array(offdiag_vals_txt)

    def summarize(name, diag, offdiag):
        print(f"\n{name} 协方差矩阵统计:")
        print(f"  对角线: mean={diag.mean():.4f}, std={diag.std():.4f}, min={diag.min():.4f}, max={diag.max():.4f}")
        print(f"  非对角: mean={offdiag.mean():.4f}, std={offdiag.std():.4f}, min={offdiag.min():.4f}, max={offdiag.max():.4f}")

    summarize("图像", diag_vals_img, offdiag_vals_img)
    summarize("文本", diag_vals_txt, offdiag_vals_txt)

    # 画直方图对比
    plt.figure(figsize=(10, 6))
    plt.hist(diag_vals_img, bins=50, alpha=0.5, label="Image Diagonal", color="tab:blue")
    plt.hist(offdiag_vals_img, bins=50, alpha=0.5, label="Image Off-diagonal", color="tab:orange")
    plt.hist(diag_vals_txt, bins=50, alpha=0.5, label="Text Diagonal", color="tab:green")
    plt.hist(offdiag_vals_txt, bins=50, alpha=0.5, label="Text Off-diagonal", color="tab:red")

    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Diagonal vs Off-diagonal Elements of Covariance Matrices")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    
    
EPS = 1e-12

def to_numpy(x):
    """支持 numpy.ndarray 或 torch.Tensor，统一返回 np.float64"""
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    if isinstance(x, np.ndarray):
        return x.astype(np.float64)
    raise TypeError(f"Unsupported type {type(x)}. Expect numpy.ndarray or torch.Tensor.")

def symmetrize(S):
    S = (S + S.T) / 2.0
    # 清理极小的非对称数值噪声
    return S

def top_offdiag_pairs(S, k=10):
    D = S.shape[0]
    off = np.abs(S.copy())
    # 只看上三角（不含对角）
    iu = np.triu_indices(D, k=1)
    vals = off[iu]
    if vals.size == 0:
        return []
    order = np.argsort(vals)[::-1][:min(k, vals.size)]
    pairs = [(int(iu[0][o]), int(iu[1][o]), float(vals[o])) for o in order]
    return pairs

def compute_r_for_sigma(sigma, mean=None, assume_second_moment=False,
                        symmetrize_matrix=True, compute_eigs=True):
    """
    计算 r 及诊断信息。
    - sigma: 协方差或二阶矩矩阵（numpy 或 torch）
    - mean: 可选，若 sigma 是 E[x x^T]（二阶矩），则可提供 mean (D,) 用于中心化： cov = sigma - outer(mean, mean)
    - assume_second_moment: 若 True 且 mean provided，会进行中心化。否则不中心化。
    返回 dict，包含 r, off_norm, total_norm, diag_norm, eigvals(若计算成功), top_offdiag 等。
    """
    S = to_numpy(sigma)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("sigma must be square 2D matrix")
    if symmetrize_matrix:
        S = symmetrize(S)
    info = {}
    info['orig_total_norm'] = float(np.linalg.norm(S, ord='fro'))
    # 如果用户指定 sigma 为二阶矩并给出均值，则中心化
    if assume_second_moment and (mean is not None):
        mu = to_numpy(mean).ravel()
        if mu.size != S.shape[0]:
            raise ValueError("mean length mismatch with sigma dim")
        S_centered = S - np.outer(mu, mu)
        S_centered = symmetrize(S_centered)
        info['centered_total_norm'] = float(np.linalg.norm(S_centered, ord='fro'))
        info['used_centering'] = True
    else:
        S_centered = S
        info['used_centering'] = False

    total_norm = np.linalg.norm(S_centered, ord='fro')
    diag = np.diag(np.diag(S_centered))
    diag_norm = np.linalg.norm(diag, ord='fro')
    off = S_centered - diag
    off_norm = np.linalg.norm(off, ord='fro')

    info.update({
        'total_norm': float(total_norm),
        'diag_norm': float(diag_norm),
        'off_norm': float(off_norm),
        # 主比率（off/total）
        'r_off_over_total': float(off_norm / (total_norm + EPS)),
        # 备用比率（off/diag），当 diag 很小时，这个会更敏感
        'r_off_over_diag': float(off_norm / (diag_norm + EPS)),
    })

    # eigenvalues 做为额外诊断（可能会耗时）
    if compute_eigs:
        try:
            eigs = np.linalg.eigvalsh(S_centered)
            info['eigvals_min'] = float(np.min(eigs))
            info['eigvals_max'] = float(np.max(eigs))
            # 前几个主成分解释率（从大到小）
            eigs_sorted = np.sort(eigs)[::-1]
            total_ev = eigs_sorted.sum() if eigs_sorted.sum() != 0 else EPS
            cum = np.cumsum(eigs_sorted) / total_ev
            info['eig_cum_1'] = float(cum[0]) if cum.size>0 else 0.0
            info['eig_cum_5'] = float(cum[4]) if cum.size>4 else float(cum[-1]) 
        except Exception as e:
            info['eig_error'] = str(e)

    info['top_offdiag'] = top_offdiag_pairs(S_centered, k=10)
    return info

def compute_r_all(sigma_dict, means_dict=None, assume_second_moment=False, **kwargs):
    """
    sigma_dict: {classname -> matrix}
    means_dict: optional {classname -> mean vector} if assume_second_moment True
    assume_second_moment: 是否把每个 sigma 当作 E[x x^T]（而非中心化的协方差）
    kwargs 传给 compute_r_for_sigma
    返回: {classname -> info dict}
    """
    results = {}
    for cname, mat in sigma_dict.items():
        mean = None
        if assume_second_moment and (means_dict is not None):
            mean = means_dict.get(cname, None)
        try:
            results[cname] = compute_r_for_sigma(mat, mean=mean, assume_second_moment=assume_second_moment, **kwargs)
        except Exception as e:
            results[cname] = {'error': str(e)}
    return results






def cov_diag_offdiag_stats(sigma_dict):
    """
    对每个类别协方差矩阵，计算：
        - 对角线元素的均值和方差
        - 非对角线元素的均值和方差
    sigma_dict: dict
        key: 类别名称
        value: 协方差矩阵 (np.ndarray 或 torch.Tensor)
    返回：
        stats_dict: dict
            key: 类别名称
            value: dict with keys:
                'diag_mean', 'diag_std', 'off_mean', 'off_std'
    """
    stats_dict = {}
    
    for cls_name, Sigma in sigma_dict.items():
        if isinstance(Sigma, torch.Tensor):
            Sigma_np = Sigma.cpu().numpy()
        else:
            Sigma_np = np.asarray(Sigma)

        D = Sigma_np.shape[0]
        diag = np.diag(Sigma_np)
        offdiag = Sigma_np[~np.eye(D, dtype=bool)]  # 非对角元素
        
        stats_dict[cls_name] = {
            'diag_mean': np.mean(diag),
            'diag_std': np.std(diag),
            'off_mean': np.mean(offdiag),
            'off_std': np.std(offdiag)
        }
    
    return stats_dict

def inverse_error_batch(sigma_dict, eps=1e-6):
    """
    计算每个类别协方差矩阵用对角矩阵替代后的逆矩阵误差
    sigma_dict: dict
        key: 类别名称
        value: 协方差矩阵 (np.ndarray 或 torch.Tensor)
    eps: 数值稳定性项，如果协方差矩阵接近奇异
    返回:
        error_dict: 每个类别的相对 Frobenius 误差
        mean_error: 平均误差
        std_error: 误差标准差
    """
    error_dict = {}
    errors = []

    for cls_name, Sigma in sigma_dict.items():
        # 转为 numpy
        if isinstance(Sigma, torch.Tensor):
            Sigma_np = Sigma.cpu().numpy()
        else:
            Sigma_np = np.asarray(Sigma)

        D = Sigma_np.shape[0]
        # 对称化
        Sigma_np = (Sigma_np + Sigma_np.T) / 2

        # 对角矩阵
        diag_Sigma = np.diag(np.diag(Sigma_np))

        # 增加小正则项防止奇异
        Sigma_inv = np.linalg.inv(Sigma_np + eps*np.eye(D))
        diag_inv = np.linalg.inv(diag_Sigma + eps*np.eye(D))

        # 相对 Frobenius 误差
        err = np.linalg.norm(Sigma_inv - diag_inv, ord='fro') / np.linalg.norm(Sigma_inv, ord='fro')
        error_dict[cls_name] = err
        errors.append(err)

    mean_error = np.mean(errors)
    std_error = np.std(errors)
    return error_dict, mean_error, std_error

def scale_sigma_dict_txt2imgori(sigma_txt_dict, Sigma_img, mode="diag", eps=1e-12, alpha=0.9):
    """
    根据单类别图像协方差矩阵对文本协方差字典进行缩放/收缩。
    保持接口与 scale_sigma_dict 相同。

    参数:
    - sigma_txt_dict: {classname -> [D,D] Tensor} 文本协方差字典
    - Sigma_img: [D,D] Tensor，单类别图像协方差
    - mode: "full" 全协方差缩放, "diag" 仅对角线缩放
    - eps: 防止数值不稳定
    - alpha: 收缩强度，0 表示只用缩放，1 表示只用图像协方差

    返回:
    - sigma_txt_adjusted: {classname -> [D,D] Tensor} 调整后的文本协方差字典
    """
    device = Sigma_img.device
    D = Sigma_img.size(0)
    
    # 构造缩放矩阵 S
    sigma_txt_sample = next(iter(sigma_txt_dict.values())).to(device)
    if mode == "diag":
        var_txt = torch.clamp(torch.diag(sigma_txt_sample), min=eps)
        var_img = torch.clamp(torch.diag(Sigma_img), min=eps)
        s = torch.sqrt(var_img / var_txt)
        S = torch.diag(s)
    else:  # full
        # 使用 Cholesky 分解求整体缩放矩阵
        # S * Sigma_txt_sample * S ≈ Sigma_img -> S ≈ sqrtm(Sigma_img @ inv(Sigma_txt_sample))
        try:
            inv_txt = torch.linalg.inv(sigma_txt_sample + eps * torch.eye(D, device=device))
            sqrt_matrix = torch.linalg.cholesky(Sigma_img @ inv_txt + eps * torch.eye(D, device=device))
            S = sqrt_matrix
        except RuntimeError:
            # 回退到对角线缩放
            var_txt = torch.clamp(torch.diag(sigma_txt_sample), min=eps)
            var_img = torch.clamp(torch.diag(Sigma_img), min=eps)
            s = torch.sqrt(var_img / var_txt)
            S = torch.diag(s)

    # 应用缩放 + alpha 收缩
    sigma_txt_adjusted = {}
    for c, St in sigma_txt_dict.items():
        St = St.to(device)
        scaled = S @ St @ S
        sigma_txt_adjusted[c] = (1 - alpha) * scaled + alpha * Sigma_img

    return sigma_txt_adjusted



def scale_sigma_dict_txt2img_congruence4(
    sigma_txt_dict,      # {classname -> [D,D] Tensor} 文本协方差字典
    Sigma_img,           # [D,D] Tensor，参考类别图像协方差
    ref_class=None,      # str，可选，指定参考类别文本协方差
    mode="diag",         # 目前仅支持 "diag"
    beta=0.0,            # 类间平滑权重
    eps=1e-12,
    clip_ratio=10.0,     # 方差比率裁剪阈值
    weak_dim_ratio=0.1,  # 对极端维度比例 top-k 弱化
    weak_slope=0.2,      # 平滑函数斜率
    pca_ratio=0.9        # 保留 PCA 主成分比例
):
    """
    升级版：文本协方差 → 图像域（对角线映射 + alpha + r_i + PCA低秩 + 类归一化 + 平滑极端维度）
    """
    device = Sigma_img.device
    D = Sigma_img.size(0)

    if ref_class is None:
        ref_class = next(iter(sigma_txt_dict))
    Sigma_txt_ref = sigma_txt_dict[ref_class].to(device)

    if mode != "diag":
        raise NotImplementedError("仅支持 diag")

    # 1. alpha 全局缩放
    diag_txt_ref = torch.clamp(torch.diag(Sigma_txt_ref), min=eps)
    diag_img_ref = torch.clamp(torch.diag(Sigma_img), min=eps)
    alpha = diag_img_ref.mean() / diag_txt_ref.mean()

    # 2. 文本域相对缩放 r_i
    r_dict = {}
    for c, Sigma_txt in sigma_txt_dict.items():
        diag_txt = torch.clamp(torch.diag(Sigma_txt.to(device)), min=eps)
        r_dict[c] = diag_txt / diag_txt_ref

    # 3. 初步映射
    diag_mapped_dict = {}
    for c in sigma_txt_dict:
        diag_mapped = alpha * r_dict[c] * diag_img_ref
        diag_mapped_dict[c] = diag_mapped

    # 4. PCA 低秩处理（保留主要方向）
    all_mapped = torch.stack(list(diag_mapped_dict.values()), dim=0)  # [C, D]
    cov_all = torch.cov(all_mapped.T) + eps * torch.eye(D, device=device)
    eigvals, eigvecs = torch.linalg.eigh(cov_all)
    # 保留主成分
    eigvals_sorted, idx = torch.sort(eigvals, descending=True)
    eigvecs_sorted = eigvecs[:, idx]
    cum_var = torch.cumsum(eigvals_sorted, dim=0) / torch.sum(eigvals_sorted)
    k = max(1, torch.sum(cum_var <= pca_ratio).item())
    U = eigvecs_sorted[:, :k]  # [D, k]
    # 投影到主成分，再还原
    for c in diag_mapped_dict:
        diag_vec = diag_mapped_dict[c]
        diag_vec = U @ (U.T @ diag_vec)
        diag_mapped_dict[c] = diag_vec

    # 5. 方差比率裁剪 + 平滑极端维度
    all_ratios = torch.stack([diag_mapped_dict[c] / diag_img_ref for c in diag_mapped_dict], dim=0)
    mean_ratio = all_ratios.mean(dim=0)
    k_extreme = max(1, int(D * weak_dim_ratio))
    extreme_dims = torch.topk(torch.abs(torch.log(mean_ratio + eps)), k=k_extreme).indices

    for c in diag_mapped_dict:
        diag_vec = diag_mapped_dict[c]
        # 裁剪
        ratio = diag_vec / diag_img_ref
        ratio = torch.clamp(ratio, min=1.0/clip_ratio, max=clip_ratio)
        diag_vec = diag_img_ref * ratio
        # 平滑弱化极端维度
        diag_vec[extreme_dims] = diag_vec[extreme_dims] / (1.0 + weak_slope * torch.abs(torch.log(ratio[extreme_dims] + eps)))
        # 类归一化
        diag_vec = diag_vec / diag_vec.mean() * diag_img_ref.mean()
        diag_mapped_dict[c] = diag_vec

    # 6. 构造对角矩阵输出
    sigma_txt_adjusted = {c: torch.diag(diag_mapped_dict[c]) for c in diag_mapped_dict}
    return sigma_txt_adjusted






def analyze_problematic_dims(sigma_img, sigma_txt, mu_dict, topk=10):
    """
    分析哪些维度可能导致协方差相关性高但马氏距离相关性低。
    
    参数:
    - sigma_img: dict {class_name -> Tensor[D,D]} 图像协方差
    - sigma_txt: dict {class_name -> Tensor[D,D]} 文本协方差
    - mu_dict: dict {class_name -> Tensor[D]} 共享的类别均值
    - topk: 输出差异最大的维度数目
    """
    # 聚合所有类别的对角线方差
    var_ratios = []
    for cls in sigma_img:
        if cls not in sigma_txt: 
            continue
        diag_img = torch.diag(sigma_img[cls]).cpu().numpy()
        diag_txt = torch.diag(sigma_txt[cls]).cpu().numpy()
        ratio = diag_img / (diag_txt + 1e-12)
        var_ratios.append(ratio)
    
    var_ratios = np.stack(var_ratios, axis=0)  # [C, D]
    mean_ratio = var_ratios.mean(axis=0)       # [D]
    
    # 找出差异最大的维度
    diff_from1 = np.abs(np.log(mean_ratio))  # log-scale 稳定
    worst_dims = np.argsort(-diff_from1)[:topk]

    print(f"差异最大的 {topk} 个维度:")
    for i, dim in enumerate(worst_dims):
        print(f"  维度 {dim}: 平均 img/txt 比率={mean_ratio[dim]:.4f}")

    # ========== 测试这些维度的马氏距离相关性 ==========
    d_img_list, d_txt_list = [], []
    classes = list(mu_dict.keys())

    for i, cls_i in enumerate(classes):
        if cls_i not in sigma_img or cls_i not in sigma_txt: 
            continue
        mu_i = mu_dict[cls_i].cpu().numpy()
        var_img_i = torch.diag(sigma_img[cls_i]).cpu().numpy()
        var_txt_i = torch.diag(sigma_txt[cls_i]).cpu().numpy()

        for j, cls_j in enumerate(classes):
            if j <= i: 
                continue
            if cls_j not in sigma_img or cls_j not in sigma_txt:
                continue

            mu_j = mu_dict[cls_j].cpu().numpy()
            diff = mu_i - mu_j

            # 仅在 worst_dims 上计算
            d_img = np.sum((diff[worst_dims] ** 2) / (var_img_i[worst_dims] + 1e-12))
            d_txt = np.sum((diff[worst_dims] ** 2) / (var_txt_i[worst_dims] + 1e-12))

            d_img_list.append(d_img)
            d_txt_list.append(d_txt)

    corr = np.corrcoef(d_img_list, d_txt_list)[0, 1]
    print(f"\n仅在差异最大 {topk} 个维度上的马氏距离相关性: {corr:.4f}")
    return worst_dims, mean_ratio[worst_dims]
from typing import Dict, Tuple




def mahalanobis_diag_corr_scaled_robust(mu_dict, sigma_img, sigma_txt, 
                                     eps=1e-12, min_var=1e-5):
    """
    稳健版对角线马氏距离相关性计算
    
    特点：
    - 对低方差维度裁剪，避免噪声主导马氏距离
    - 保留全局缩放
    - 高维低样本下更稳健

    参数:
    - mu_dict: dict {class_name -> Tensor[D]} 类均值特征
    - sigma_img: dict {class_name -> Tensor[D,D]} 图像协方差
    - sigma_txt: dict {class_name -> Tensor[D,D]} 文本协方差
    - eps: float, 数值稳定性
    - min_var: float, 最小方差阈值

    返回:
    - corr: float, 图像域与文本域马氏距离的相关系数
    - stats: dict, 辅助信息
    """
    classes = list(mu_dict.keys())
    n = len(classes)

    # 提取对角线方差
    diag_img = {c: np.clip(np.diag(sigma_img[c].cpu().numpy()), a_min=min_var, a_max=None) for c in classes}
    diag_txt = {c: np.clip(np.diag(sigma_txt[c].cpu().numpy()), a_min=min_var, a_max=None) for c in classes}

    # 全局缩放
    mean_img = np.mean([v.mean() for v in diag_img.values()])
    mean_txt = np.mean([v.mean() for v in diag_txt.values()])
    global_scale = mean_img / (mean_txt + eps)

    # 计算类别间马氏距离
    d_img, d_txt = [], []
    for i in range(n):
        for j in range(i+1, n):
            ci, cj = classes[i], classes[j]
            diff = (mu_dict[ci] - mu_dict[cj]).detach().cpu().numpy()

            d_img_ij = np.sum(diff**2 / (diag_img[ci] + eps))
            d_txt_ij = np.sum(diff**2 / (diag_txt[ci] + eps))

            d_img.append(d_img_ij)
            d_txt.append(d_txt_ij)

    d_img = np.array(d_img)
    d_txt = np.array(d_txt)

    # 相关性
    if len(d_img) > 1:
        corr = np.corrcoef(d_img, d_txt)[0,1]
    else:
        corr = float('nan')

    stats = {
        "global_scale": global_scale,
        "mean_img_var": mean_img,
        "mean_txt_var": mean_txt,
        "n_pairs": len(d_img)
    }

    return corr, stats

def mahalanobis_diag_corr_pca(mu_dict, sigma_img, sigma_txt, 
                              target_dim=46, min_var=1e-5, eps=1e-12):
    """
    对角线马氏距离相关性计算（PCA 版，高维低样本稳健）
    
    改进点：
    - 先剔除低方差维度（min_var）
    - 对图像和文本都做 PCA，固定目标维度 target_dim
    - 保持图像和文本维度一致
    - 马氏距离按 PCA 后的方差计算

    参数：
    - mu_dict: dict {class_name -> Tensor[D]} 类均值特征
    - sigma_img: dict {class_name -> Tensor[D,D]} 图像协方差
    - sigma_txt: dict {class_name -> Tensor[D,D]} 文本协方差
    - target_dim: int, PCA 目标维度
    - min_var: float, 剔除方差小于该值的维度
    - eps: float, 数值稳定性

    返回：
    - corr: float, 图像域与文本域马氏距离的相关系数
    - stats: dict, 辅助信息
    """
    import numpy as np
    from sklearn.decomposition import PCA

    classes = list(mu_dict.keys())
    n = len(classes)

    # 提取对角线方差，并剔除低方差维度
    diag_img_all = np.stack([np.diag(sigma_img[c].cpu().numpy()) for c in classes], axis=0)
    diag_txt_all = np.stack([np.diag(sigma_txt[c].cpu().numpy()) for c in classes], axis=0)

    mask_img = diag_img_all.mean(axis=0) >= min_var
    mask_txt = diag_txt_all.mean(axis=0) >= min_var
    mask_keep = mask_img & mask_txt  # 两个域都满足

    diag_img_all = diag_img_all[:, mask_keep]
    diag_txt_all = diag_txt_all[:, mask_keep]

    # 均值特征矩阵
    mu_all = np.stack([mu_dict[c].detach().cpu().numpy()[mask_keep] for c in classes], axis=0)

    # PCA
    k = min(target_dim, mu_all.shape[1])
    pca_img = PCA(n_components=k)
    mu_img_pca = pca_img.fit_transform(mu_all)
    var_img_pca = pca_img.explained_variance_

    pca_txt = PCA(n_components=k)
    mu_txt_pca = pca_txt.fit_transform(mu_all)
    var_txt_pca = pca_txt.explained_variance_

    # 计算类别间马氏距离
    d_img, d_txt = [], []
    for i in range(n):
        for j in range(i+1, n):
            diff_img = mu_img_pca[i] - mu_img_pca[j]
            diff_txt = mu_txt_pca[i] - mu_txt_pca[j]

            d_img_ij = np.sum(diff_img**2 / (var_img_pca + eps))
            d_txt_ij = np.sum(diff_txt**2 / (var_txt_pca + eps))

            d_img.append(d_img_ij)
            d_txt.append(d_txt_ij)

    d_img = np.array(d_img)
    d_txt = np.array(d_txt)

    # 相关性
    if len(d_img) > 1:
        corr = np.corrcoef(d_img, d_txt)[0,1]
    else:
        corr = float('nan')

    stats = {
        "mean_img_var": np.mean(var_img_pca),
        "mean_txt_var": np.mean(var_txt_pca),
        "n_pairs": len(d_img),
        "pca_components_img": len(var_img_pca),
        "pca_components_txt": len(var_txt_pca)
    }

    return corr, stats

def compare_mahalanobis_pca(mu_dict, sigma_txt, target_dim=None, eps=1e-12):
    """
    比较文本协方差矩阵的原始马氏距离和 PCA 降维后的马氏距离
    
    参数：
    - mu_dict: dict {class_name -> Tensor[D]} 各类别均值特征
    - sigma_txt: dict {class_name -> Tensor[D,D]} 文本协方差矩阵
    - target_dim: int, PCA 降维目标维度（<= n_samples - 1）
    - eps: float, 数值稳定性
    
    返回：
    - corr: float, 原始马氏距离与 PCA 马氏距离的相关系数
    - mean_rel_diff: float, 平均相对误差
    """
    classes = list(mu_dict.keys())
    n = len(classes)
    
    # Step1: 原始对角线马氏距离
    diag_txt = {c: np.clip(np.diag(sigma_txt[c].cpu().numpy()), a_min=eps, a_max=None) for c in classes}
    d_orig = []
    for i in range(n):
        for j in range(i+1, n):
            ci, cj = classes[i], classes[j]
            diff = (mu_dict[ci] - mu_dict[cj]).detach().cpu().numpy()
            d_ij = np.sum(diff**2 / (diag_txt[ci] + eps))
            d_orig.append(d_ij)
    d_orig = np.array(d_orig)
    
    # Step2: 构建 PCA 投影
    D = next(iter(mu_dict.values())).size(0)
    X = np.stack([mu_dict[c].detach().cpu().numpy() for c in classes], axis=0)  # [n_classes, D]
    
    if target_dim is None or target_dim > min(X.shape[0]-1, D):
        target_dim = min(X.shape[0]-1, D)
    
    pca = PCA(n_components=target_dim)
    X_pca = pca.fit_transform(X)  # [n_classes, target_dim]
    
    # 对应协方差矩阵只取投影后的对角线
    sigma_txt_pca = {}
    for idx, c in enumerate(classes):
        # 将协方差矩阵投影到 PCA 空间：C' = P^T C P
        C = sigma_txt[c].cpu().numpy()
        P = pca.components_.T  # [D, target_dim]
        C_pca = P.T @ C @ P
        sigma_txt_pca[c] = np.clip(np.diag(C_pca), a_min=eps, a_max=None)
    
    # Step3: PCA 后马氏距离
    d_pca = []
    for i in range(n):
        for j in range(i+1, n):
            ci, cj = classes[i], classes[j]
            diff = X_pca[i] - X_pca[j]
            d_ij = np.sum(diff**2 / (sigma_txt_pca[ci] + eps))
            d_pca.append(d_ij)
    d_pca = np.array(d_pca)
    
    # Step4: 相关性与平均相对误差
    corr = np.corrcoef(d_orig, d_pca)[0,1]
    mean_rel_diff = np.mean(np.abs(d_orig - d_pca) / (d_orig + eps))
    
    return corr, mean_rel_diff


def mahalanobis_diag_corr_variance_ratio_both(mu_dict, sigma_img, sigma_txt, 
                                             target_ratio=0.98, eps=1e-12, min_var=1e-12):
    """
    对角线马氏距离相关性计算（图像+文本协方差均基于方差贡献比选择维度）

    - 图像与文本协方差矩阵分别选择方差累积贡献达到 target_ratio 的维度
    - 各自在子空间里计算马氏距离
    - 最终比较相关性

    参数:
    - mu_dict: dict {class_name -> Tensor[D]} 类均值特征
    - sigma_img: dict {class_name -> Tensor[D,D]} 图像协方差
    - sigma_txt: dict {class_name -> Tensor[D,D]} 文本协方差
    - target_ratio: float, 保留方差累积比例 (e.g., 0.9 表示保留90%信息)
    - eps: float, 数值稳定性
    - min_var: float, 方差下限

    返回:
    - corr: float, 图像域与文本域马氏距离的相关系数
    - stats: dict, 辅助信息（包括每个类别保留维度数量等）
    """
    classes = list(mu_dict.keys())
    n = len(classes)

    diag_img = {c: np.clip(np.diag(sigma_img[c].cpu().numpy()), a_min=min_var, a_max=None) for c in classes}
    diag_txt = {c: np.clip(np.diag(sigma_txt[c].cpu().numpy()), a_min=min_var, a_max=None) for c in classes}

    selected_dims_img, selected_dims_txt = {}, {}

    # === Step 1: 按方差贡献比选择维度 ===
    for c in classes:
        # 图像域
        sorted_idx_img = np.argsort(-diag_img[c])
        cum_var_img = np.cumsum(diag_img[c][sorted_idx_img])
        total_var_img = cum_var_img[-1]
        n_keep_img = np.searchsorted(cum_var_img / (total_var_img + eps), target_ratio) + 1
        selected_dims_img[c] = sorted_idx_img[:n_keep_img]

        # 文本域
        sorted_idx_txt = np.argsort(-diag_txt[c])
        cum_var_txt = np.cumsum(diag_txt[c][sorted_idx_txt])
        total_var_txt = cum_var_txt[-1]
        n_keep_txt = np.searchsorted(cum_var_txt / (total_var_txt + eps), target_ratio) + 1
        selected_dims_txt[c] = sorted_idx_txt[:n_keep_txt]

    # === Step 2: 计算类别间马氏距离 ===
    d_img, d_txt = [], []
    for i in range(n):
        for j in range(i+1, n):
            ci, cj = classes[i], classes[j]
            diff = (mu_dict[ci] - mu_dict[cj]).detach().cpu().numpy()

            idx_i_img = selected_dims_img[ci]
            idx_i_txt = selected_dims_txt[ci]

            d_img_ij = np.sum((diff[idx_i_img] ** 2) / (diag_img[ci][idx_i_img] + eps))
            d_txt_ij = np.sum((diff[idx_i_txt] ** 2) / (diag_txt[ci][idx_i_txt] + eps))

            d_img.append(d_img_ij)
            d_txt.append(d_txt_ij)

    d_img = np.array(d_img)
    d_txt = np.array(d_txt)

    # === Step 3: 计算相关性 ===
    if len(d_img) > 1:
        corr = np.corrcoef(d_img, d_txt)[0, 1]
    else:
        corr = float('nan')

    # === Step 4: 统计保留维度信息 ===
    num_dims_img = {c: len(selected_dims_img[c]) for c in classes}
    num_dims_txt = {c: len(selected_dims_txt[c]) for c in classes}

    stats = {
        "mean_img_var": np.mean([v.mean() for v in diag_img.values()]),
        "mean_txt_var": np.mean([v.mean() for v in diag_txt.values()]),
        "n_pairs": len(d_img),
        "num_dims_img": num_dims_img,
        "num_dims_txt": num_dims_txt,
        "mean_retained_dims_img": np.mean(list(num_dims_img.values())),
        "mean_retained_dims_txt": np.mean(list(num_dims_txt.values())),
        "max_retained_dims_img": np.max(list(num_dims_img.values())),
        "max_retained_dims_txt": np.max(list(num_dims_txt.values())),
        "min_retained_dims_img": np.min(list(num_dims_img.values())),
        "min_retained_dims_txt": np.min(list(num_dims_txt.values())),
    }

    return corr, stats


def compare_full_vs_reduced_mahalanobis(mu_dict, sigma_img, target_ratio=0.98, eps=1e-12, min_var=1e-12):
    """
    比较完整图像协方差与 PCA 裁剪后图像协方差下马氏距离的相关性

    参数:
    - mu_dict: dict {class_name -> Tensor[D]} 类均值特征
    - sigma_img: dict {class_name -> Tensor[D,D]} 图像协方差
    - target_ratio: float, 保留方差累积比例 (e.g., 0.9 表示保留90%信息)
    - eps: float, 数值稳定性
    - min_var: float, 方差下限

    返回:
    - corr: float, 全部维度 vs 裁剪维度下马氏距离的相关性
    - stats: dict, 辅助信息（维度数量等）
    """
    classes = list(mu_dict.keys())
    n = len(classes)

    # 提取完整对角线方差
    diag_img = {c: np.clip(np.diag(sigma_img[c].cpu().numpy()), a_min=min_var, a_max=None) for c in classes}

    selected_dims_img = {}

    # Step 1: 选择方差贡献比达到 target_ratio 的维度
    for c in classes:
        sorted_idx_img = np.argsort(-diag_img[c])
        cum_var_img = np.cumsum(diag_img[c][sorted_idx_img])
        total_var_img = cum_var_img[-1]
        n_keep_img = np.searchsorted(cum_var_img / (total_var_img + eps), target_ratio) + 1
        selected_dims_img[c] = sorted_idx_img[:n_keep_img]

    # Step 2: 计算类别间马氏距离
    d_full, d_reduced = [], []
    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = classes[i], classes[j]
            diff = (mu_dict[ci] - mu_dict[cj]).detach().cpu().numpy()

            # 全部维度
            d_full_ij = np.sum(diff**2 / (diag_img[ci] + eps))

            # 裁剪维度
            idx_i_img = selected_dims_img[ci]
            d_reduced_ij = np.sum((diff[idx_i_img] ** 2) / (diag_img[ci][idx_i_img] + eps))

            d_full.append(d_full_ij)
            d_reduced.append(d_reduced_ij)

    d_full = np.array(d_full)
    d_reduced = np.array(d_reduced)

    # Step 3: 相关性
    if len(d_full) > 1:
        corr = np.corrcoef(d_full, d_reduced)[0, 1]
    else:
        corr = float('nan')

    # Step 4: 输出统计信息
    num_dims_img = {c: len(selected_dims_img[c]) for c in classes}

    stats = {
        "n_pairs": len(d_full),
        "mean_retained_dims_img": np.mean(list(num_dims_img.values())),
        "min_retained_dims_img": np.min(list(num_dims_img.values())),
        "max_retained_dims_img": np.max(list(num_dims_img.values())),
    }

    return corr, stats


def compute_diag_corr(a, b, eps=1e-12):
    """
    计算两个对角线向量的 Pearson 相关系数
    """
    a = a - a.mean()
    b = b - b.mean()
    corr = (a @ b) / (torch.norm(a) * torch.norm(b) + eps)
    return corr.item()

def analyze_sigma_corr(sigma_txt_dict, sigma_img_dict, sigma_img_dict2, scale_fn, save_path=None):
    """
    计算每个类别使用 scale_fn 转换后的文本协方差对角线
    与真实图像协方差对角线的相关性，并绘制频率直方图（频率形式）
    """
    corr_list = []

    for cls_ref in sigma_img_dict2:
        # 使用当前类别参考图像协方差转换
        sigma_txt_adjusted = scale_fn(
            sigma_txt_dict=sigma_txt_dict,
            Sigma_img=sigma_img_dict2[cls_ref],
            ref_class=cls_ref
        )

        # 对每个类别计算对角线相关性
        for cls in sigma_txt_dict:
            diag_txt = torch.diag(sigma_txt_adjusted[cls]).cpu().numpy()
            diag_img = torch.diag(sigma_img_dict[cls]).cpu().numpy()
            corr = np.corrcoef(diag_txt, diag_img)[0, 1]
            corr_list.append(corr)

    corr_array = np.array(corr_list)
    mean_corr = np.mean(corr_array)

    # 绘制直方图（频率而非密度）
    plt.figure(figsize=(7, 5))
    plt.hist(
        corr_array,
        bins=30,
        weights=np.ones_like(corr_array) / len(corr_array),  # 频率
        alpha=0.75,
        color="#4C72B0",
        edgecolor="white",
        linewidth=1.2
    )

    # 均值线
    plt.axvline(mean_corr, color="#DD8452", linestyle="--", linewidth=2, label=f"Mean = {mean_corr:.3f}")

    # 标签和样式
    plt.xlabel("Diagonal Correlation", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Histogram of Diagonal Correlations (text → image)", fontsize=14, weight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return corr_array


# 计算对角线马氏距离相关性
def mahalanobis_diag_corr(mu_dict, sigma_img, sigma_txt, eps=1e-12):
    classes = list(mu_dict.keys())
    n = len(classes)

    diag_img = {c: torch.diag(sigma_img[c]).cpu().numpy() for c in classes}
    diag_txt = {c: torch.diag(sigma_txt[c]).cpu().numpy() for c in classes}

    d_img, d_txt = [], []
    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = classes[i], classes[j]
            diff = (mu_dict[ci] - mu_dict[cj]).detach().cpu().numpy()

            d_img_ij = np.sum(diff**2 / (diag_img[ci] + eps))
            d_txt_ij = np.sum(diff**2 / (diag_txt[ci] + eps))

            d_img.append(d_img_ij)
            d_txt.append(d_txt_ij)

    d_img = np.array(d_img)
    d_txt = np.array(d_txt)
    corr = np.corrcoef(d_img, d_txt)[0, 1] if len(d_img) > 1 else float("nan")
    return corr


# 主流程：遍历参考类
def run_and_plot(mu_dict, sigma_img, sigma_txt, sigma_img_dict, save_path="corr_hist.png"):
    correlations = []
    for ref_class in sigma_img_dict.keys():
        sigma_txt_adjusted = scale_sigma_dict_txt2img_congruence4_noclip(
            sigma_txt, Sigma_img=sigma_img_dict[ref_class], ref_class=ref_class
        )
        corr = mahalanobis_diag_corr_variance_ratio_both(mu_dict, sigma_img, sigma_txt_adjusted)[0]
        correlations.append(corr)

    correlations = np.array(correlations)
    mean_corr = correlations.mean()

    # 绘制直方图（频率）
    plt.figure(figsize=(7, 5))
    plt.hist(
        correlations,
        bins=30,
        weights=np.ones_like(correlations) / len(correlations),  # 频率
        alpha=0.75,
        color="#4C72B0",  # 柔和蓝色
        edgecolor="white",
        linewidth=1.2
    )

    # 均值线
    plt.axvline(mean_corr, color="#DD8452", linestyle="--", linewidth=2, label=f"Mean = {mean_corr:.3f}")

    # 标签和样式
    plt.xlabel("Correlation", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Histogram of Correlations (txt→img adjusted)", fontsize=14, weight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    return correlations
def to_numpy(x):
    if isinstance(x, list):
        x = np.stack([np.array(xx) for xx in x], axis=0)
    elif isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    elif isinstance(x, np.ndarray):
        pass
    else:
        x = np.array(x)
    return x

def softmax1(logits):
    # logits: [N,C]
    a = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=1, keepdims=True)

def accuracy_from_probs(probs, labels):
    preds = probs.argmax(axis=1)
    return (preds == labels).mean(), preds

def compute_ece1(probs, labels, n_bins=15):
    # probs: [N, C], labels: [N]
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (confs > lo) & (confs <= hi) if i>0 else (confs >= lo) & (confs <= hi)
        if mask.sum() == 0:
            bin_accs.append(0.0); bin_confs.append(0.0); bin_counts.append(0)
            continue
        acc = correct[mask].mean()
        conf = confs[mask].mean()
        ece += (mask.sum() / probs.shape[0]) * abs(conf - acc)
        bin_accs.append(acc); bin_confs.append(conf); bin_counts.append(mask.sum())
    return ece, bin_accs, bin_confs, bin_counts
def load_imagenet_folder2name(path):
    dict_imagenet_folder2name = {}
    with open(path) as f:
        line = f.readline()
        while line:
            split_name = line.strip().split()
            cat_name = split_name[2]
            id = split_name[0]
            dict_imagenet_folder2name[id] = cat_name
            line = f.readline()
    # print(dict_imagenet_folder2name)
    return dict_imagenet_folder2name

def refine_classname(class_names):
    for i, class_name in enumerate(class_names):
        class_names[i] = class_name.lower().replace('_', ' ').replace('-', ' ').replace('/', ' ')
    return class_names


def main():
    
#     import math
#     all_logits_base = torch.load('/media/cqu/D/FXV/R-TPT-main6/all_logits_tta.pth')
#     all_logits_mh = torch.load('/media/cqu/D/FXV/R-TPT-main6/all_logits_ttaours.pth')
#     all_labels = torch.load('/media/cqu/D/FXV/R-TPT-main6/all_labels.pth')
    

#     # ---- load/convert data ----
#     logits_base = to_numpy(all_logits_base)   # shape [N,C]
#     logits_mh   = to_numpy(all_logits_mh)     # shape [N,C]
#     labels      = to_numpy(all_labels).astype(int)  # shape [N]
#     
#     assert logits_base.shape == logits_mh.shape, "两个logits维度必须相同"
#     N, C = logits_base.shape
# 
#     # ---- probs / preds / acc ----
#     probs_base = softmax(logits_base)
#     probs_mh   = softmax(logits_mh)
# 
#     acc_base, preds_base = accuracy_from_probs(probs_base, labels)
#     acc_mh,   preds_mh   = accuracy_from_probs(probs_mh, labels)
# 
#     ece_base, bins_acc_base, bins_conf_base, bins_count_base = compute_ece(probs_base, labels, n_bins=15)
#     ece_mh,   bins_acc_mh,   bins_conf_mh,   bins_count_mh   = compute_ece(probs_mh, labels, n_bins=15)
# 
#     print(f"总体 Accuracy before: {acc_base:.4f}, after: {acc_mh:.4f}")
#     print(f"总体 ECE     before: {ece_base:.4f}, after: {ece_mh:.4f}")
#     
# # ---- 关键诊断指标：高置信错误比例 & 平均错误置信度 ----
#     def high_conf_error_stats(probs, labels, threshold=0.9):
#         confs = probs.max(axis=1)
#         preds = probs.argmax(axis=1)
#         errors_mask = (preds != labels)
#         n_errors = errors_mask.sum()
#         high_conf_errors = np.logical_and(errors_mask, confs >= threshold).sum()
#         prop_high_conf_errors = high_conf_errors / max(1, errors_mask.sum())
#         mean_conf_errors = confs[errors_mask].mean() if n_errors>0 else 0.0
#         return {
#         "n_errors": int(n_errors),
#         "high_conf_errors": int(high_conf_errors),
#         "prop_high_conf_errors": prop_high_conf_errors,
#         "mean_conf_errors": float(mean_conf_errors)
#     }
# 
#     th = 0.9
#     stats_base = high_conf_error_stats(probs_base, labels, threshold=th)
#     stats_mh   = high_conf_error_stats(probs_mh, labels, threshold=th)
# 
#     print("\n=== 高置信错误统计 (阈值 = {:.2f}) ===".format(th))
#     print("Before: ", stats_base)
#     print("After:  ", stats_mh)
#     print("Delta prop_high_conf_errors: {:.4f}".format(stats_mh["prop_high_conf_errors"] - stats_base["prop_high_conf_errors"]))
#     print("Delta mean_conf_errors: {:.4f}".format(stats_mh["mean_conf_errors"] - stats_base["mean_conf_errors"]))
# 
# # ---- 平均正确/错误置信度变化 ----
#     def mean_conf_correct_vs_error(probs, labels):
#         confs = probs.max(axis=1)
#         preds = probs.argmax(axis=1)
#         correct_mask = preds == labels
#         error_mask = ~correct_mask
#         return confs[correct_mask].mean() if correct_mask.sum()>0 else 0.0, confs[error_mask].mean() if error_mask.sum()>0 else 0.0
# 
#     mean_conf_corr_base, mean_conf_err_base = mean_conf_correct_vs_error(probs_base, labels)
#     mean_conf_corr_mh,   mean_conf_err_mh   = mean_conf_correct_vs_error(probs_mh, labels)
# 
#     print("\nMean conf (correct)  before:{:.4f}  after:{:.4f}  Δ:{:.4f}".format(mean_conf_corr_base, mean_conf_corr_mh, mean_conf_corr_mh-mean_conf_corr_base))
#     print("Mean conf (error)    before:{:.4f}  after:{:.4f}  Δ:{:.4f}".format(mean_conf_err_base,  mean_conf_err_mh,  mean_conf_err_mh-mean_conf_err_base))
# 
#     # ---- per-class ECE / delta (可选但很有用) ----
#     def per_class_stats(probs_a, probs_b, labels, n_bins=10):
#         C = probs_a.shape[1]
#         per = []
#         for c in range(C):
#             mask = (labels == c)
#             if mask.sum() == 0:
#                 per.append((c, np.nan, np.nan, 0))
#                 continue
#             ece_a, *_ = compute_ece(probs_a[mask], labels[mask], n_bins=n_bins)
#             ece_b, *_ = compute_ece(probs_b[mask], labels[mask], n_bins=n_bins)
#             per.append((c, ece_a, ece_b, mask.sum()))
#         return per
# 
#     perclass = per_class_stats(probs_base, probs_mh, labels, n_bins=10)
#     # 打印若干类：ECE 提升最多的 top_k
#     perclass_valid = [p for p in perclass if not math.isnan(p[1])]
#     perclass_sorted = sorted(perclass_valid, key=lambda x: (x[2]-x[1]), reverse=True)
#     print("\nTop 8 classes with ECE increase (class, ece_before, ece_after, n_samples):")
#     for item in perclass_sorted[:8]:
#         print(item)
# 
#     # ---- 查找置信度提升最多的错误样本 ----
#     confs_base = probs_base.max(axis=1)
#     confs_mh   = probs_mh.max(axis=1)
#     delta_conf = confs_mh - confs_base
# 
#     # 只保留最终为错误的样本（after-case 错误），然后看他们置信度提升
#     err_after_mask = (probs_mh.argmax(axis=1) != labels)
#     idxs = np.where(err_after_mask)[0]
#     if len(idxs)>0:
#         idxs_sorted = idxs[np.argsort(-delta_conf[idxs])]  # 按 delta_conf 降序
#         topk = min(20, len(idxs_sorted))
#         print(f"\nTop {topk} examples (indices) whose confidence increased the most but are errors after MH:")
#         for i in idxs_sorted[:topk]:
#             print(f"idx={i}, label={labels[i]}, pred_before={probs_base[i].argmax()}, pred_after={probs_mh[i].argmax()}, conf_before={confs_base[i]:.4f}, conf_after={confs_mh[i]:.4f}, Δ={delta_conf[i]:.4f}")
#     else:
#         print("\nNo errors after MH (err_after_mask empty).")
# 
#     # ---- 可视化（可选）: 置信度直方图 和 reliability diagram 对比 ----
#     fig, axes = plt.subplots(1,2, figsize=(12,4))
#     axes[0].hist(confs_base, bins=50, alpha=0.6, label='before'); axes[0].hist(confs_mh, bins=50, alpha=0.6, label='after')
#     axes[0].set_title("Confidence histograms (before vs after)")
#     axes[0].legend()
# 
#     # reliability diagram: plot avg accuracy vs avg confidence per bin
#     bins = np.linspace(0.0, 1.0, 11)
#     bin_centers = (bins[:-1] + bins[1:]) / 2
#     accs_b = np.array(bins_acc_base)
#     confs_b = np.array(bins_conf_base)
#     accs_m = np.array(bins_acc_mh)
#     confs_m = np.array(bins_conf_mh)
# 
#     axes[1].plot(bin_centers, accs_b, marker='o', label='acc before'); axes[1].plot(bin_centers, confs_b, marker='x', label='conf before')
#     axes[1].plot(bin_centers, accs_m, marker='o', linestyle='--', label='acc after'); axes[1].plot(bin_centers, confs_m, marker='x', linestyle='--', label='conf after')
#     axes[1].set_ylim(0,1)
#     axes[1].set_title("Reliability Diagram (binned acc vs conf)")
#     axes[1].legend()
#     plt.tight_layout()
#     plt.show()
    
    args = parser.parse_args()
    set_random_seed(args.seed)

#     sigma_img = torch.load('/media/cqu/D/FXV/R-TPT-main6/sigma_img.pth')
#     sigma_txt = torch.load('/media/cqu/D/FXV/R-TPT-main6/sigma_dict.pth')
#          
#     features_per_class = torch.load("/media/cqu/D/FXV/R-TPT-main6/features_per_class.pth")
# 
#     sigma_img = tikhonov_shrinkage_preserve_type(sigma_img)
#     sigma_txt = tikhonov_shrinkage_preserve_type(sigma_txt)
# 
#     sigma_img_dict = torch.load('/media/cqu/D/FXV/R-TPT-main6/sigma_img_dict.pth')

    
#     correlations = analyze_sigma_corr(
#     sigma_txt_dict=sigma_txt,
#     sigma_img_dict=sigma_img,
#     sigma_img_dict2=sigma_img_dict,
#     scale_fn=scale_sigma_dict_txt2img_congruence4_noclip,
#     save_path="/media/cqu/D/FXV/R-TPT-main6/diag_corr_hist.pdf"
# )

#     out = scale_sigma_dict_txt2img_congruence4_noclip(sigma_txt, sigma_img_dict['sprinkled'], ref_class='sprinkled')
#     out = scale_sigma_dict_txt2imgori(sigma_txt, sigma_img_dict['pleated'])
#     covariance_correlations_histogram(sigma_img, sigma_txt, out, save_path="/media/cqu/D/FXV/R-TPT-main6/cov_corr_histnew2.pdf")


    args.alpha = args.eps / 4.0
    args.output_dir = os.path.join(args.output_dir, args.arch, args.test_sets, 'eps_'+str(args.eps)+'_alpha_'+str(args.alpha)+'_step_'+str(args.steps))

    if not os.path.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.out_file = open(os.path.join(args.output_dir, 'log.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()

    assert args.gpu is not None

    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))

    # model
    dset = args.test_sets
    
    ###### load robust vision encoder (TeCoA) ######
    if len(args.load_tecoa) > 0:
        args.robust_pretrain_path = {
            'RN50-eps1': 'pretrain/tecoa/rn50_eps1.pth.tar',
        }[args.load_tecoa]
        robust_state_dict = torch.load(args.robust_pretrain_path, map_location='cpu')
        model.image_encoder.load_state_dict(robust_state_dict['vision_encoder_state_dict'])
        print('load robust vision encoder')

#     for name, param in model.named_parameters():
#         if "prompt_learner" not in name:
#                 param.requires_grad_(False)



    scaler = None
    cudnn.benchmark = True
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    # iterating through eval datasets
    
    results = {}
    if True:
        base_transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution)])
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            # normalize
            ])
        
        
        preprocess = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15), # TODO: may use later
        transforms.ToTensor()
    ])
        preprocess224 = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),  # 统一通道
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15), # TODO: may use later
        transforms.ToTensor()
    ])
        preprocess224_interpolate = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15), # TODO: may use later
        transforms.ToTensor()
    ])
        
        imagenet_root = '/media/cqu/D/FXV/PSSR_master/TINY-IMAGE/ILSVRC2012_val'
        tinyimagenet_root = '/media/cqu/D/FXV/PSSR_master/TINY-IMAGE/tiny-imagenet-200'
        
        data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size-1, 
                                        augmix=len(dset)>1)
        batchsize = 50
        
        if args.test_sets == 'tinyImageNet':
            val_dataset = torchvision.datasets.ImageFolder(
                os.path.join(tinyimagenet_root, 'val/images'),
                transform=preprocess224)
        elif args.test_sets == 'cifar10':
            val_dataset = CIFAR10(args.data, transform=preprocess224,
                                            download=True, train=False)
        elif args.test_sets == 'cifar100':
            val_dataset = CIFAR100(args.data, transform=preprocess224,
                                            download=True, train=False)
        elif args.test_sets == 'STL10':
            val_dataset = STL10(args.data, transform=preprocess224,split='test',
                                            download=True)
        elif args.test_sets == 'Caltech256':
            val_dataset = Caltech256(args.data, transform=preprocess224,
                                            download=True)
        elif args.test_sets == 'ImageNet':
            val_dataset = torchvision.datasets.ImageFolder(
                os.path.join(imagenet_root, 'ILSVRC2012_val2'),
                transform=preprocess224)
        elif args.test_sets == 'PCAM':
            val_dataset = PCAM(args.data, transform=preprocess224,split='test',
                                            download=True)
        elif args.test_sets == 'SUN397':
            val_dataset = SUN397(args.data, transform=preprocess224,
                                            download=True)

        val_dataset = build_dataset(dset, preprocess224, args.data, mode=args.dataset_mode)
        print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=False,
                    num_workers=args.workers, pin_memory=True)
# 
    if len(dset) > 1: 
        classnames = eval("{}_classes".format(dset.lower()))
    else:
        assert dset in ['A', 'R', 'K', 'V', 'I']
        classnames_all = imagenet_classes
        classnames = []
        if dset in ['A', 'R', 'V']:
            label_mask = eval("imagenet_{}_mask".format(dset.lower()))
            if dset == 'R':
                for i, m in enumerate(label_mask):
                    if m:
                        classnames.append(classnames_all[i])
            else:
                classnames = [classnames_all[i] for i in label_mask]
        else:
            classnames = classnames_all
# #             
#     classnames = val_dataset.classes        
#     folder2name = load_imagenet_folder2name('imagenet_classes_names.txt')
#     new_class_names = []
#     for each in classnames:
#         new_class_names.append(folder2name[each])
# 
#     classnames = new_class_names
# 
#     classnames = refine_classname(classnames)
#     print(classnames)


#cifar 100
    #classnames = val_dataset.clip_prompts
#     classnames = val_dataset.classes
#     classnames = [s.replace("A photo of a ", "").replace(".", "") for s in classnames]

#PCAM
#     classnames = val_dataset.clip_prompts
#     classnames = [s.replace("Tthis is a photo of ", "").replace(".", "") for s in classnames]
#     classnames = [s.replace("This is a photo of ", "").replace(".", "") for s in classnames]

    #classnames = [template.format(label) for label in texts_tmp]
    #classnames = refine_classname(classnames)
    print(classnames)
    
    
    
    args.classnames = classnames
    
    model = get_coop(args.arch, classnames, args.gpu, args.n_ctx, args.ctx_init)
    model_state = None
    
    print("=> Model created: visual backbone {}".format(args.arch))
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    trainable_param = model.prompt_learner.parameters()
    optimizer = torch.optim.AdamW(trainable_param, args.lr)
    optim_state = deepcopy(optimizer.state_dict())
    
    mu_dict = 1
    sigma_dict = 1
    


#     correlations_ma = run_and_plot(mu_dict, sigma_img, sigma_txt, sigma_img_dict, save_path="/media/cqu/D/FXV/R-TPT-main6/corr_hist.pdf")



    
    


        
    results, all_logits, all_logits_tta, all_labels, sigma_img, center_dict, sigma_img_dict = test_time_adapt_eval(classnames, sigma_dict,mu_dict,  val_loader, model, model_state, optimizer, optim_state, scaler, args, data_transform)
        #torch.save(sigma_img_dict, "/media/cqu/D/FXV/R-TPT-main6/sigma_img_dict.pth")
    
    ece_dict_tta = compute_classwise_ece_true(all_logits_tta, all_labels)
    dist_dict_diag = compute_mean_mahalanobis(mu_dict, sigma_img, mode="diag")
   


#         torch.save(sigma_img, "/media/cqu/D/FXV/R-TPT-main6/sigma_img.pth")
#         torch.save(sigma_dict, "/media/cqu/D/FXV/R-TPT-main6/sigma_dict.pth")

        # 在测试结束后或某个时刻做 shrink：
    sigma_shrinked = shrink_text_covariance(sigma_dict, sigma_img_dict, alpha=0.3, diag_shrink=True)
        
    results_tta = []
    for i, cls in enumerate(classnames):
            results_tta.append({
            "class": cls,
            "mahalanobis": dist_dict_diag.get(cls, None),
            "ece": ece_dict_tta.get(i, None)  # labels 是类别 index
        })
        #plot_mahalanobis_vs_ece(results_tta, save_path="/media/cqu/D/FXV/R-TPT-main6/mahalanobis_vs_ecetta.pdf")

    del val_dataset, val_loader
    if args.eps <= 0:
        print_log = "=> Acc. on testset [{}]: Clean Acc @1 {}/ TTA Clean Acc @1 {}".format(dset, results[0], results[1])
        save_log = {'clean_acc': results[0], 'tta_clean_acc': results[1]}
    else:
        print_log = "=> Acc. on testset [{}]: Adv Acc @1 {}/ TTA Adv Acc @1 {} ".format(dset, results[0], results[1])
        save_log = {'adv_acc': results[0], 'tta_adv_acc': results[1]}
      
    args.out_file.write(print_log + '\n')
    args.out_file.flush()
    print(print_log+'\n')

    torch.save(save_log, os.path.join(args.output_dir, 'results_log.pt'))
def compute_per_class_stats(all_logits, all_labels, n_bins=15):
    """
    计算每个类别的平均置信度、ECE和样本数量
    
    输入：
        all_logits: [N, D] Tensor，未经过 softmax
        all_labels: [N] Tensor，标签
        n_bins: ECE 计算的区间数
    返回：
        stats_dict: {class_idx: {'avg_conf': float, 'ece': float, 'n_samples': int}}
    """
    N, D = all_logits.shape
    all_probs = torch.softmax(all_logits, dim=1)
    pred_labels = torch.argmax(all_probs, dim=1)
    max_confs = torch.max(all_probs, dim=1).values

    classes = torch.unique(all_labels)
    stats_dict = {}

    for c in classes:
        mask = (all_labels == c)
        probs_c = all_probs[mask]          # [n_c, D]
        pred_c = pred_labels[mask]         # [n_c]
        max_conf_c = max_confs[mask]       # [n_c]
        n_c = mask.sum().item()

        # 平均置信度
        avg_conf = max_conf_c.mean().item()

        # ECE
        ece = 0.0
        if n_c > 0:
            bin_boundaries = torch.linspace(0, 1, n_bins + 1)
            for i in range(n_bins):
                low, high = bin_boundaries[i], bin_boundaries[i + 1]
                in_bin = (max_conf_c >= low) & (max_conf_c < high)
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    acc_in_bin = (pred_c[in_bin] == all_labels[mask][in_bin]).float().mean()
                    avg_conf_in_bin = max_conf_c[in_bin].mean()
                    ece += torch.abs(avg_conf_in_bin - acc_in_bin) * prop_in_bin

        stats_dict[int(c.item())] = {'avg_conf': avg_conf, 'ece': ece.item(), 'n_samples': n_c}

    return stats_dict


def shuffle_image(image, block_size=(32, 32), return_indices=False):
    """
    将图像划分为多个块，随机打乱这些块并重新组合。

    参数：
    - image: 输入图像，形状为 [1, 3, H, W]
    - block_size: 每个块的大小，默认为 (32, 32)
    - return_indices: 是否返回打乱索引，用于后续恢复

    返回：
    - 打乱后的图像，形状为 [1, 3, H, W]
    - 如果return_indices=True，同时返回打乱索引
    """
    batch_size, channels, h, w = image.shape
    
    # 确保图像尺寸能被块大小整除
    assert h % block_size[0] == 0 and w % block_size[1] == 0, \
        "图像尺寸必须能被块大小整除"
    
    # 计算块的数量
    num_blocks_h = h // block_size[0]
    num_blocks_w = w // block_size[1]
    num_blocks = num_blocks_h * num_blocks_w
    
    # 将图像划分为块 [B, C, H//bh, bh, W//bw, bw]
    blocks = image.view(batch_size, channels, num_blocks_h, block_size[0], num_blocks_w, block_size[1])
    
    # 重排列为 [B, C, (H//bh * W//bw), bh, bw]
    blocks = blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
    blocks = blocks.view(batch_size, channels, num_blocks, block_size[0], block_size[1])
    
    # 生成随机打乱索引
    random_indices = torch.randperm(num_blocks)
    
    # 应用打乱
    shuffled_blocks = blocks[:, :, random_indices]
    
    # 重组为图像 [B, C, H, W]
    # 先重组为 [B, C, H//bh, W//bw, bh, bw]
    shuffled_blocks = shuffled_blocks.view(batch_size, channels, num_blocks_h, num_blocks_w, block_size[0], block_size[1])
    # 再重组为 [B, C, H, W]
    shuffled_image = shuffled_blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
    shuffled_image = shuffled_image.view(batch_size, channels, h, w)
    
    if return_indices:
        return shuffled_image, random_indices
    else:
        return shuffled_image

def restore_image(shuffled_image, original_indices, block_size=(32, 32)):
    """
    将打乱的图像恢复为原始顺序。

    参数：
    - shuffled_image: 打乱后的图像，形状为 [1, 3, H, W]
    - original_indices: 原始的打乱索引
    - block_size: 每个块的大小，默认为 (32, 32)

    返回：
    - 恢复后的图像，形状为 [1, 3, H, W]
    """
    batch_size, channels, h, w = shuffled_image.shape
    
    # 计算块的数量
    num_blocks_h = h // block_size[0]
    num_blocks_w = w // block_size[1]
    num_blocks = num_blocks_h * num_blocks_w
    
    # 将打乱图像划分为块 [B, C, H//bh, bh, W//bw, bw]
    blocks = shuffled_image.view(batch_size, channels, num_blocks_h, block_size[0], num_blocks_w, block_size[1])
    
    # 重排列为 [B, C, (H//bh * W//bw), bh, bw]
    blocks = blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
    blocks = blocks.view(batch_size, channels, num_blocks, block_size[0], block_size[1])
    
    # 计算逆索引
    inverse_indices = torch.argsort(original_indices)
    
    # 使用逆索引恢复原始顺序
    restored_blocks = blocks[:, :, inverse_indices]
    
    # 重组为原始图像 [B, C, H, W]
    restored_blocks = restored_blocks.view(batch_size, channels, num_blocks_h, num_blocks_w, block_size[0], block_size[1])
    restored_image = restored_blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
    restored_image = restored_image.view(batch_size, channels, h, w)
    
    return restored_image

def add_gaussian_noise(image, noise_std=0.3, noise_mean=0.1, clip=True, return_noise=True):
    """
    给图像添加高斯噪声
    
    参数:
    - image: 输入图像，形状为 [B, C, H, W] 或 [C, H, W]，值范围建议为[0,1]或[-1,1]
    - noise_std: 噪声的标准差，控制噪声强度
    - noise_mean: 噪声的均值，通常为0
    - clip: 是否将结果裁剪到与输入相同的范围
    - return_noise: 是否返回噪声张量
    
    返回:
    - 添加噪声后的图像
    - 如果return_noise=True，同时返回噪声张量
    """
    
    # 确保输入是张量
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image, dtype=torch.float32)
    
    # 保存原始设备
    device = image.device
    
    # 生成与输入图像相同形状的高斯噪声
    noise = torch.randn_like(image) * noise_std + noise_mean
    
    # 添加噪声
    noisy_image = image + noise
    
    # 如果需要，将结果裁剪到原始范围
    if clip:
        if image.min() >= 0 and image.max() <= 1:  # [0,1]范围
            noisy_image = torch.clamp(noisy_image, 0, 1)
        elif image.min() >= -1 and image.max() <= 1:  # [-1,1]范围
            noisy_image = torch.clamp(noisy_image, -1, 1)
        else:
            # 保持原始范围
            min_val, max_val = image.min(), image.max()
            noisy_image = torch.clamp(noisy_image, min_val, max_val)
    
    if return_noise:
        return noisy_image, noise
    else:
        return noisy_image
    
    
import torch

def add_uniform_noise(
    image,
    noise_std=0.3,
    noise_mean=0.0,
    clip=True,
    return_noise=True
):
    """
    给图像添加 Uniform 噪声，方差与 Gaussian(noise_std) 相同
    ε ~ U(-sqrt(3)*σ, sqrt(3)*σ)
    """

    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image, dtype=torch.float32)

    device = image.device

    # Uniform noise with same variance
    a = (3 ** 0.5) * noise_std
    noise = (torch.rand_like(image) * 2 - 1) * a + noise_mean

    noisy_image = image + noise

    if clip:
        if image.min() >= 0 and image.max() <= 1:
            noisy_image = torch.clamp(noisy_image, 0, 1)
        elif image.min() >= -1 and image.max() <= 1:
            noisy_image = torch.clamp(noisy_image, -1, 1)
        else:
            min_val, max_val = image.min(), image.max()
            noisy_image = torch.clamp(noisy_image, min_val, max_val)

    if return_noise:
        return noisy_image, noise
    else:
        return noisy_image

def cosine_similarity(tensor1, tensor2, dim=None, eps=1e-8):
    """
    计算两个张量之间的余弦相似度
    
    参数:
    - tensor1, tensor2: 输入张量
    - dim: 沿着哪个维度计算相似度，None表示整体相似度
    - eps: 小值避免除零
    
    返回:
    - 余弦相似度，范围[-1, 1]
    """
    # 确保张量形状相同
    assert tensor1.shape == tensor2.shape, "张量形状必须相同"
    
    if dim is None:
        # 整体余弦相似度
        tensor1_flat = tensor1.flatten()
        tensor2_flat = tensor2.flatten()
        
        dot_product = torch.dot(tensor1_flat, tensor2_flat)
        norm1 = torch.norm(tensor1_flat)
        norm2 = torch.norm(tensor2_flat)
        
        similarity = dot_product / (norm1 * norm2 + eps)
        
    else:
        # 沿着指定维度计算
        dot_product = torch.sum(tensor1 * tensor2, dim=dim)
        norm1 = torch.norm(tensor1, dim=dim)
        norm2 = torch.norm(tensor2, dim=dim)
        
        similarity = dot_product / (norm1 * norm2 + eps)
    
    return similarity

def analyze_vector_relationship(eta_a, g_a):
    """
    深入分析两个噪声向量的关系
    """
    # 基本统计
    print("=== 噪声向量分析 ===")
    print(f"η_a 范数: {torch.norm(eta_a).item():.4f}")
    print(f"g_a 范数: {torch.norm(g_a).item():.4f}")
    print(f"余弦相似度: {cosine_similarity(eta_a, g_a).item():.4f}")
    
    # 计算夹角（度）
    angle_rad = torch.acos(torch.clamp(cosine_similarity(eta_a, g_a), -1, 1))
    angle_deg = angle_rad * 180 / torch.pi
    print(f"向量夹角: {angle_deg.item():.1f}°")
    
    # 分析分量关系
    dot_product = torch.dot(eta_a.flatten(), g_a.flatten())
    print(f"点积: {dot_product.item():.4f}")
    
    # 投影分析
    eta_norm = torch.norm(eta_a)
    g_projection_on_eta = dot_product / eta_norm
    print(f"g_a 在 η_a 方向上的投影: {g_projection_on_eta.item():.4f}")
    
    return {
        'angle_degrees': angle_deg.item(),
        'norms': (torch.norm(eta_a).item(), torch.norm(g_a).item()),
        'dot_product': dot_product.item()
    }

def entropy_from_logits(logits, eps=1e-8):
    """
    从未归一化的logits计算香农熵
    
    参数:
    - logits: 神经网络的原始输出，形状为 [batch_size, num_classes] 或 [num_classes]
    - eps: 小值避免log(0)
    
    返回:
    - entropy: 熵值，形状为 [batch_size] 或 标量
    """
    # 应用softmax得到概率分布
    probabilities = F.softmax(logits, dim=-1)
    
    # 计算熵: H = -Σ p_i * log(p_i)
    log_probs = torch.log(probabilities + eps)
    entropy = -torch.sum(probabilities * log_probs, dim=-1)
    
    return entropy
def entropy(p, dim=-1, keepdim=False):
    """
    计算离散概率分布的信息熵
    
    参数:
        p: 概率分布张量，每一行是一个概率分布
        dim: 计算熵的维度
        keepdim: 是否保持维度
    
    返回:
        熵值张量
    """
    # 添加小量避免log(0)
    
    p = F.softmax(p, dim=-1)
    epsilon = 1e-10
    p = p + epsilon
    
    # 计算熵: H(p) = -∑ p_i * log(p_i)
    entropy_val = -torch.sum(p * torch.log(p), dim=dim, keepdim=keepdim)
    return entropy_val


def row_wise_correlation(tensor1, tensor2):
    """
    计算两个张量行与行之间的皮尔逊相关系数
    
    Args:
        tensor1: 形状为 [n, m] 的张量
        tensor2: 形状为 [n, m] 的张量
        
    Returns:
        长度为 n 的张量，包含每对行向量的相关系数
    """
    # 确保形状相同
    assert tensor1.shape == tensor2.shape, "两个张量形状必须相同"
    
    n, m = tensor1.shape
    
    # 计算每行的均值
    mean1 = tensor1.mean(dim=1, keepdim=True)
    mean2 = tensor2.mean(dim=1, keepdim=True)
    
    # 中心化
    centered1 = tensor1 - mean1
    centered2 = tensor2 - mean2
    
    # 计算分子：协方差的和
    numerator = (centered1 * centered2).sum(dim=1)
    
    # 计算分母：标准差的乘积
    std1 = centered1.norm(dim=1)  # 等价于 sqrt(sum(centered1^2))
    std2 = centered2.norm(dim=1)  # 等价于 sqrt(sum(centered2^2))
    
    # 计算相关系数，避免除以0
    denominator = std1 * std2
    # 添加一个小值避免除0
    epsilon = 1e-8
    corr = numerator / (denominator + epsilon)
    
    return corr



def add_rademacher_noise(
    image,
    noise_std=0.3,
    noise_mean=0.0,
    clip=True,
    return_noise=True
):
    """
    给图像添加 Rademacher 噪声（±1），方差与 Gaussian(noise_std) 相同
    ε = σ * r, r ∈ {+1, -1}
    """

    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image, dtype=torch.float32)

    device = image.device

    # Rademacher noise
    noise = torch.empty_like(image).bernoulli_(0.5)
    noise = noise * 2 - 1          # {+1, -1}
    noise = noise * noise_std + noise_mean

    noisy_image = image + noise

    if clip:
        if image.min() >= 0 and image.max() <= 1:
            noisy_image = torch.clamp(noisy_image, 0, 1)
        elif image.min() >= -1 and image.max() <= 1:
            noisy_image = torch.clamp(noisy_image, -1, 1)
        else:
            min_val, max_val = image.min(), image.max()
            noisy_image = torch.clamp(noisy_image, min_val, max_val)

    if return_noise:
        return noisy_image, noise
    else:
        return noisy_image


def compute_cosine_similarity_excluding_row(A, B, idx):
    """
    计算A与B（排除idx行）的余弦相似度均值
    
    Args:
        A: Tensor of shape [1, 1024]
        B: Tensor of shape [47, 1024]
        idx: 要排除的行索引
    
    Returns:
        余弦相似度均值
    """
    # 从B中排除idx行
    mask = torch.ones(B.shape[0], dtype=torch.bool)
    mask[idx] = False
    B_excluded = B[mask]  # 形状变为 [46, 1024]
    
    # 计算余弦相似度
    # 使用F.cosine_similarity，注意要指定维度
    # A的形状是[1, 1024]，B_excluded是[46, 1024]，需要扩展A或使用广播
    similarity = F.cosine_similarity(
        A.expand_as(B_excluded),  # 将A扩展为[46, 1024]
        B_excluded,
        dim=1  # 在特征维度1024上计算相似度
    )
    
    # 计算均值
    mean_similarity = similarity.mean().item()
    
    return mean_similarity

def generate_clip_style_augs(x, n_views=128, add_noise=True):
    """
    x: tensor, shape [1, 3, 224, 224]
    return: tensor [n_views, 3, 224, 224]
    """
    assert x.ndim == 4 and x.shape[1] == 3, "Input must be [1, 3, 224, 224]"
    img = x[0]  # [3,224,224]

    def clip_aug(img):
        # --- 1. 随机ResizedCrop（更CLIP-friendly） ---
        # scale 控制裁剪大小（保持语义不变）
        scale = random.uniform(0.8, 1.0)   # 只做少量裁剪/缩放
        new_size = int(224 * scale)
        top = random.randint(0, 224 - new_size)
        left = random.randint(0, 224 - new_size)

        out = img[:, top:top+new_size, left:left+new_size]
        out = F.interpolate(out.unsqueeze(0), size=(224, 224),
                            mode="bilinear", align_corners=False)[0]

        # --- 2. 随机水平翻转（无语义损伤） ---
        if random.random() < 0.5:
            out = torch.flip(out, [2])

        # --- 3. CLIP-friendly 轻微高斯噪声 ---
        if add_noise:
            # 控制噪声极轻，只影响高频噪声（非语义）
            if random.random() < 0.7:
                noise_std = random.uniform(0.005, 0.02)
                noise = torch.randn_like(out) * noise_std
                out = (out + noise).clamp(0, 1)

        return out

    return torch.stack([clip_aug(img) for _ in range(n_views)], dim=0)


def feature_lowpass(feat, keep_ratio=0.95):
    # feat: [1024]
    f = torch.fft.fft(feat)
    N = len(f)
    keep = int(N * keep_ratio)
    
    f_filtered = torch.zeros_like(f)
    f_filtered[:keep] = f[:keep]      # 保留低频
    f_filtered[-keep:] = f[-keep:]    # 保留负频

    out = torch.fft.ifft(f_filtered).real
    return out.unsqueeze(0)

def gaussian_blur(x, kernel_size=3, sigma=5.0):
    """
    x: [1, 3, H, W]
    """
    device = x.device
    channels = x.shape[1]

    # 1D gaussian kernel
    grid = torch.arange(kernel_size, dtype=torch.float32, device=device)
    mean = kernel_size // 2
    gaussian = torch.exp(-((grid - mean)**2) / (2 * sigma**2))
    kernel1d = gaussian / gaussian.sum()

    # 2D kernel
    kernel2d = torch.outer(kernel1d, kernel1d)
    kernel2d = kernel2d.expand(channels, 1, kernel_size, kernel_size)

    return F.conv2d(x, kernel2d, padding=kernel_size//2, groups=channels)

def nonlocal_feature_denoise(z, V, tau=0.07, topk=32):
    """
    特征域非局部去噪
    z: [D] 待去噪特征
    V: [N, D] 候选特征集合（多视角特征）
    tau: 温度控制
    topk: 选择最相似的 topk 特征做加权
    """
    sims = F.cosine_similarity(z.unsqueeze(0), V, dim=1)  # [N]
    vals, idx = sims.topk(topk, largest=True)
    weights = F.softmax(vals / tau, dim=0)                # [topk]
    V_sel = V[idx]                                       # [topk, D]
    z_denoised = (weights.unsqueeze(1) * V_sel).sum(0)   # [D]
    return z_denoised

def pca_soft_threshold(Z, lam=0.5):
    """
    PCA soft-threshold 去噪
    Z: [N, D] 多视角特征
    lam: 阈值，控制软收缩强度
    """
    mean = Z.mean(0, keepdim=True)
    Zc = Z - mean
    U, S, Vt = torch.linalg.svd(Zc, full_matrices=False)
    S_shrink = F.relu(S - lam)
    Z_rec = (U * S_shrink.unsqueeze(0)) @ Vt + mean
    return Z_rec

def feature_denoise_pipeline(feats, tau=0.07, topk=32, lam=0.5):
    """
    feats: [64, 1024] 多视角特征
    返回: [1, 1024] 去噪特征
    """
    N, D = feats.shape
    denoised_feats = []
    
    # 非局部去噪
    for i in range(N):
        z_denoised = nonlocal_feature_denoise(feats[i], feats, tau=tau, topk=topk)
        denoised_feats.append(z_denoised)
    
    denoised_feats = torch.stack(denoised_feats, dim=0)  # [N,D]
    
    # PCA soft-threshold 去噪
    denoised_feats = pca_soft_threshold(denoised_feats, lam=lam)  # [N,D]
    
    # 最终平均得到单个特征
    final_feat = denoised_feats.mean(0, keepdim=True)  # [1,D]
    return final_feat
def denoise_bilateral(image_tensor, d=5, sigmaColor=30, sigmaSpace=30):
    B, C, H, W = image_tensor.shape
    out = torch.zeros_like(image_tensor)
    for b in range(B):
        img = (image_tensor[b].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        img_denoise = cv2.bilateralFilter(img, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
        img_denoise = torch.tensor(img_denoise, dtype=image_tensor.dtype).permute(2,0,1) / 255.0
        out[b] = img_denoise.to(image_tensor.device)
    return out
def denoise_gaussian(image, kernel_size=5, sigma=0.95):
    blur = T.GaussianBlur(kernel_size=kernel_size, sigma=(sigma, sigma))
    return blur(image)
def geometric_median(feats, eps=1e-6, iters=15):
    """
    feats: [K, D], normalized
    return: [1, D]
    """
    y = feats.mean(dim=0)

    for _ in range(iters):
        dist = torch.norm(feats - y, dim=1)
        weight = 1.0 / (dist + eps)
        y = (feats * weight.unsqueeze(1)).sum(dim=0) / weight.sum()

    y = y / y.norm()
    return y.unsqueeze(0)
def principal_direction(feats):
    """
    feats: [K, D], normalized
    return: [1, D]
    """
    X = feats - feats.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    v = Vh[0]
    v = v / v.norm()
    return v.unsqueeze(0)
def consensus_weighted_mean(feats):
    """
    feats: [K, D], normalized
    return: [1, D]
    """
    sim = feats @ feats.t()      # [K, K]
    weight = sim.mean(dim=1)     # [K]
    weight = torch.softmax(weight, dim=0)
    out = (feats * weight.unsqueeze(1)).sum(dim=0)
    out = out / out.norm()
    return out.unsqueeze(0)

def extract_noise_features(
    model,
    x,
    num_samples=2,
    sigma=0.03
):
    """
    返回:
        f_x      : [1, D]
        f_xeps   : [N, D]
    """
    #f_x = model(x)  # [1, D]
    f_x, text_features, logit_scale = model.forward_features(x)
    feats = []
    for _ in range(num_samples):
        eps = torch.randn_like(x) * (sigma ** 0.5)
        x_eps = x + eps
        f, text_features, logit_scale = model.forward_features(x_eps)
        feats.append(f)

    f_xeps = torch.cat(feats, dim=0)  # [N, D]
    return f_x, f_xeps


def random_projection_ratio(
    f_xeps,
    f_x,
    K=10,
    num_trials=100
):
    V_noise = f_xeps - f_x
    V_noise = V_noise - V_noise.mean(dim=0, keepdim=True)

    _, _, Vh = torch.linalg.svd(V_noise, full_matrices=False)
    U_noise = Vh[:K].T  # [D, K]

    ratios = []
    D = U_noise.shape[0]

    for _ in range(num_trials):
        v_rand = torch.randn(D, device=U_noise.device)
        v_rand = v_rand / v_rand.norm()
        r = torch.norm(U_noise.T @ v_rand)
        ratios.append(r.item())

    return sum(ratios) / len(ratios)
def adversarial_spectral_energy(
    f_x,        # [1, D]
    f_xadv,     # [1, D]
    f_xeps      # [N, D]
):
    """
    返回：
        energy: [D] 对抗方向在每个噪声 PCA 方向上的能量
        S:      [D] 噪声奇异值（用于画谱）
    """

    # ---- 对抗方向 ----
    v_adv = f_xadv - f_x
    v_adv = F.normalize(v_adv, dim=-1).squeeze(0)  # [D]

    # ---- 噪声方向 ----
    V_noise = f_xeps - f_x
    V_noise = V_noise - V_noise.mean(dim=0, keepdim=True)

    # ---- SVD ----
    _, S, Vh = torch.linalg.svd(V_noise, full_matrices=False)
    U = Vh  # [D, D]，每一行是一个 PCA 方向

    # ---- 对抗能量分布 ----
    # energy_i = |<u_i, v_adv>|^2
    energy = (U @ v_adv) ** 2  # [D]

    return energy.cpu(), S.cpu()
def top1_pca_energy_ratio(
    f_x,        # [1, D]
    f_xadv,     # [1, D]
    f_xeps      # [N, D]
):
    """
    返回：
        adv_ratio  : 对抗方向在 Top-1 PCA 上的能量占比
        rand_ratio : 随机方向在 Top-1 PCA 上的能量占比
    """

    # ---------- 对抗方向 ----------
    v_adv = f_xadv - f_x
    v_adv = F.normalize(v_adv, dim=-1).squeeze(0)  # [D]

    # ---------- 噪声方向 ----------
    V_noise = f_xeps - f_x
    V_noise = V_noise - V_noise.mean(dim=0, keepdim=True)

    # ---------- PCA ----------
    _, _, Vh = torch.linalg.svd(V_noise, full_matrices=False)
    u1 = Vh[0]   # Top-1 PCA direction [D]

    # ---------- 能量 ----------
    adv_ratio = (torch.dot(v_adv, u1)) ** 2

    v_rand = torch.randn_like(v_adv)
    v_rand = v_rand / v_rand.norm()
    rand_ratio = (torch.dot(v_rand, u1)) ** 2

    return adv_ratio.item(), rand_ratio.item()

def compute_noise_pca(noise_feats, f_clean):
    """
    noise_feats: [N, D] = f(x + eps_i)
    f_clean:     [1, D]
    """
    # noise directions
    X = noise_feats - f_clean  # [N, D]
    X = X - X.mean(dim=0, keepdim=True)

    # PCA via SVD
    # X = U S V^T,  V: [D, D]
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    # PCA directions are rows of Vh
    return Vh  # [D, D]

def projection_ratio(v, basis, k):
    """
    v:     [D]
    basis: [D, D]  (PCA basis, row-wise)
    """
    Uk = basis[:k]          # [k, D]
    proj = Uk @ v           # [k]
    return (proj.norm() ** 2 / (v.norm() ** 2 + 1e-8)).item()

def run_topk_alignment(
    f_clean, f_adv, noise_feats, ks=range(1, 51)
):
    """
    f_clean:     [1, D]
    f_adv:       [1, D]
    noise_feats: [N, D]
    """
    D = f_clean.shape[-1]

    # PCA
    Vh = compute_noise_pca(noise_feats, f_clean)

    # adv direction
    v_adv = (f_adv - f_clean).squeeze(0)

    # random direction baseline
    v_rand = torch.randn_like(v_adv)
    v_rand = v_rand / v_rand.norm()
    
    v_adv = v_adv / v_adv.norm()

    adv_curve = []
    rand_curve = []

    for k in ks:
        adv_curve.append(projection_ratio(v_adv, Vh, k))
        rand_curve.append(projection_ratio(v_rand, Vh, k))

    return adv_curve, rand_curve



def estimate_semantic_anchor(
    x_adv,
    encode,
    num_samples=64,
    noise_std=0.1
):
    zs = []

    for _ in range(num_samples):
        eps = torch.randn_like(x_adv) * noise_std

        
        z, text_features, logit_scale = encode.forward_features(x_adv + eps)
        
        z = F.normalize(z, dim=-1)
        zs.append(z)

    z_bar = torch.mean(torch.stack(zs, dim=0), dim=0)
    z_bar = F.normalize(z_bar, dim=-1)

    return z_bar  # [1, D]


def mode_control_projection(
    x_adv,
    encode,
    z_anchor,
    lam=0.1
):
    #z = encode(x_adv)
    
    z, text_features, logit_scale = encode.forward_features(x_adv)
    
    
    z = F.normalize(z, dim=-1)

    z_anchor = F.normalize(z_anchor, dim=-1)

    # projection
    coeff = (z * z_anchor).sum(dim=-1, keepdim=True)
    z_parallel = coeff * z_anchor
    z_orth = z - z_parallel

    z_def = z_parallel + lam * z_orth
    z_def = F.normalize(z_def, dim=-1)

    return z_def


@torch.no_grad()
def extract_noisy_features(
    image_encoder,
    x,
    num_samples=16,
    sigma=0.05,
):
    """
    x: [1, 3, H, W]
    return: [num_samples, D]
    """
    feats = []
    for _ in range(num_samples):
        noise = torch.randn_like(x) * sigma
        #z = image_encoder(x + noise)   # [1, D]
        z, text_features, logit_scale = image_encoder.forward_features(x + noise)
        z = F.normalize(z, dim=-1)
        feats.append(z.squeeze(0))
    return torch.stack(feats, dim=0),z   # [N, D]

def compute_covariance(Z):
    """
    Z: [N, D]
    """
    Z_centered = Z - Z.mean(dim=0, keepdim=True)
    cov = Z_centered.T @ Z_centered / (Z.shape[0] - 1)
    return cov, Z.mean(dim=0, keepdim=True)
def get_noise_invariant_basis(cov, keep_ratio=0.9):
    """
    keep_ratio: 保留最稳定的特征维度比例（小特征值）
    """
    eigvals, eigvecs = torch.linalg.eigh(cov)  # ascending
    D = eigvals.shape[0]
    k = int(D * keep_ratio)

    # 最小特征值对应的方向 = noise-invariant
    U_inv = eigvecs[:, :k]   # [D, k]
    return U_inv


@torch.no_grad()
def defend_feature(
    image_encoder,
    x,
    U_inv,
):
    """
    x: [1, 3, H, W]
    U_inv: [D, k]
    """
    #z = image_encoder(x)         # [1, D]
#     z, text_features, logit_scale = image_encoder.forward_features(x)
#     z = F.normalize(z, dim=-1)
#     z = z.squeeze(0)
    image_encoder = image_encoder.squeeze(0)
    # 投影到 noise-invariant 子空间
    z_def = U_inv @ (U_inv.T @ image_encoder)
    z_def = F.normalize(z_def, dim=-1)
    return z_def

def noise_response_slope_interval(features_fn, x, sigmas, device='cuda',
                                  slope_frac=0.25, max_interval_frac=0.3):
    """
    Compute safe sigma sub-interval based on maximal delta_norm slope.
    Sample-adaptive, training-free, no clean reference.

    Args:
        features_fn: function mapping input tensor to embedding
        x: input tensor [B, C, H, W]
        sigmas: 1D tensor/list of increasing noise std
        device: device
        slope_frac: fraction of max slope to define interval end
        max_interval_frac: maximum fraction of sigmas to include in safe interval

    Returns:
        z_list: embeddings at each sigma
        delta_norm: L2 distance from z0
        safe_sigma_indices: indices of safe sigma
        safe_interval: (start_idx, end_idx)
    """
    x = x.to(device)
    B = x.size(0)
    sigmas = torch.tensor(sigmas, device=device)
    z_list = []

    # Step 1: embeddings
    with torch.no_grad():
        z0,a,b = features_fn(x)
        z_list.append(z0)
        for sigma in sigmas:
            noise = torch.randn_like(x) * sigma
            z,a,b = features_fn(x + noise)
            z_list.append(z)

    # Step 2: compute delta_norm from z0
    delta_norm = [0.0]
    z0_flat = z0.view(B, -1)
    for z in z_list[1:]:
        z_flat = z.view(B, -1)
        delta = (z_flat - z0_flat).norm(dim=1)
        delta_norm.append(delta.mean().item())
    delta_norm = torch.tensor(delta_norm, device=device)

    # Step 3: compute discrete slope
    slope = delta_norm[1:] - delta_norm[:-1]  # Δ_norm difference

    # Step 4: find start_idx = max slope
    start_idx = torch.argmax(slope).item()

    # Step 5: find end_idx = last index where slope >= max_slope * slope_frac
    max_slope = slope[start_idx]
    end_idx = start_idx
    for i in range(start_idx, len(slope)):
        if slope[i] >= max_slope * slope_frac:
            end_idx = i
        else:
            break

    # Limit interval length
    max_len = int(len(sigmas) * max_interval_frac)
    end_idx = min(len(sigmas)-1, start_idx + max_len - 1)

    safe_sigma_indices = list(range(start_idx, end_idx + 1))
    safe_interval = (start_idx, end_idx)

    return z_list, delta_norm, safe_sigma_indices, safe_interval
def pca_defense(embeddings):
    """
    embeddings: [N, D]
    返回每个 embedding 与均值的 L2 距离
    """
    Z = embeddings.squeeze(1)
    mean_z = Z.mean(dim=0, keepdim=True)
    delta = (Z - mean_z).norm(dim=1)
    return delta


def noise_response_elbow(features_fn, x, image, sigmas, device='cuda'):
    """
    Noise-response elbow detection (NED)
    
    Args:
        features_fn: function, maps input tensor to embedding
        x: input tensor, shape [B, C, H, W]
        sigmas: list or 1D tensor of noise std, increasing
        device: device

    Returns:
        z_list: list of embeddings at each sigma
        delta_norm: list of L2 changes from z0
        elbow_idx: index of elbow point in sigmas
    """
    x = x.to(device)
    B = x.size(0)
    z_list = []
    confs = []
    feat_ori,a,b = features_fn(x)
    feat_flat = feat_ori.view(B, -1)
    # Step 1: compute embeddings for each sigma

    with torch.no_grad():
        z0,a,b = features_fn(image)
        #z_list.append(z0)
        for sigma in sigmas:
            noise = torch.randn_like(image) * sigma
            z,a,b = features_fn(image + noise)
            conf = b * z @ a.t()
            confidence_scores = F.softmax(conf, dim=1).max(dim=1).values
            confs.append(conf.cpu())            
            z_list.append(z)
    
    # Step 2: compute L2 difference from z0
    delta_norm = [0]  # z0 -> itself
    dis = [F.cosine_similarity(z0.view(B, -1), feat_flat, dim=1).mean().item()]

    z0_flat = z0.view(B, -1)
    for z in z_list[1:]:
        z_flat = z.view(B, -1)
        delta = (z_flat - z0_flat).norm(dim=1)
        
        #delta = F.cosine_similarity(z_flat, z0_flat, dim=1).mean().item()
        delta_norm.append(torch.tensor([delta]).cuda())
        
        distance = (z_flat - feat_flat).norm(dim=1)
        #distance = F.cosine_similarity(z_flat, feat_flat, dim=1)
        dis.append(distance.mean().item())

    delta_norm = torch.tensor(delta_norm)
    dis = torch.tensor(dis)
    
    # Step 3: normalize to [0,1]
#     delta_norm_norm = (delta_norm - delta_norm[0]) / (delta_norm[-1] - delta_norm[0] + 1e-8)
    delta_norm_norm = delta_norm
    
    # Step 4: compute discrete second-order diff (curvature)
    C = delta_norm_norm[2:] - 2 * delta_norm_norm[1:-1] + delta_norm_norm[:-2]
    # elbow = index of max curvature
    if len(C) == 0:
        elbow_idx = 0
    else:
        elbow_idx = torch.argmax(C).item() + 1  # +1 to align with delta_norm indices
    
    return z_list, delta_norm, elbow_idx, dis,confs

from matplotlib.lines import Line2D

def plot_confidence_for_example(logits_list, sigmas, target):
    """
    针对示例数据绘制置信度随噪声强度变化的曲线
    
    Args:
        logits_list: 包含多个logits的列表，每个logits形状为[1, num_classes]
        sigmas: 噪声强度张量
        target: 真实标签张量，形状为[1]（如tensor([0], device='cuda:0')）
    """
    # 1. 数据处理
    # 将sigmas转换为numpy数组
    if isinstance(sigmas, torch.Tensor):
        sigmas_np = sigmas.cpu().numpy()
    else:
        sigmas_np = np.array(sigmas)
    
    # 提取target值（从tensor([0])中提取0）
    target_value = target.item() if isinstance(target, torch.Tensor) else target
    
    # 2. 计算置信度和预测
    confidences = []
    predictions = []
    
    for logit in logits_list:
        # 确保logit在CPU上并去掉批次维度
        logit_cpu = logit.cpu().squeeze()  # 从[1, 47]变为[47]
        
        # 计算softmax概率
        probs = F.softmax(logit_cpu, dim=-1)
        
        # 获取置信度和预测类别
        confidence, prediction = probs.max(dim=-1)
        confidences.append(confidence.item())
        predictions.append(prediction.item())
    
    # 3. 计算置信度的平均值
    confidences_np = np.array(confidences)
    mean_confidence = np.mean(confidences_np)
    
    # 4. 判断是否正确分类
    is_correct = [pred == target_value for pred in predictions]
    num_correct = sum(is_correct)
    num_total = len(sigmas_np)
    if num_correct>0:
        # 5. 创建图形
        plt.figure(figsize=(12, 8))
        
        # 绘制置信度曲线（如果多个点）
        if len(sigmas_np) > 1:
            plt.plot(sigmas_np, confidences, 'b-', linewidth=2, alpha=0.7, label='Confidence')
        
        # 绘制每个点，用颜色和形状区分正确/错误
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']  # 多种标记样式
        
        for i, (sigma, conf) in enumerate(zip(sigmas_np, confidences)):
            # 选择颜色和标记
            color = 'green' if is_correct[i] else 'red'
            marker = markers[i % len(markers)]  # 循环使用标记样式
            
            # 绘制点
            plt.scatter(sigma, conf, color=color, marker=marker, s=150, 
                       edgecolors='black', linewidth=1.5, zorder=5,
                       label=f'σ={sigma:.3f}' if i == 0 else "")
            
            # 在点上添加正确/错误标记
            plt.annotate('✓' if is_correct[i] else '✗',
                        xy=(sigma, conf),
                        xytext=(0, 12),
                        textcoords='offset points',
                        ha='center',
                        fontsize=14,
                        fontweight='bold',
                        color='white' if is_correct[i] else 'black')
        
        # 6. 添加关键参考线
        # 0.5阈值线
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5, label='0.5 Threshold')
        
        # 置信度平均值线
        plt.axhline(y=mean_confidence, color='orange', linestyle='-.', alpha=0.8, 
                    linewidth=2, label=f'Mean Confidence ({mean_confidence:.3f})')
        
        # 随机猜测线（对于47分类，随机水平为1/47≈0.0213）
        random_level = 1.0 / 47
        plt.axhline(y=random_level, color='purple', linestyle=':', alpha=0.5, linewidth=1, label=f'Random ({random_level:.3f})')
        
        # 7. 设置图形属性
        plt.xlabel('Noise Standard Deviation (σ)', fontsize=14)
        plt.ylabel('Prediction Confidence', fontsize=14)
        plt.title(f'Confidence vs Noise Level\nTarget Class: {target_value} | Accuracy: {num_correct}/{num_total} ({num_correct/num_total:.1%})', 
                  fontsize=16, fontweight='bold', pad=20)
        
        # 设置坐标轴范围
        plt.xlim(min(sigmas_np) - 0.02, max(sigmas_np) + 0.02)
        plt.ylim(0, 1.05)
        
        # 添加网格
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # 8. 添加统计信息文本框
        stats_text = (f'Target Class: {target_value}\n'
                      f'Total Points: {num_total}\n'
                      f'Correct Predictions: {num_correct}\n'
                      f'Accuracy: {num_correct/num_total:.1%}\n'
                      f'Max Confidence: {max(confidences):.3f}\n'
                      f'Min Confidence: {min(confidences):.3f}\n'
                      f'Mean Confidence: {mean_confidence:.3f}')
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                 fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        # 9. 添加图例
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Correct', 
                   markerfacecolor='green', markersize=12, markeredgecolor='black', markeredgewidth=1),
            Line2D([0], [0], marker='o', color='w', label='Incorrect', 
                   markerfacecolor='red', markersize=12, markeredgecolor='black', markeredgewidth=1),
            Line2D([0], [0], color='blue', linewidth=2, label='Confidence Curve'),
            Line2D([0], [0], color='orange', linestyle='-.', linewidth=2, label=f'Mean ({mean_confidence:.3f})'),
            Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5, label='0.5 Threshold'),
            Line2D([0], [0], color='purple', linestyle=':', linewidth=1, label='Random Guess')
        ]
        
        plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.18), 
                   ncol=3, fontsize=10, framealpha=0.9)
        
        # 10. 调整布局并显示
        plt.tight_layout()
        plt.show()
    

def noise_response_elbow_batch(features_fn, x, image, sigmas, device='cuda'):
    """
    Noise-response elbow detection (NED) - Batch版本
    
    Args:
        features_fn: function, maps input tensor to embedding
        x: input tensor, shape [B, C, H, W]
        image: 原始图像，用于计算z0，shape [B, C, H, W]
        sigmas: list or 1D tensor of noise std, increasing
        device: device

    Returns:
        z_list: list of embeddings at each sigma, 每个元素shape [B, ...]
        delta_norm: list of L2 changes from z0, shape [len(sigmas), B]
        elbow_idx: index of elbow point in sigmas (标量)
        dis: 距离列表, shape [len(sigmas), B]
    """
    # 确保输入都在正确的设备上
    x = x.to(device)
    image = image.to(device)
    B = x.size(0)
    
    # Step 1: 计算原始特征
    with torch.no_grad():
        feat_ori, a, b = features_fn(x)  # feat_ori: [B, ...]
        z0, a0, b0 = features_fn(image)  # z0: [B, ...]
        
    # 展平特征以便计算距离
    feat_flat = feat_ori.view(B, -1)  # [B, D]
    z0_flat = z0.view(B, -1)          # [B, D]
    
    # Step 2: 计算每个噪声级别的特征
    z_list = []
    confs = []
    with torch.no_grad():
        for sigma in sigmas:
            # 为每个样本生成独立的噪声
            noise = torch.randn_like(image) * sigma
            z, a, b = features_fn(image + noise)  # z: [B, ...]
            conf = b * z @ a.t()
            confidence_scores = F.softmax(conf, dim=1).max(dim=1).values
            confs.append(conf)
            z_list.append(z)
    
    # Step 3: 计算L2距离和余弦相似度
    delta_norm = torch.zeros(len(z_list), B, device=device)  # [num_sigmas, B]
    dis = torch.zeros(len(z_list), B, device=device)         # [num_sigmas, B]
    
    # 第一个元素是sigma=0的情况
    delta_norm[0, :] = 0.0  # z0到自身的距离为0
    
    # 计算与feat_ori的距离
    z0_flat_expanded = z0_flat.unsqueeze(0)  # [1, B, D] 用于广播
    feat_flat_expanded = feat_flat.unsqueeze(0)  # [1, B, D]
    
    # 遍历每个噪声级别的特征
    for i, z in enumerate(z_list):
        z_flat = z.view(B, -1)  # [B, D]
        
        if i > 0:  # 跳过第一个（已经是0）
            # 计算与z0的L2距离
            delta = torch.norm(z_flat - z0_flat, dim=1)  # [B]
            delta_norm[i, :] = delta
        
        # 计算与feat_ori的距离
        distance = torch.norm(z_flat - feat_flat, dim=1)  # [B]
        dis[i, :] = distance
    
    # Step 4: 计算肘点（使用batch的平均值）
    # 取每个sigma下batch的平均距离
    delta_norm_mean = delta_norm.mean(dim=1)  # [num_sigmas]
    dis_mean = dis.mean(dim=1)                # [num_sigmas]
    
    # 归一化到[0,1]
    if delta_norm_mean[-1] - delta_norm_mean[0] > 1e-8:
        delta_norm_norm = (delta_norm_mean - delta_norm_mean[0]) / (delta_norm_mean[-1] - delta_norm_mean[0])
    else:
        delta_norm_norm = delta_norm_mean
    
    # Step 5: 计算曲率并找到肘点
    if len(delta_norm_norm) >= 3:
        # 计算二阶差分（曲率）
        C = delta_norm_norm[2:] - 2 * delta_norm_norm[1:-1] + delta_norm_norm[:-2]
        if len(C) > 0:
            elbow_idx = torch.argmax(C).item() + 1  # +1对齐索引
        else:
            elbow_idx = 0
    else:
        elbow_idx = 0
    
    return z_list, delta_norm, elbow_idx, dis,confs



def noise_response_converge_interval(features_fn, x, sigmas, device='cuda',
                                     smoothing_window=5, polyorder=2,
                                     delta_thresh=1e-3, consec=3, max_interval_frac=0.3):
    """
    Compute safe sigma sub-interval based on feature convergence (no clean sample required)
    
    Args:
        features_fn: function mapping input tensor to embedding
        x: input tensor [B, C, H, W]
        sigmas: list or 1D tensor of noise std, increasing
        device: device
        smoothing_window: window length for Savitzky-Golay smoothing
        polyorder: polynomial order for smoothing
        delta_thresh: threshold for considering embedding change as "converged"
        consec: number of consecutive steps below threshold to trigger convergence
        max_interval_frac: maximal fraction of sigmas to include in safe interval
        
    Returns:
        z_list: embeddings at each sigma
        delta_norm: L2 changes between consecutive sigmas
        safe_sigma_indices: indices of safe sigma
        safe_interval: (start_idx, end_idx)
    """
    x = x.to(device)
    B = x.size(0)
    z_list = []

    # Step 1: embeddings
    with torch.no_grad():
        z0,a,b = features_fn(x)
        z_list.append(z0)
        for sigma in sigmas:
            noise = torch.randn_like(x) * sigma
            z,a,b = features_fn(x + noise)
            z_list.append(z)

    # Step 2: compute Δ_norm between consecutive embeddings
    delta_norm = []
    z_prev = z0.view(B, -1)
    for z in z_list[1:]:
        z_flat = z.view(B, -1)
        delta = (z_flat - z_prev).norm(dim=1).mean().item()
        delta_norm.append(delta)
        z_prev = z_flat
    delta_norm = torch.tensor(delta_norm)

    # Step 3: smooth curve
    win_len = min(smoothing_window, len(delta_norm) if len(delta_norm)%2==1 else len(delta_norm)-1)
    delta_smooth = torch.tensor(savgol_filter(delta_norm.numpy(), window_length=win_len, polyorder=polyorder))

    # Step 4: find first convergence point
    start_idx = None
    for i in range(len(delta_smooth) - consec + 1):
        if torch.all(delta_smooth[i:i+consec] < delta_thresh):
            start_idx = i
            break
    if start_idx is None:
        # fallback: use first sigma
        start_idx = 0

    # Step 5: determine end of interval
    max_len = int(len(sigmas) * max_interval_frac)
    end_idx = min(len(sigmas)-1, start_idx + max_len - 1)

    safe_sigma_indices = list(range(start_idx, end_idx + 1))
    safe_interval = (start_idx, end_idx)

    return z_list, delta_norm, safe_sigma_indices, safe_interval
# def plot_noise_response(delta_norm, sigmas, elbow_idx):
#     plt.figure()
#     plt.plot([0]+sigmas, delta_norm, marker='o')
#     plt.axvline(x=sigmas[elbow_idx-1] if elbow_idx>0 else sigmas[0], color='r', linestyle='--', label='elbow')
#     plt.xlabel('Noise sigma')
#     plt.ylabel('L2 change from z0')
#     plt.title('Noise-response curve with elbow point')
#     plt.legend()
#     plt.show()
    


def plot_noise_response(delta_norm, dis, sigmas, elbow_idx):
    # ---- 强制转换为 numpy ----
    if torch.is_tensor(delta_norm):
        delta_norm = delta_norm.detach().cpu().numpy()
    if torch.is_tensor(dis):
        dis = dis.detach().cpu().numpy()

    #x = np.array([0] + sigmas)
    x = np.array(sigmas)

    fig, ax1 = plt.subplots()

    # ---- delta_norm ----
    ax1.plot(x, delta_norm, marker='o', linewidth=2, label='delta_norm')
    ax1.set_xlabel('Noise sigma')
    ax1.set_ylabel('L2 change from z0')

    # elbow
    elbow_x = sigmas[elbow_idx - 1] if elbow_idx > 0 else sigmas[0]
    ax1.axvline(x=elbow_x, linestyle='--', label='elbow')

    # ---- dis ----
    ax2 = ax1.twinx()
    ax2.plot(x, dis, marker='s', linestyle='--', label='dis')
    ax2.set_ylabel('dis')

    # legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.title('Noise-response curve')
    plt.tight_layout()
    plt.show()
    


def plot_noise_response_simple(delta_norm,dis,  sigmas, elbow_idx):
    # ---- 转换和斜率计算（同上）----
    if torch.is_tensor(delta_norm):
        delta_norm = delta_norm.detach().cpu().numpy()
    if torch.is_tensor(dis):
        dis = dis.detach().cpu().numpy()
    
    x = np.array(sigmas)
    
    # 计算斜率
    slopes = np.zeros_like(dis, dtype=float)
    for i in range(1, len(dis) - 1):
        slopes[i] = (dis[i+1] - dis[i-1]) / (x[i+1] - x[i-1])
        #slopes[i] = (dis[i+1] - dis[i]) / (x[i+1] - x[i])
    if len(dis) > 1:
        slopes[0] = (dis[1] - dis[0]) / (x[1] - x[0])
        slopes[-1] = (dis[-1] - dis[-2]) / (x[-1] - x[-2])
    if len(dis) == 1:
        slopes[0] = 0
        
        
# #     
    # 创建图形
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 绘制主曲线
    ax1.plot(x, delta_norm, marker='o', linewidth=2, label='dis')
    ax1.set_xlabel('Noise sigma')
    ax1.set_ylabel('L2 change from z0')
    
    # 肘点
    elbow_x = sigmas[elbow_idx - 1] if elbow_idx > 0 else sigmas[0]
    ax1.axvline(x=elbow_x, linestyle='--', label=f'elbow (σ={elbow_x:.3f})')
    
    # 绘制dis曲线
    ax2 = ax1.twinx()
    ax2.plot(x, dis, marker='s', linestyle='--', color='red', label='delta_norm')
    ax2.set_ylabel('Cosine Similarity')
    
    # 简洁的斜率标注：直接在点旁边显示数值
    for i, (xi, dis_i, slope_i) in enumerate(zip(x, dis, slopes)):
        # 选择标注位置：点上方
        offset = 0.02 * (dis.max() - dis.min())
        text_y = dis_i + offset if i % 2 == 0 else dis_i - offset
        
        ax2.text(
            xi, 
            text_y, 
            f'{slope_i:.2f}',  # 只保留2位小数
            fontsize=8, 
            color='darkred',
            ha='center',
            va='bottom' if i % 2 == 0 else 'top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7)
        )
    
    # 图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.title('Noise-response curve')
    plt.tight_layout()
    plt.show()
    
    plt.close(fig)
    
    return slopes


def adaptive_sigma_search_final(
    features_fn,
    x,
    sigmas,
    device='cuda',
    k=3,
    alpha=0.25,
    beta=0.8
):
    """
    Logically consistent final version of adaptive sigma search.

    Args:
        features_fn: function mapping input -> embedding
        x: input tensor [B, C, H, W]
        sigmas: increasing list/tensor of candidate sigmas
        device: cuda/cpu
        k: window size for slope smoothing
        alpha: stopping ratio for slope decay
        beta: fallback percentile threshold (0~1)

    Returns:
        chosen_sigma: selected sigma
        curve: list of (sigma, delta_norm)
    """

    x = x.to(device)

    with torch.no_grad():
        z0, _, _ = features_fn(x)
        z0_flat = z0.view(z0.size(0), -1)

    deltas = []
    slopes = []

    window = []
    max_avg_slope = 0.0

    prev_delta = None
    first_drop_idx = None

    # ============ Iterative Search ============
    for i, sigma in enumerate(sigmas):

        # ----- forward -----
        with torch.no_grad():
            noise = torch.randn_like(x) * sigma
            z, _, _ = features_fn(x + noise)
            z_flat = z.view(z.size(0), -1)

        # ----- compute delta -----
        delta = (z_flat - z0_flat).norm(dim=1).mean().item()
        deltas.append(delta)

        # ----- detect first drop -----
        if prev_delta is not None and delta < prev_delta:
            if first_drop_idx is None:
                first_drop_idx = i
        prev_delta = delta

        # ----- compute slope -----
        if i > 0:
            slope = abs(deltas[i] - deltas[i - 1])
        else:
            slope = 0.0

        slopes.append(slope)

        # ----- maintain window -----
        window.append(slope)
        if len(window) > k:
            window.pop(0)

        # ======= not enough points for window =======
        if len(window) < k:
            continue

        # ----- compute smoothed slope -----
        avg_slope = sum(window) / len(window)

        # ----- update max average slope -----
        max_avg_slope = max(max_avg_slope, avg_slope)

        # ======= main stopping rule =======
        if max_avg_slope > 0:
            if avg_slope < alpha * max_avg_slope:
                chosen_sigma = sigmas[i]
                curve = list(zip(sigmas[:i+1], deltas))
                return i, curve

    # ============ After full traversal ============

    # ----- Strategy 2: first drop point -----
    if first_drop_idx is not None:
        return first_drop_idx, list(zip(sigmas, deltas))

    # ----- Strategy 3: fallback 80% max distance -----
    max_delta = max(deltas)
    target = beta * max_delta

    for s, d in zip(sigmas, deltas):
        if d >= target:
            return s, list(zip(sigmas, deltas))

    # ----- Extreme case -----
    return -1, list(zip(sigmas, deltas))



# ==============================
# Part 1: Batch feature collection
# ==============================


def collect_all_embeddings(features_fn, x, sigmas, device='cuda'):
    x = x.to(device)
    B = x.size(0)
    N = len(sigmas)

    with torch.no_grad():
        z0, _, _ = features_fn(x)
        z0_flat = z0.view(B, -1)

        # [N, B, C, H, W]
        noises = torch.randn(N, *x.shape, device=device)

        for i, sigma in enumerate(sigmas):
            noises[i] *= float(sigma)

        # broadcast add
        x_rep = x.unsqueeze(0) + noises   # [N, B, C, H, W]

        # merge batch dims
        x_rep = x_rep.view(N * B, *x.shape[1:])

        z_all, _, _ = features_fn(x_rep)

        z_all = z_all.view(N, B, -1)

    return z0_flat, z_all



# ==============================
# Part 2: Curve computation
# ==============================

def compute_delta_curve(z0_flat, z_all):
    """
    计算 delta_norm 曲线

    Returns:
        deltas: list of float
    """
    
    deltas = []

    N = z_all.size(0)

    for i in range(N):
        delta = (z_all[i] - z0_flat).norm(dim=1).mean().item()
        deltas.append(delta)

    return deltas


def compute_delta_curve_per_sample(z0_flat, z_all):
    """
    z0_flat: [B, D]  50,512
    z_all: [N, B, D] 57,50,512

    返回: [B, N]
    """

    N, B, D = z_all.shape

    deltas = torch.zeros(B, N)

    for i in range(N):
        # 对每个样本单独算
        deltas[:, i] = (z_all[i] - z0_flat).norm(dim=1)

    return deltas


# ==============================
# Part 3: Search on curve
# ==============================

def search_on_curve(sigmas, deltas, k=3, alpha=0.25, beta=0.8):
    """
    只在曲线上搜索，不再需要任何forward

    Returns:
        chosen_idx: int
        curve: list of (sigma, delta)
    """

    slopes = []
    window = []

    max_avg_slope = 0.0
    first_drop_idx = None

    prev_delta = None

    N = len(deltas)

    for i in range(N):

        delta = deltas[i]

        # ----- detect first drop -----
        if prev_delta is not None and delta < prev_delta:
            if first_drop_idx is None:
                first_drop_idx = i
        prev_delta = delta

        # ----- compute slope -----
        if i > 0:
            slope = abs(deltas[i] - deltas[i - 1])
        else:
            slope = 0.0

        slopes.append(slope)

        # ----- maintain window -----
        window.append(slope)
        if len(window) > k:
            window.pop(0)

        # not enough for window
        if len(window) < k:
            continue

        avg_slope = sum(window) / len(window)

        max_avg_slope = max(max_avg_slope, avg_slope)

        # ----- main stopping -----
        if max_avg_slope > 0:
            if avg_slope < alpha * max_avg_slope:
                return i, list(zip(sigmas, deltas))

    # ===== fallback strategies =====

    # strategy 1: first drop
    if first_drop_idx is not None:
        return first_drop_idx, list(zip(sigmas, deltas))

    # strategy 2: 80% max distance
    max_delta = max(deltas)
    target = beta * max_delta

    for i, d in enumerate(deltas):
        if d >= target:
            return i, list(zip(sigmas, deltas))

    # strategy 3: extreme case
    return N - 1, list(zip(sigmas, deltas))


def search_on_curve_per_sample(sigmas, deltas_per_sample, k=3, alpha=0.25, beta=0.8):
    """
    对每个样本独立执行 search_on_curve 的逻辑

    Args:
        sigmas: list of length N
        deltas_per_sample: Tensor [B, N]

    Returns:
        chosen_indices: Tensor [B]
    """

    B, N = deltas_per_sample.shape
    chosen = torch.zeros(B, dtype=torch.long)

    for b in range(B):

        deltas = deltas_per_sample[b].tolist()

        slopes = []
        window = []

        max_avg_slope = 0.0
        first_drop_idx = None
        prev_delta = None

        chosen_idx = N - 1   # 默认最保守

        for i in range(N):

            delta = deltas[i]

            # ---- detect first drop ----
            if prev_delta is not None and delta < prev_delta:
                if first_drop_idx is None:
                    first_drop_idx = i
            prev_delta = delta

            # ---- compute slope ----
            if i > 0:
                slope = abs(deltas[i] - deltas[i - 1])
            else:
                slope = 0.0

            slopes.append(slope)

            # ---- maintain window ----
            window.append(slope)
            if len(window) > k:
                window.pop(0)

            # not enough for window
            if len(window) < k:
                continue

            avg_slope = sum(window) / len(window)

            max_avg_slope = max(max_avg_slope, avg_slope)

            # ---- main stopping ----
            if max_avg_slope > 0:
                if avg_slope < alpha * max_avg_slope:
                    chosen_idx = i
                    break

        # ===== fallback strategies =====

        if chosen_idx == N - 1:

            # strategy 1: first drop
            if first_drop_idx is not None:
                chosen_idx = first_drop_idx

            else:
                # strategy 2: 80% max distance
                max_delta = max(deltas)
                target = beta * max_delta

                for i, d in enumerate(deltas):
                    if d >= target:
                        chosen_idx = i
                        break

        chosen[b] = chosen_idx

    return chosen



def generate_sigmas(
    fine_start=0.0,
    fine_end=0.3,
    fine_step=0.005,
    coarse_end=0.6,
    coarse_step=0.01
):
    fine = torch.arange(fine_start, fine_end + 1e-8, fine_step)
    coarse = torch.arange(fine_end + coarse_step, coarse_end + 1e-8, coarse_step)

    sigmas = torch.cat([fine, coarse])
    return fine

def compute_simple_slopes(delta_norm, sigmas):
    """
    计算delta_norm在每个sigma值处的简单斜率
    
    计算方法：
    - 第一个点：slope[0] = (delta_norm[1] - delta_norm[0]) / (sigmas[1] - sigmas[0])
    - 中间点：slope[i] = (delta_norm[i+1] - delta_norm[i]) / (sigmas[i+1] - sigmas[i])
    - 最后一个点：slope[-1] = (delta_norm[-2] - delta_norm[-1]) / (sigmas[-2] - sigmas[-1])
    
    Args:
        delta_norm: 距离矩阵，形状为 [num_sigmas, batch_size] 或 [num_sigmas]
        sigmas: 噪声标准差列表，长度为 num_sigmas
        
    Returns:
        slopes: 斜率矩阵，形状与 delta_norm 相同
    """
    # 转换为torch张量
    if isinstance(delta_norm, (list, np.ndarray)):
        delta_norm = torch.tensor(delta_norm, dtype=torch.float32)
    if isinstance(sigmas, (list, np.ndarray)):
        sigmas = torch.tensor(sigmas, dtype=torch.float32)
    
    # 检查维度
    if delta_norm.dim() == 1:
        delta_norm = delta_norm.unsqueeze(1)  # 转换为 [num_sigmas, 1]
    
    num_sigmas, batch_size = delta_norm.shape
    
    # 初始化斜率矩阵
    slopes = torch.zeros_like(delta_norm)
    
    # 需要至少2个点才能计算斜率
    if num_sigmas < 2:
        return slopes
    
    # 计算sigma之间的差异（分母）
    sigma_diffs = sigmas[1:] - sigmas[:-1]
    
    # 第一个点：用第二个点减第一个点
    slopes[0] = (delta_norm[1] - delta_norm[0]) / sigma_diffs[0]
    
    # 中间点：用后一个点减前一个点
    for i in range(1, num_sigmas - 1):
        slopes[i] = (delta_norm[i+1] - delta_norm[i]) / sigma_diffs[i]
    
    # 最后一个点：用倒数第二个点减最后一个点
    slopes[-1] = (delta_norm[-2] - delta_norm[-1]) / sigma_diffs[-1]
    
    # 如果原始输入是1D，则返回1D
    if slopes.shape[1] == 1 and delta_norm.dim() == 1:
        slopes = slopes.squeeze(1)
    
    return slopes

import pandas as pd

def simple_slope_scatter(slope1_list, slope2_list,
                         slope1_name='Slope1', slope2_name='Slope2',
                         title='Slope Value Comparison',
                         save_path=None):
    """
    最简单的点图对比两个斜率列表
    
    Args:
        slope1_list: 第一个斜率列表（包含多个batch的max_values）
        slope2_list: 第二个斜率列表（包含多个batch的max_values）
        slope1_name: 第一个斜率的标签名
        slope2_name: 第二个斜率的标签名
        title: 图表标题
        save_path: 保存路径（可选）
    """
    # 合并数据
    if isinstance(slope1_list[0], torch.Tensor):
        slope1_data = torch.cat([tensor.cpu() for tensor in slope1_list]).numpy()
    else:
        slope1_data = np.concatenate([np.array(arr) for arr in slope1_list])
    
    if isinstance(slope2_list[0], torch.Tensor):
        slope2_data = torch.cat([tensor.cpu() for tensor in slope2_list]).numpy()
    else:
        slope2_data = np.concatenate([np.array(arr) for arr in slope2_list])
    
    # 创建图形
    plt.figure(figsize=(8, 6))
    
    # 设置随机种子（让抖动可重复）
    np.random.seed(42)
    
    # 为每组数据创建x坐标（添加随机抖动避免点重叠）
    x1 = 1 + np.random.normal(0, 0.05, len(slope1_data))
    x2 = 2 + np.random.normal(0, 0.05, len(slope2_data))
    
    # 绘制散点
    plt.scatter(x1, slope1_data, alpha=0.6, color='blue', label=slope1_name, s=30)
    plt.scatter(x2, slope2_data, alpha=0.6, color='red', label=slope2_name, s=30)
    
    # 计算平均值
    mean1 = np.mean(slope1_data)
    mean2 = np.mean(slope2_data)
    
    # 添加平均值线
    plt.axhline(mean1, color='blue', linestyle='--', alpha=0.7, label=f'{slope1_name} mean')
    plt.axhline(mean2, color='red', linestyle='--', alpha=0.7, label=f'{slope2_name} mean')
    
    # 设置x轴
    plt.xlim(0.5, 2.5)
    plt.xticks([1, 2], [slope1_name, slope2_name])
    
    # 添加标签和标题
    plt.xlabel('Group')
    plt.ylabel('Slope Value')
    plt.title(title)
    
    # 添加图例
    plt.legend()
    
    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"图表已保存到: {save_path}")
    else:
        plt.show()
    
    # 打印简单统计
    print(f"{slope1_name}: 均值={mean1:.4f}, 样本数={len(slope1_data)}")
    print(f"{slope2_name}: 均值={mean2:.4f}, 样本数={len(slope2_data)}")
    print(f"差值: {mean2-mean1:.4f}")
    
def normalize_instance_einsum(batch_tensor):
    """
    使用einsum进行逐样本逐通道归一化
    """
    # 计算每个样本每个通道的均值和标准差
    mean = batch_tensor.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
    std = batch_tensor.std(dim=(2, 3), keepdim=True)    # [B, C, 1, 1]
    
    return (batch_tensor - mean) / (std + 1e-8)
def find_peaks_for_batch(confidences_batch):
    """
    为batch中每个样本找到置信度序列的前两个峰值点索引（局部极大值）
    confidences_batch: [num_sigmas, batch_size]
    返回: [batch_size, 2] 每个样本的两个峰值点索引
    """
    batch_size = confidences_batch.shape[1]
    peaks = torch.zeros((batch_size, 2), dtype=torch.long)
    
    for i in range(batch_size):
        conf_seq = confidences_batch[:, i].cpu().numpy()
        
        # 寻找峰值点（局部极大值）
        peak_indices = []
        for j in range(1, len(conf_seq) - 1):
            if conf_seq[j] > conf_seq[j - 1] and conf_seq[j] > conf_seq[j + 1]:
                peak_indices.append(j)
        
        # 处理峰值点数量不足的情况
        if len(peak_indices) >= 2:
            peaks[i, 0] = peak_indices[0]  # 第一个峰值
            peaks[i, 1] = peak_indices[1]  # 第二个峰值
        elif len(peak_indices) == 1:
            peaks[i, 0] = peak_indices[0]
            peaks[i, 1] = min(peak_indices[0] + 1, len(conf_seq) - 1)
        else:
            # 如果没有峰值，取置信度最高的两个点
            top_two = np.argsort(conf_seq)[-2:]
            peaks[i, 0] = min(top_two)
            peaks[i, 1] = max(top_two)
    
    return peaks

def analyze_batch_categories(logits_list, sigmas, targets):
    """
    分析batch中样本的分类情况，返回50×4的矩阵
    
    Args:
        logits_list: 包含多个logits的列表，每个元素形状为[batch_size, num_classes]
        sigmas: 噪声强度列表或张量，长度与logits_list相同
        targets: 真实标签，形状为[batch_size]
    
    Returns:
        torch.Tensor: 形状为[batch_size, 4]的矩阵，每列对应一种情况
        列0: σ=0正确，后续存在正确点，a和b都正确 (类别1)
        列1: σ=0正确，后续存在正确点，a和b都错误 (类别2)
        列2: σ=0错误，后续存在正确点，a和b都正确 (类别3)
        列3: σ=0错误，后续存在正确点，a和b都错误 (类别4)
    """
    batch_size = logits_list[0].shape[0]
    num_sigmas = len(logits_list)
    
    # 转换为张量 [num_sigmas, batch_size, num_classes]
    logits_tensor = torch.stack(logits_list)
    
    # 计算每个样本在每个sigma处的预测结果
    preds = torch.argmax(logits_tensor, dim=-1)  # [num_sigmas, batch_size]
    
    # 计算正确性矩阵
    targets_expanded = targets.unsqueeze(0).expand(num_sigmas, -1)
    correct_matrix = (preds == targets_expanded)  # [num_sigmas, batch_size]
    
    # 计算置信度（用于找峰值点）
    confidences = F.softmax(logits_tensor, dim=-1).max(dim=-1).values  # [num_sigmas, batch_size]
    
    # 找到每个样本的两个峰值点索引
    peak_indices = find_peaks_for_batch(confidences)  # [batch_size, 2]
    
    # 初始化结果矩阵 [batch_size, 4]
    result_matrix = torch.zeros((batch_size, 4), dtype=torch.float32)
    
    # 检查每个样本在sigmas[1:]中是否存在正确点
    later_correct_exists = torch.any(correct_matrix[1:, :], dim=0)  # [batch_size]
    
    # 逐个样本分析
    for i in range(batch_size):
        # 如果后续没有正确点，跳过该样本
        if not later_correct_exists[i]:
            continue
            
        # 获取该样本的峰值点索引
        a_idx, b_idx  = peak_indices[i, 0].item(), peak_indices[i, 1].item()
        
        # 获取该样本的分类正确性
        sigma0_correct = correct_matrix[0, i].item()
        a_correct = correct_matrix[a_idx, i].item()
        b_correct = correct_matrix[b_idx, i].item()

        
        # 判断a和b是否都正确或都错误
        a_and_b_correct = a_correct and b_correct
        a_and_b_wrong = (not a_correct) and (not b_correct) 
        
        # 根据新规则分类
        if sigma0_correct:
            # σ=0正确
            if a_and_b_correct:
                result_matrix[i, 0] = 1  # 类别1
            elif a_and_b_wrong:
                result_matrix[i, 1] = 1  # 类别2
        else:
            # σ=0错误
            if a_and_b_correct:
                result_matrix[i, 2] = 1  # 类别3
            elif a_and_b_wrong:
                result_matrix[i, 3] = 1  # 类别4
    
    return result_matrix


def batch_adaptive_noise_features(
    x,                  # [B, C, H, W]
    model, normalized,             # 分类模型（输出 logits）
    threshold=0.15,
):
    device = x.device
    B = x.size(0)

    # ---------- 1. clean ----------
    z_clean, text_featuresclean, logit_scaleclean = model.forward_features(x)
    logits_clean = logit_scaleclean * (z_clean) @ text_featuresclean.t()
    prob_clean = F.softmax(logits_clean, dim=1)
    conf_clean, _ = prob_clean.max(dim=1)      # [B]

    # ---------- 2. sigma = 0.02 probe ----------
    noise_002 = torch.randn_like(x) * 0.02
    x_002 = x + noise_002

    #logits_002 = model(x_002)
    z_002, text_features002, logit_scale002 = model.forward_features(x_002)
    logits_002 = logit_scale002 * (z_002) @ text_features002.t()
    prob_002 = F.softmax(logits_002, dim=1)
    conf_002, _ = prob_002.max(dim=1)           # [B]


    # ---------- 3. confidence difference ----------
    delta = (conf_clean - conf_002).abs()      # [B]


    # mask: True → high-noise, False → mid-noise
    high_mask = delta > threshold               # [B]


    # ---------- 4. prepare noises ----------

    noise_004 = torch.randn_like(x) * 0.02# * normalized
    noise_006 = torch.randn_like(x) * 0.04# * normalized
    noise_008 = torch.randn_like(x) * 0.08# * normalized
    noise_010 = torch.randn_like(x) * 0.10 #* normalized

    x_004 = x + noise_004
    x_006 = x + noise_006
    x_008 = x + noise_008
    x_010 = x + noise_010

    # ---------- 5. extract all features ----------
#     f_004 = encoder(x_004)
#     f_006 = encoder(x_006)
#     f_008 = encoder(x_008)
#     f_010 = encoder(x_010)
    f_004, text_featuresclean, logit_scaleclean = model.forward_features(x_004)
    f_006, text_featuresclean, logit_scaleclean = model.forward_features(x_006)
    f_008, text_featuresclean, logit_scaleclean = model.forward_features(x_008)
    f_010, text_featuresclean, logit_scaleclean = model.forward_features(x_010)
    # ---------- 6. per-sample selection ----------
    # shape assumed: [B, D]
    high_mask = high_mask.view(B, *([1] * (f_004.dim() - 1)))

    # 对每个 sample：
    #   high_mask=True  → (0.08, 0.10)
    #   high_mask=False → (0.04, 0.06)
#     feat_1 = torch.where(high_mask, f_008, f_004)
#     feat_2 = torch.where(high_mask, f_010, f_006)
    
    feat_1 = torch.where(high_mask, f_010, f_006)
    feat_2 = torch.where(high_mask, f_008, f_004)

    # ---------- 7. return ----------
    return {
        "feat_1": feat_1,            # [B, D]
        "feat_2": feat_2,            # [B, D]
        "high_mask": high_mask.squeeze(),
        "delta_conf": delta,
    }


def batch_delta_confidence(
    images,           # [B, C, H, W]
    image_encoder,   # [K, D], normalized
    probe_sigma=0.05,
    num_probe=1
):
    """
    计算每个样本的 Δc（最大 logit 置信度下降量）
    返回: delta_c_batch, shape [B]
    """
    B = images.shape[0]
    
    # 原始 logits
    
    
    
    z_sou, text_features, logit_scale = image_encoder.forward_features(images.cuda())        
    logits_clean = logit_scale * (z_sou) @ text_features.t()
    logits_orig = F.softmax(logits_clean, dim=1)
    
    conf_clean, _ = logits_orig.max(dim=1) 
    
    # 用来累积 delta
    delta_accum = torch.zeros(B, device=images.device)
    
    for _ in range(num_probe):
        noise = torch.randn_like(images) * probe_sigma
        noisy_images = images + noise
        
        
        z_noisy, text_features, logit_scale = image_encoder.forward_features(noisy_images.cuda())        
        logits_noisy = logit_scale * (z_noisy) @ text_features.t()
        logits_noisy = F.softmax(logits_noisy, dim=-1)
        conf_noisy, _ = logits_noisy.max(dim=1)
    
        
        # gather 对应最大类别
        delta = torch.abs(
            conf_noisy - 
            conf_clean
        ) # [B]
        
        delta_accum += delta
    
    delta_c_batch = delta_accum / num_probe
    return delta_c_batch  # [B]
def decide_batch_sigma(
    E_batch,
    tau1=0.15,
    tau2=0.22,
    sigma_clean=0.00,
    sigma_weak=0.12,
    sigma_strong=0.2,
):
    if E_batch < tau1:
        return sigma_clean
    elif E_batch < tau2:
        return sigma_weak
    else:
        return sigma_strong
# def find_plateau_by_consecutive_small_increment(data, threshold=0.04, min_consecutive=2):
#     """
#     寻找平稳点：当连续出现至少两个点的增加量不超过阈值时，确认平稳开始点
#     
#     参数：
#     data: 数值列表
#     threshold: 增加量的阈值，默认0.03
#     min_consecutive: 最小连续点数，默认2
#     
#     返回：
#     平稳点开始的索引，如果没有找到则返回-1
#     """
#     n = len(data)
#     
#     # 检查数据长度是否足够
#     if n < min_consecutive + 1:
#         return -1
#     
#     consecutive_count = 0
#     start_index = -1
#     
#     for i in range(n - 1):
#         # 计算增加量（下一个值减当前值）
#         increment = data[i + 1] - data[i]
#         
#         if increment <= threshold:  # 只考虑非负的小增加量
#             if consecutive_count == 0:
#                 start_index = i  # 记录第一个符合条件的点
#             consecutive_count += 1
#             
#             # 检查是否达到最小连续点数
#             if consecutive_count >= min_consecutive:
#                 return start_index
#         else:
#             # 如果增加量超过阈值或为负，重置计数
#             consecutive_count = 0
#             start_index = -1
#     
#     # 如果没有找到符合条件的连续序列，检查最后几个点
#     # 有时候平稳点在末尾，但可能不满min_consecutive个点
#     if start_index != -1 and consecutive_count > 0:
#         # 如果已经到了末尾且没有重置，返回最后一个可能的平稳点
#         return start_index
#     
#     return -1



def batch_delta_confidence(
    images,           # [B, C, H, W]
    image_encoder,   # [K, D], normalized
    probe_sigma=0.05,
    num_probe=1
):
    """
    计算每个样本的 Δc（最大 logit 置信度下降量）
    返回: delta_c_batch, shape [B]
    """
    B = images.shape[0]
    
    # 原始 logits
    z_sou, text_features, logit_scale = image_encoder.forward_features(images.cuda())        
    logits_clean = logit_scale * (z_sou) @ text_features.t()
    logits_orig = F.softmax(logits_clean, dim=1)
    
    conf_clean, _ = logits_orig.max(dim=1) 
    
    # 用来累积 delta
    delta_accum = torch.zeros(B, device=images.device)
    
    for _ in range(num_probe):
        noise = torch.randn_like(images) * probe_sigma
        noisy_images = images + noise
        
        z_noisy, text_features, logit_scale = image_encoder.forward_features(noisy_images.cuda())        
        logits_noisy1 = logit_scale * (z_noisy) @ text_features.t()
        logits_noisy = F.softmax(logits_noisy1, dim=-1)
        conf_noisy, _ = logits_noisy.max(dim=1)
        
        # gather 对应最大类别
        delta = torch.abs(conf_noisy - conf_clean) # [B]
        delta_accum += delta
    
    delta_c_batch = delta_accum / num_probe
    return delta_c_batch, logits_noisy1  # [B]

def find_plateau_by_consecutive_small_increment(data, threshold=0.04, min_consecutive=2):
    """
    寻找平稳点：当连续出现至少两个点的增加量不超过阈值时，确认平稳开始点
    
    参数：
    data: 数值列表
    threshold: 增加量的阈值，默认0.03
    min_consecutive: 最小连续点数，默认2
    
    返回：
    平稳点开始的索引，如果没有找到则返回-1
    """
    n = len(data)
    
    # 检查数据长度是否足够
    if n < min_consecutive + 1:
        return -1
    
    consecutive_count = 0
    start_index = -1
    
    for i in range(n - 1):
        # 计算增加量（下一个值减当前值）
        increment = data[i + 1] - data[i]
        
        if increment <= threshold:  # 只考虑非负的小增加量
            if consecutive_count == 0:
                start_index = i  # 记录第一个符合条件的点
            consecutive_count += 1
            
            # 检查是否达到最小连续点数
            if consecutive_count >= min_consecutive:
                return start_index
        else:
            # 如果增加量超过阈值或为负，重置计数
            consecutive_count = 0
            start_index = -1
    
    # 如果没有找到符合条件的连续序列，检查最后几个点
    # 有时候平稳点在末尾，但可能不满min_consecutive个点
    if start_index != -1 and consecutive_count > 0:
        # 如果已经到了末尾且没有重置，返回最后一个可能的平稳点
        return start_index
    
    return -1

class EarlyStopDeltaCalculator:
    """
    早期停止的delta计算器
    特点：按顺序计算不同噪声强度，一旦检测到平稳点就停止
    """
    
    def __init__(
        self,
        image_encoder,
        sigma_list: List[float] = None,
        num_probe: int = 1,
        plateau_threshold: float = 0.04,
        min_consecutive: int = 2
    ):
        """
        初始化
        
        Args:
            image_encoder: 图像编码器模型
            sigma_list: 噪声强度列表，默认为[0.02, 0.04, 0.06]
            num_probe: 每个噪声强度的采样次数
            plateau_threshold: 平稳点检测阈值
            min_consecutive: 最小连续平稳点数
        """
        self.image_encoder = image_encoder
        self.num_probe = num_probe
        self.plateau_threshold = plateau_threshold
        self.min_consecutive = min_consecutive
        
        # 默认的sigma列表
        if sigma_list is None:
            self.sigma_list = [0.02, 0.04, 0.06]
        else:
            self.sigma_list = sigma_list
    
    def compute_with_early_stop(
        self,
        images: torch.Tensor,
        return_all: bool = False
    ) -> tuple:
        """
        计算delta，一旦检测到平稳点就停止
        
        Args:
            images: 输入图像 [B, C, H, W]
            return_all: 是否返回所有计算过的delta均值
            
        Returns:
            如果return_all为False: (plateau_idx, final_delta_mean)
            如果return_all为True: (plateau_idx, final_delta_mean, all_delta_means, all_sigma_used)
        """
        images = images.cuda()
        delta_means = []
        sigma_used = []
        
        # 按顺序计算每个sigma
        for sigma in self.sigma_list:
            # 计算当前sigma的delta
            delta = batch_delta_confidence(
                images, 
                self.image_encoder, 
                probe_sigma=sigma, 
                num_probe=self.num_probe
            )
            
            delta_mean = delta.mean().item()
            delta_means.append(delta_mean)
            sigma_used.append(sigma)
            
            # 检查是否达到平稳点（至少需要3个点才能检测）
            if len(delta_means) >= self.min_consecutive + 1:
                plateau_idx = find_plateau_by_consecutive_small_increment(
                    delta_means,
                    threshold=self.plateau_threshold,
                    min_consecutive=self.min_consecutive
                )
                
                # 如果找到平稳点，停止计算
                if plateau_idx != -1:
                    final_delta_mean = delta_means[plateau_idx]
                    
                    if return_all:
                        return plateau_idx, final_delta_mean, delta_means, sigma_used
                    else:
                        return plateau_idx, final_delta_mean
        
        # 如果遍历完所有sigma都没有找到平稳点
        final_delta_mean = delta_means[-1] if delta_means else 0.0
        
        if return_all:
            return -1, final_delta_mean, delta_means, sigma_used
        else:
            return -1, final_delta_mean
    
    def compute_full_sequence(
        self,
        images: torch.Tensor
    ) -> dict:
        """
        计算完整的delta序列（不提前停止），用于分析和可视化
        """
        images = images.cuda()
        delta_means = []
        delta_values = []  # 存储每个sigma的完整delta tensor
        
        for sigma in self.sigma_list:
            delta = batch_delta_confidence(
                images, 
                self.image_encoder, 
                probe_sigma=sigma, 
                num_probe=self.num_probe
            )
            
            delta_mean = delta.mean().item()
            delta_means.append(delta_mean)
            delta_values.append(delta)
        
        # 检测平稳点
        plateau_idx = find_plateau_by_consecutive_small_increment(
            delta_means,
            threshold=self.plateau_threshold,
            min_consecutive=self.min_consecutive
        )
        
        return {
            'delta_means': delta_means,
            'delta_values': delta_values,  # 每个sigma对应的完整delta tensor
            'sigma_list': self.sigma_list,
            'plateau_idx': plateau_idx,
            'plateau_sigma': self.sigma_list[plateau_idx] if plateau_idx != -1 else None,
            'plateau_delta': delta_means[plateau_idx] if plateau_idx != -1 else None
        }

# 保持你的原始调用方式完全兼容
def compute_delta_with_plateau_detection(
    images: torch.Tensor,
    model,
    sigma_list: List[float] = None,
    num_probe: int = 1,
    return_details: bool = False
):
    """
    保持与你原始调用方式兼容的函数
    
    Args:
        images: 输入图像
        model: 图像编码器
        sigma_list: 噪声强度列表，默认为[0.02, 0.04, 0.06]
        num_probe: 每个噪声强度的采样次数
        return_details: 是否返回详细信息
        
    Returns:
        与原始调用方式相同的结果
    """
    if sigma_list is None:
        sigma_list = [0.02, 0.04, 0.06]
    
    # 创建计算器
    calculator = EarlyStopDeltaCalculator(
        image_encoder=model,
        sigma_list=sigma_list,
        num_probe=num_probe,
        plateau_threshold=0.03,  # 保持你的默认值
        min_consecutive=2        # 保持你的默认值
    )
    
    # 使用早期停止计算
    plateau_idx, final_delta_mean, all_delta_means, all_sigma_used = calculator.compute_with_early_stop(
        images, 
        return_all=True
    )
    
    if return_details:
        return {
            'datasclns': plateau_idx,  # 保持你的变量名
            'delta_mean': final_delta_mean,
            'all_deltas': all_delta_means,
            'sigma_used': all_sigma_used,
            'stopped_early': plateau_idx != -1 and len(all_sigma_used) < len(sigma_list)
        }
    else:
        return plateau_idx, final_delta_mean
    
    
    
def analyze_tensor_distribution_simple(t1, t2, num_bins=3):
    """
    分析两个张量列表的分布
    t1, t2: 张量列表，如 [tensor(0.1500, device='cuda:0'), ...]
    num_bins: 区间数量
    """
    # 将GPU上的张量列表转换为CPU上的numpy数组
    def to_numpy(tensor_list):
        return torch.stack(tensor_list).cpu().numpy()
    
    # 转换数据
    t1_np = to_numpy(t1)
    t2_np = to_numpy(t2)
    
    # 计算最大值
    max_value = max(t1_np.max(), t2_np.max())
    
    # 创建区间
    bins = np.linspace(0, max_value, num_bins + 1)
    
    # 统计并打印结果
    for name, data in [('Tensor 1', t1_np), ('Tensor 2', t2_np)]:
        counts, _ = np.histogram(data, bins=bins)
        percents = counts / counts.sum() * 100
        
        print(f'{name} 区间统计:')
        for i in range(len(bins)-1):
            print(f'  [{bins[i]:.4f}, {bins[i+1]:.4f}): {percents[i]:.2f}% ({counts[i]}个)')
        print()
        
def plot_scatter_for_tensor_lists(t1, t2):
    """
    为张量列表绘制散点图
    
    Args:
        t1: 张量列表，如 [tensor(0.1500, device='cuda:0'), ...]
        t2: 张量列表，如 [tensor(0.1200, device='cuda:0'), ...]
    """
    # 将GPU上的张量转换为CPU上的Python列表
    def tensor_list_to_values(tensor_list):
        return [t.cpu().item() for t in tensor_list]
    
    # 转换为数值列表
    t1_values = tensor_list_to_values(t1)
    t2_values = tensor_list_to_values(t2)
    
    # 创建图形
    plt.figure(figsize=(10, 5))
    
    # 绘制散点图
    plt.scatter(range(len(t1_values)), t1_values, label='Tensor 1', alpha=0.7, color='blue', s=100)
    plt.scatter(range(len(t2_values)), t2_values, label='Tensor 2', alpha=0.7, color='red', s=100)
    
    # 添加标签和标题
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Scatter Plot of Two Tensors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('/media/cqu/D/FXV/R-TPT-main6/test.png', dpi=300, bbox_inches='tight')
    
    # 显示图形
    plt.tight_layout()
    plt.close()
    
    
    
def compute(x1, x2):
    """
    根据输入x返回结果：
    如果 x < 0.02，返回0；
    否则，找到与x绝对值最接近的a值，并返回对应的b值。
    """
    a_list = [0, 0.85, 1.565, 2.365, 2.97, 3.455, 3.88, 4.295]
    b_list = [0.1, 0.12, 0.16, 0.18, 0.18, 0.2, 0.2, 0.22]
    
    if x1 < 0.8 * 0.21945 and x2 == 0:
        return 0.03
    else:
        # 找到最接近的a的索引
        closest_index = min(range(len(a_list)), key=lambda i: abs(a_list[i] - x2))
        return b_list[closest_index]
def generate_neighbors(x, k):
    """
    如果 x == 0，返回 0。
    否则，返回以 x 为中心、步长 0.01 的左右各 k 个邻近值（不包括 x 本身）。
    例如 x=0.1, k=4 → [0.06, 0.07, 0.08, 0.09, 0.11, 0.12, 0.13, 0.14]。
    结果保留两位小数。
    """
    if x == 0:
        return 0

    step = 0.005
    left = [round(x - i * step, 3) for i in range(1, k + 1)]
    right = [round(x + i * step, 3) for i in range(1, k + 1)]
    
    # 合并并排序
    neighbors = sorted(left + right)
    return neighbors


def combine_features_by_average(z_sou_list):
    """
    将多个批处理特征按样本位置取平均。
    
    参数:
        z_sou_list (list of torch.Tensor): 特征列表，每个元素形状为 (batch_size, feature_dim)
    
    返回:
        combined_feature (torch.Tensor): 平均后的特征，形状为 (batch_size, feature_dim)
    """
    assert len(z_sou_list) > 0, "特征列表不能为空"
    # 将所有特征堆叠起来，形状 (num_sources, batch_size, feature_dim)
    features = torch.stack(z_sou_list)
    # 在源维度上取平均
    combined = features.mean(dim=0)
    return combined
def combine_features_by_entropy(z_sou_list, output_list, eps=1e-10):
    """
    根据熵值加权组合多个批处理特征（每个样本独立加权）。

    参数:
        z_sou_list (list of torch.Tensor): 特征列表，每个元素形状为 (batch_size, feature_dim)
        output_list (list of torch.Tensor): 对应的logits列表，每个元素形状为 (batch_size, num_classes)
        eps (float): 计算熵时防止log(0)的小常数

    返回:
        combined_feature (torch.Tensor): 组合后的特征，形状为 (batch_size, feature_dim)
        weights (torch.Tensor): 归一化后的权重，形状为 (batch_size, len(z_sou_list))
    """
    assert len(z_sou_list) == len(output_list), "两个列表长度必须相等"
    batch_size = z_sou_list[0].size(0)
    num_sources = len(z_sou_list)

    # 1. 计算每个源、每个样本的熵
    entropy_per_source = []  # 每个元素形状 (batch_size,)
    for logits in output_list:
        probs = F.softmax(logits, dim=-1)               # (batch_size, num_classes)
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)  # (batch_size,)
        entropy_per_source.append(entropy)

    # 2. 堆叠得到熵矩阵 (num_sources, batch_size)
    entropy_matrix = torch.stack(entropy_per_source)

    # 3. 计算权重：熵越小权重越大，使用倒数后沿源维度归一化
    inv_entropy = 1.0 / (entropy_matrix + eps)          # (num_sources, batch_size)
    weights = inv_entropy / inv_entropy.sum(dim=0, keepdim=True)  # (num_sources, batch_size)

    # 4. 转换为 (batch_size, num_sources) 方便后续加权
    weights = weights.transpose(0, 1)                    # (batch_size, num_sources)

    # 5. 构建特征张量 (num_sources, batch_size, feature_dim)
    features = torch.stack(z_sou_list)                   # (num_sources, batch_size, feature_dim)

    # 6. 加权求和（对每个样本在源维度上加权）
    # 扩展 weights 维度以匹配 features 进行广播
    weights_expanded = weights.transpose(0, 1).unsqueeze(-1)  # (num_sources, batch_size, 1)
    weighted_features = weights_expanded * features            # (num_sources, batch_size, feature_dim)
    combined = weighted_features.sum(dim=0)                    # (batch_size, feature_dim)

    return combined

class FeatureSpaceAnalyzer:
    def __init__(self, temperature=0.01):
        """
        temperature: CLIP 相似度计算时的温度系数（或者倒数 logit_scale）。
                     CLIP 官方的 logit_scale 通常接近 100 (即 temperature ≈ 0.01)
        """
        self.temp = temperature

    def analyze(self, V_clean, V_adv, T, V_raw_clean=None, V_raw_adv=None):
        """
        提取对抗攻击在多模态特征空间中的四个核心统计指标。
        参数：
        - V_clean: 干净图像的归一化特征 [50, 512]
        - V_adv:   对抗图像的归一化特征 [50, 512]
        - T:       文本锚点的归一化特征 [47, 512]
        - V_raw_clean: (可选) 归一化前的干净图像特征 [50, 512]
        - V_raw_adv:   (可选) 归一化前的对抗图像特征 [50, 512]
        """
        results = {}
        
        # ==========================================
        # 1. 秩坍塌度 (Rank Collapse / Anisotropy)
        # 物理意义：特征是否失去了多样性，挤成了一团？
        # ==========================================
        # 对特征进行中心化
        V_clean_centered = V_clean - V_clean.mean(dim=0, keepdim=True)
        V_adv_centered = V_adv - V_adv.mean(dim=0, keepdim=True)
        
        # SVD 分解获取奇异值
        _, S_clean, _ = torch.linalg.svd(V_clean_centered)
        _, S_adv, _ = torch.linalg.svd(V_adv_centered)
        
        # 计算 Top-1 奇异值能量占比 (越大说明越坍塌到一个单一方向)
        results['Top1_Singular_Ratio_Clean'] = (S_clean[0] / S_clean.sum()).item()
        results['Top1_Singular_Ratio_Adv'] = (S_adv[0] / S_adv.sum()).item()

        # ==========================================
        # 2. 跨模态相似度信息熵 (Similarity Entropy)
        # 物理意义：模型是被明确误导了(低熵)，还是彻底发懵了(高熵)？
        # ==========================================
        # 计算与 47 个文本类的相似度 (注意除以温度系数)
        logits_adv = (V_adv @ T.T) / self.temp
        
        # 转化为概率分布
        probs_adv = F.softmax(logits_adv, dim=-1)
        
        # 计算信息熵
        entropy_adv = -torch.sum(probs_adv * torch.log(probs_adv + 1e-8), dim=-1).mean()
        results['Similarity_Entropy_Adv'] = entropy_adv.item()

        # ==========================================
        # 3. 原始特征 L2 范数爆炸 (Pre-norm L2 Norm)
        # 物理意义：大扰动是否在网络深层引发了极端激活/能量溢出？
        # ==========================================
        if V_raw_clean is not None and V_raw_adv is not None:
            norm_clean = torch.norm(V_raw_clean, p=2, dim=-1).mean()
            norm_adv = torch.norm(V_raw_adv, p=2, dim=-1).mean()
            results['Raw_Feature_Norm_Clean'] = norm_clean.item()
            results['Raw_Feature_Norm_Adv'] = norm_adv.item()

        # ==========================================
        # 4. 流形正交偏离度 (Orthogonal Deviation)
        # 物理意义：扰动是顺着原意图滑行(平行)，还是直接刺穿了流形(正交)？
        # ==========================================
        # 扰动在 512 维特征空间引起的纯位移向量
        delta_V = V_adv - V_clean 
        
        # 计算位移向量与原始特征的余弦相似度
        # 越接近 0，说明扰动方向与特征方向越“垂直(正交)”，刺穿效应越强
        cos_deviation = F.cosine_similarity(delta_V, V_clean, dim=-1).mean()
        results['Orthogonal_Deviation'] = cos_deviation.item()

        return results
    
def test_time_adapt_eval(classnames, sigma_dict,mu_dict,  val_loader, model, model_state, optimizer, optim_state, scaler, args, data_transform):
    


    sigma_img_dict = {}
    
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    tpt1 = AverageMeter('TTAAcc@1', ':6.2f', Summary.AVERAGE)
    top1002 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    tpt1002 = AverageMeter('TTAAcc@1', ':6.2f', Summary.AVERAGE)
    top1004 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    tpt1004 = AverageMeter('TTAAcc@1', ':6.2f', Summary.AVERAGE)
    top1006 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    tpt1006 = AverageMeter('TTAAcc@1', ':6.2f', Summary.AVERAGE)
    top1008 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    tpt1008 = AverageMeter('TTAAcc@1', ':6.2f', Summary.AVERAGE)
    top101 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    tpt101 = AverageMeter('TTAAcc@1', ':6.2f', Summary.AVERAGE)
    top1012 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    tpt1012 = AverageMeter('TTAAcc@1', ':6.2f', Summary.AVERAGE)
    top1014 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    tpt1014 = AverageMeter('TTAAcc@1', ':6.2f', Summary.AVERAGE)
    top1016 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    tpt1016 = AverageMeter('TTAAcc@1', ':6.2f', Summary.AVERAGE)
    top1018 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    tpt1018 = AverageMeter('TTAAcc@1', ':6.2f', Summary.AVERAGE)
    top102 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    tpt102 = AverageMeter('TTAAcc@1', ':6.2f', Summary.AVERAGE)
    top1022 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    tpt1022 = AverageMeter('TTAAcc@1', ':6.2f', Summary.AVERAGE)
    top1024 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    tpt1024 = AverageMeter('TTAAcc@1', ':6.2f', Summary.AVERAGE)
    top1026 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    tpt1026 = AverageMeter('TTAAcc@1', ':6.2f', Summary.AVERAGE)
    top1028 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    tpt1028 = AverageMeter('TTAAcc@1', ':6.2f', Summary.AVERAGE)
    top103 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    tpt103 = AverageMeter('TTAAcc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, tpt1],
        prefix='Test: ')
    
    progress2 = ProgressMeter(
        len(val_loader),
        [batch_time, top1002,tpt1002],
        prefix='Test: ')
    progress4 = ProgressMeter(
        len(val_loader),
        [batch_time, top1004,tpt1004],
        prefix='Test: ')
    progress6 = ProgressMeter(
        len(val_loader),
        [batch_time, top1006,tpt1006],
        prefix='Test: ')
    progress8 = ProgressMeter(
        len(val_loader),
        [batch_time, top1008,tpt1008],
        prefix='Test: ')
    progress10 = ProgressMeter(
        len(val_loader),
        [batch_time, top101,tpt101],
        prefix='Test: ')
    progress12 = ProgressMeter(
        len(val_loader),
        [batch_time, top1012,tpt1012],
        prefix='Test: ')
    progress14 = ProgressMeter(
        len(val_loader),
        [batch_time, top1014,tpt1014],
        prefix='Test: ')
    progress16 = ProgressMeter(
        len(val_loader),
        [batch_time, top1016,tpt1016],
        prefix='Test: ')
    progress18 = ProgressMeter(
        len(val_loader),
        [batch_time, top1018,tpt1018],
        prefix='Test: ')
    progress20 = ProgressMeter(
        len(val_loader),
        [batch_time, top102,tpt102],
        prefix='Test: ')
    progress22 = ProgressMeter(
        len(val_loader),
        [batch_time, top1022,tpt1022],
        prefix='Test: ')
    progress24 = ProgressMeter(
        len(val_loader),
        [batch_time, top1024,tpt1024],
        prefix='Test: ')
    progress26 = ProgressMeter(
        len(val_loader),
        [batch_time, top1026,tpt1026],
        prefix='Test: ')
    progress28 = ProgressMeter(
        len(val_loader),
        [batch_time, top1028,tpt1028],
        prefix='Test: ')
    progress30 = ProgressMeter(
        len(val_loader),
        [batch_time, top103,tpt103],
        prefix='Test: ')

    # reset model and switch to evaluate mode
    model.eval()
    model.cuda()

    if args.eps > 0.0:
        assert args.steps > 0
        atk = torchattacks.PGD(model, eps=4/255, alpha=1/255, steps=10)
        #atk = torchattacks.CW(model)
        #atk2 = torchattacks.PGD(model, eps=args.eps/255, alpha=args.alpha/255, steps=3)
        #atk.set_mode_targeted_random(True)

        
    end = time.time()
    
    all_logits = []
    all_labels = []
    all_logits_tta = []
    
    cos = []
    cos2 = []
    ents = []
    confs = []
    
    distance = []
    adv_vals, rand_vals = [], []
    features_per_class = defaultdict(list)
    
    slope1 = []
    slope2 = []
    
    class1 = []
    class2 = []    
    class3 = []
    class4 = []
    start_time = time.time()
    deltaadvs = []
    deltaclns = []
            
    datasvalue = []
    datasclnvalue = []
    
    datasvalue2 = []
    datasclnvalue2 = []
    
    datasx = []
    datasxx = []
    datasx1 = []
    datasx2 = []
    datasx3 = []
    datasx4 = []
    datasx5 = []
    datasx6 = []
    datasx7 = []
    datasx8 = []
    datasx9 = []
    datasx10 = []
    datasx11 = []
    datasx12 = []
    datasx13 = []
    datasx14 = []
    datasx15 = []

    for i, (images, target) in enumerate(val_loader):

        assert args.gpu is not None
        target = target.cuda(args.gpu, non_blocking=True)
        
        datas = []
        datascln = []
        datas2 = []
        datascln2 = []



        if args.eps > 0.0:
#             image = images[0].cuda(args.gpu, non_blocking=True)
#             img_ori = image
#             adv_image = atk(image, target)        
#             img_adv = transforms.ToPILImage()(adv_image.squeeze(0))
#             images = data_transform(img_adv)
#             images = [_.unsqueeze(0) for _ in images]






            adv_image = atk(images, target)
            advnoise = adv_image.cuda() - images.cuda()
            #adv_image = images.cuda()
            
            
            
            
            
            
            
            
            
            
            
            
            
            
#             idx = torch.randint(0, len(classnames), (images.size(0),)).cuda()
#             
#             adv_image2, grad = atk2(images,idx)
#             adv_image3, grad2 = atk2(adv_image,idx)
#             
#             normalized = (grad - grad.view(images.size(0), 3, -1).min(dim=2, keepdim=True)[0].view(images.size(0), 3, 1, 1)) / \
#              (grad.view(images.size(0), 3, -1).max(dim=2, keepdim=True)[0].view(images.size(0), 3, 1, 1) - 
#               grad.view(images.size(0), 3, -1).min(dim=2, keepdim=True)[0].view(images.size(0), 3, 1, 1) + 1e-8)
# 
#             normalized2 = (grad2 - grad2.view(images.size(0), 3, -1).min(dim=2, keepdim=True)[0].view(images.size(0), 3, 1, 1)) / \
#              (grad2.view(images.size(0), 3, -1).max(dim=2, keepdim=True)[0].view(images.size(0), 3, 1, 1) - 
#               grad2.view(images.size(0), 3, -1).min(dim=2, keepdim=True)[0].view(images.size(0), 3, 1, 1) + 1e-8)
# #             print(F.cosine_similarity(normalized, normalized2, dim=1, eps=1e-8).mean())
# 
#             B, C, H, W = normalized.shape
# 
#             # 展平空间维度
#             normalized_flat = normalized.view(B, C, -1)          # [50, 3, 224*224]
# 
#             # 每个 (B,C) 需要保留的元素数
#             k = int(0.9 * H * W)
# 
#             # 找 top-k 的阈值
#             topk_vals, _ = torch.topk(normalized_flat, k, dim=-1)
#             threshold = topk_vals[..., -1].unsqueeze(-1)  # [B, C, 1]
# 
#             # 构造 mask
#             mask = normalized_flat >= threshold
# 
#             # 应用 mask
#             normalized_flat = normalized_flat * mask
# 
#             # 还原形状
#             normalized_filtered = mask.view(B, C, H, W)
#             
#             
#             
# 
#             normalized2_flat = normalized2.view(B, C, -1)          # [50, 3, 224*224]
# 
#             # 每个 (B,C) 需要保留的元素数
#             k = int(0.9 * H * W)
# 
#             # 找 top-k 的阈值
#             topk_vals2, _ = torch.topk(normalized2_flat, k, dim=-1)
#             threshold2 = topk_vals2[..., -1].unsqueeze(-1)  # [B, C, 1]
# 
#             # 构造 mask
#             mask2 = normalized2_flat >= threshold2
# 
#             # 应用 mask
#             normalized2_flat = normalized2_flat * mask2
# 
#             # 还原形状
#             normalized_filtered2 = mask2.view(B, C, H, W)
            

            def model_fn(x):
                z_sou, text_features, logit_scale = model.forward_features(x)
                output_a = logit_scale * (z_sou) @ text_features.t()
                return output_a.to(torch.float32)    
#             
#             adversary = AutoAttack(model_fn, norm='Linf', eps=1/255, version='standard')
#             adv_image = adversary.run_standard_evaluation(images.cuda(), target.cuda(), bs=100)            
#             

#         if isinstance(images, list):
#             for k in range(len(images)):
#                 images[k] = images[k].cuda(args.gpu, non_blocking=True)
#             image = images[0]
#         else:
#             if len(images.size()) > 4:
#                 # when using ImageNet Sampler as the dataset
#                 assert images.size()[0] == 1
#                 images = images.squeeze(0)
#             images = images.cuda(args.gpu, non_blocking=True)
#             image = images
#         
#         images = torch.cat(images, dim=0)
# 
#         # reset model
#         with torch.no_grad():
#             model.reset()
#         optimizer.load_state_dict(optim_state)
#         noise_image1, noise = add_gaussian_noise(images, noise_std=0.0324)
#         noise_image_adv1, noise = add_gaussian_noise(adv_image, noise_std=0.0324)
#         
#         noise_image2, noise = add_gaussian_noise(images, noise_std=0.0324)
#         noise_image_adv2, noise = add_gaussian_noise(adv_image, noise_std=0.0324)
#         
#         noise_image3, noise = add_gaussian_noise(images, noise_std=0.0324)
#         noise_image_adv3, noise = add_gaussian_noise(adv_image, noise_std=0.0324)
#         
#         noise_image4, noise = add_gaussian_noise(images, noise_std=0.0324)
#         noise_image_adv4, noise = add_gaussian_noise(adv_image, noise_std=0.0324)
#         
#         noise_image5, noise = add_gaussian_noise(images, noise_std=0.0324)
#         noise_image_adv5, noise = add_gaussian_noise(adv_image, noise_std=0.0324)


        
        with torch.no_grad():
#             noise_image1, noise = add_gaussian_noise(images, noise_std=0.12)
#             noise_image2, noise = add_gaussian_noise(adv_image, noise_std=0.12)
            delta,tta_output2 = batch_delta_confidence(images.cuda(),model,probe_sigma=0.02)
            delta2,tta_output4 = batch_delta_confidence(images.cuda(),model,probe_sigma=0.04)
            delta3,tta_output6 = batch_delta_confidence(images.cuda(),model,probe_sigma=0.06)
            delta4,tta_output8 = batch_delta_confidence(images.cuda(),model,probe_sigma=0.08)
            delta5,tta_output10 = batch_delta_confidence(images.cuda(),model,probe_sigma=0.1)
            delta6,tta_output12 = batch_delta_confidence(images.cuda(),model,probe_sigma=0.12)
            delta7,tta_output14 = batch_delta_confidence(images.cuda(),model,probe_sigma=0.14)
#             delta8,tta_output16 = batch_delta_confidence(images.cuda(),model,probe_sigma=0.16)
#             delta9,tta_output18 = batch_delta_confidence(images.cuda(),model,probe_sigma=0.18)
#             delta10,tta_output20 = batch_delta_confidence(images.cuda(),model,probe_sigma=0.2)
#             delta11,tta_output22 = batch_delta_confidence(images.cuda(),model,probe_sigma=0.22)
#             delta12,tta_output24 = batch_delta_confidence(images.cuda(),model,probe_sigma=0.24)
#             delta13,tta_output26 = batch_delta_confidence(images.cuda(),model,probe_sigma=0.26)
#             delta14,tta_output28 = batch_delta_confidence(images.cuda(),model,probe_sigma=0.28)
#             delta15,tta_output30 = batch_delta_confidence(images.cuda(),model,probe_sigma=0.3)


            # step 5: adaptive sigma
            
            
            datascln.append(delta.mean()/0.02)
            datascln.append(delta2.mean()/0.04)
            datascln.append(delta3.mean()/0.06)
            datascln.append(delta4.mean()/0.08)
            datascln.append(delta5.mean()/0.1)
            datascln.append(delta6.mean()/0.12)
            datascln.append(delta7.mean()/0.14)
#             datascln.append(delta8.mean()/0.16)
#             datascln.append(delta9.mean()/0.18)
#             datascln.append(delta10.mean()/0.2)
#             datascln.append(delta11.mean()/0.22)
#             datascln.append(delta12.mean()/0.24)
#             datascln.append(delta13.mean()/0.26)
#             datascln.append(delta14.mean()/0.28)
#             datascln.append(delta15.mean()/0.3)

            
            
            datascln2.append(delta.mean())
            datascln2.append(delta2.mean())
            datascln2.append(delta3.mean())
            datascln2.append(delta4.mean())
            datascln2.append(delta5.mean())
            datascln2.append(delta6.mean())
            datascln2.append(delta7.mean())
#             datascln2.append(delta8.mean())
#             datascln2.append(delta9.mean())
#             datascln2.append(delta10.mean())
#             datascln2.append(delta11.mean())
#             datascln2.append(delta12.mean())
#             datascln2.append(delta13.mean())
#             datascln2.append(delta14.mean())
#             datascln2.append(delta15.mean())
            
            stacked_tensor = torch.stack(datascln)  # 这会得到一个一维张量，长度为4
            max_value = stacked_tensor.max().item()
            max_index = stacked_tensor.argmax().item()  # 如果有多个最大值，返回第一个的索引
    
            datasclnvalue.append(max_index)
            datasclnvalue2.append(datascln2[max_index])
            
            


            
            #datasclns = find_plateau_by_consecutive_small_increment(datascln)
            
#             datasclns, final_delta_mean = compute_delta_with_plateau_detection(
#         images, 
#         model,
#         sigma_list=[0.02, 0.04, 0.06, 0.08, 0.10, 0.12,0.14,0.16,0.18,0.2],  # 可以扩展更多sigma
#         num_probe=1  # 可以增加采样次数
#     )
            
    
            #sigma_x = decide_batch_sigma(final_delta_mean)






            deltaadv,tta_outputadv2 = batch_delta_confidence(adv_image.cuda(),model,probe_sigma=0.02)
            deltaadv2,tta_outputadv4 = batch_delta_confidence(adv_image.cuda(),model,probe_sigma=0.04)
            deltaadv3,tta_outputadv6 = batch_delta_confidence(adv_image.cuda(),model,probe_sigma=0.06)
            deltaadv4,tta_outputadv8 = batch_delta_confidence(adv_image.cuda(),model,probe_sigma=0.08)
            deltaadv5,tta_outputadv10 = batch_delta_confidence(adv_image.cuda(),model,probe_sigma=0.1)
            deltaadv6,tta_outputadv12 = batch_delta_confidence(adv_image.cuda(),model,probe_sigma=0.12)
            deltaadv7,tta_outputadv14 = batch_delta_confidence(adv_image.cuda(),model,probe_sigma=0.14)
#             deltaadv8,tta_outputadv16 = batch_delta_confidence(adv_image.cuda(),model,probe_sigma=0.16)
#             deltaadv9,tta_outputadv18 = batch_delta_confidence(adv_image.cuda(),model,probe_sigma=0.18)
#             deltaadv10,tta_outputadv20 = batch_delta_confidence(adv_image.cuda(),model,probe_sigma=0.2)
#             deltaadv11,tta_outputadv22 = batch_delta_confidence(adv_image.cuda(),model,probe_sigma=0.22)
#             deltaadv12,tta_outputadv24 = batch_delta_confidence(adv_image.cuda(),model,probe_sigma=0.24)
#             deltaadv13,tta_outputadv26 = batch_delta_confidence(adv_image.cuda(),model,probe_sigma=0.26)
#             deltaadv14,tta_outputadv28 = batch_delta_confidence(adv_image.cuda(),model,probe_sigma=0.28)
#             deltaadv15,tta_outputadv30 = batch_delta_confidence(adv_image.cuda(),model,probe_sigma=0.3)
            
            
            datas2.append(deltaadv.mean()/0.02)
            datas2.append(deltaadv2.mean()/0.04)
            datas2.append(deltaadv3.mean()/0.06)
            datas2.append(deltaadv4.mean()/0.08)
            datas2.append(deltaadv5.mean()/0.1)
            datas2.append(deltaadv6.mean()/0.12)
            datas2.append(deltaadv7.mean()/0.14)
#             datas2.append(deltaadv8.mean()/0.16)
#             datas2.append(deltaadv9.mean()/0.18)
#             datas2.append(deltaadv10.mean()/0.2)
#             datas2.append(deltaadv11.mean()/0.22)
#             datas2.append(deltaadv12.mean()/0.24)
#             datas2.append(deltaadv13.mean()/0.26)
#             datas2.append(deltaadv14.mean()/0.28)
#             datas2.append(deltaadv15.mean()/0.3)

            
            datasx.append(delta.mean())
            datasxx.append(deltaadv.mean())   
            datasx1.append(deltaadv.mean()/0.02)
            datasx2.append(deltaadv2.mean()/0.04)
            datasx3.append(deltaadv3.mean()/0.06)
            datasx4.append(deltaadv4.mean()/0.08)
            datasx5.append(deltaadv5.mean()/0.1)
            datasx6.append(deltaadv6.mean()/0.12)
            datasx7.append(deltaadv7.mean()/0.14)
#             datasx8.append(deltaadv8.mean()/0.16)
#             datasx9.append(deltaadv9.mean()/0.18)
#             datasx10.append(deltaadv10.mean()/0.2)
#             datasx11.append(deltaadv11.mean()/0.22)
#             datasx12.append(deltaadv12.mean()/0.24)
#             datasx13.append(deltaadv13.mean()/0.26)
#             datasx14.append(deltaadv14.mean()/0.28)
#             datasx15.append(deltaadv15.mean()/0.3)

            
            stacked_tensor2 = torch.stack(datas2)  # 这会得到一个一维张量，长度为4
            max_value2 = stacked_tensor2.max().item()
            max_index2 = stacked_tensor2.argmax().item()  # 如果有多个最大值，返回第一个的索引
            datasvalue.append(max_index2)
            datasvalue2.append(datas2[max_index2])
            
            noise_cln = compute(delta.mean(), max_index)
            noise_adv = compute(deltaadv.mean(), max_index2)

        
            
            noise_clns = generate_neighbors(noise_cln, 4)
            noise_advs = generate_neighbors(noise_adv, 4)


 

        
            
            
# #             
#             datas.append(deltaadv.mean())
#             datas.append(deltaadv2.mean())
#             datas.append(deltaadv3.mean())
#             datas.append(deltaadv4.mean())
#             datas.append(deltaadv5.mean())
#             datas.append(deltaadv6.mean())
#             datas.append(deltaadv7.mean())
#             datas.append(deltaadv8.mean())
#             datas.append(deltaadv9.mean())
#             datas.append(deltaadv10.mean())
#             datas.append(deltaadv11.mean())
#             datas.append(deltaadv12.mean())
#             datas.append(deltaadv13.mean())
            if noise_cln == 0:
                noise = torch.randn_like(images).cuda() * noise_cln
                defended_images = images.cuda() + noise.cuda()
                z_source, text_features, logit_scale = model.forward_features(defended_images.cuda())
            else:
                z_sou_list = []
                output_list = []
                for zaosheng in noise_clns:#noise_clns
                    noise = torch.randn_like(images).cuda() * zaosheng
                    defended_images = images.cuda() + noise.cuda()
                    z_sou, text_features, logit_scale = model.forward_features(defended_images.cuda())
                    output = logit_scale * (z_sou) @ text_features.t()
                    z_sou_list.append(z_sou)
                    output_list.append(output)
                z_source = combine_features_by_entropy(z_sou_list, output_list)
            if noise_adv == 0:
                noiseadv = torch.randn_like(images).cuda() * noise_adv
                defended_imagesadv = adv_image.cuda() + noiseadv.cuda()
                z_adv_source, text_features, logit_scale = model.forward_features(defended_imagesadv)
            else:
                z_sou_list_adv = []
                output_list_adv = []
                for zaosheng_adv in noise_advs:
                   
                    noise_adv = torch.randn_like(images).cuda() * zaosheng_adv
                    defended_images_adv = adv_image.cuda() + noise_adv.cuda()
                    z_sou_adv, text_features, logit_scale = model.forward_features(defended_images_adv.cuda())
                    output_adv = logit_scale * (z_sou_adv) @ text_features.t()


                    z_sou_list_adv.append(z_sou_adv)
                    output_list_adv.append(output_adv)
                z_adv_source = combine_features_by_entropy(z_sou_list_adv, output_list_adv)
#                 
                
                
                
                
                
#             noise = torch.randn_like(images).cuda() * noise_cln
#             defended_images = images.cuda() + noise.cuda()
#             z_source, text_features, logit_scale = model.forward_features(defended_images.cuda())
#             
#             noiseadv = torch.randn_like(images).cuda() * noise_adv
#             defended_imagesadv = adv_image.cuda() + noiseadv.cuda()
#             z_adv_source, text_features, logit_scale = model.forward_features(defended_imagesadv)            
#             
            f_sou, _, _ = model.forward_features(images.cuda())
            f_adv_sou, _, _ = model.forward_features(adv_image)
    
            z_source = -1* f_sou +0 * z_source
            z_adv_source = 1 * f_adv_sou + 0 * z_adv_source
            
            
            
            
            
            
            
            
#             logits_clean = logit_scale * (z_sou) @ text_features.t()
#             prob_clean = F.softmax(logits_clean, dim=1)
#             max_idx_clean = prob_clean.argmax(dim=-1)        # [B]
#             conf_clean, _ = prob_clean.max(dim=1)      # [B]
#             
#             logits_adv = logit_scale * (z_adv_sou) @ text_features.t()
#             prob_adv = F.softmax(logits_adv, dim=1)
#             max_idx_adv = prob_adv.argmax(dim=-1)
#             conf_adv, _ = prob_adv.max(dim=1)      # [B]
#             
#             
#             noise_002 = torch.randn_like(images) * 0.05
#             x_002 = images + noise_002
# 
#             #logits_002 = model(x_002)
#             z_002, text_features002, logit_scale002 = model.forward_features(x_002.cuda())
#             logits_002 = logit_scale002 * (z_002) @ text_features002.t()
#             prob_002 = F.softmax(logits_002, dim=1)
#             conf_002, _ = prob_002.max(dim=1)           # [B]
#             
#             
#             x_002adv = adv_image + noise_002.cuda()
# 
#             #logits_002 = model(x_002)
#             z_002adv, text_features002, logit_scale002 = model.forward_features(x_002adv)
#             logits_002adv = logit_scale002 * (z_002adv) @ text_features002.t()
#             prob_002adv = F.softmax(logits_002adv, dim=1)
#             conf_002adv, _ = prob_002adv.max(dim=1)           # [B]


#             out = batch_adaptive_noise_features(images.cuda(), model,normalized_filtered)
#             feat_a = out["feat_1"]
#             feat_b = out["feat_2"]
#             deltaadv = out["delta_conf"]
            
#             deltaadvs.append((conf_clean - conf_002).abs())
#             deltaclns.append((conf_adv - conf_002adv).abs())
#             deltaclns.append(delta.mean().cpu())
#             deltaadvs.append(deltaadv.mean().cpu())


#             # 比如：平均 / 拼接 / 相似度投票
#             feat = (feat_a + feat_b) / 2
# 
#             
#             outadv = batch_adaptive_noise_features(adv_image, model,normalized_filtered2)
#             feat_aadv = outadv["feat_1"]
#             feat_badv = outadv["feat_2"]
#             deltacln = outadv["delta_conf"]
#             deltaclns.append(deltacln)
#             
# 
# 
#             # 比如：平均 / 拼接 / 相似度投票
#             featadv = (feat_aadv + feat_badv) / 2

            
#             z_defended_cln = -0.2 * z_sou + 1.2*feat
#             z_defended = -0.2* z_adv_sou + 1.2*featadv
            z_defended_cln = z_source 
            z_defended = z_adv_source


            all_labels.append(target.cpu())
            
        tta_output = logit_scale * (z_defended_cln) @ text_features.t()
        tta_output1 = logit_scale * (z_defended) @ text_features.t()
        analyzer = FeatureSpaceAnalyzer(temperature=0.01)
# 
        print("=== 1/255 对抗特征分析 ===")
        res_1 = analyzer.analyze(f_sou, f_adv_sou, text_features)
        for k, v in res_1.items(): print(f"{k}: {v:.4f}")
#             z1, text_features, logit_scale = model.forward_features(noise_image1)
#             z_adv1, text_features, logit_scale = model.forward_features(noise_image_adv1)
#             
#             z2, text_features, logit_scale = model.forward_features(noise_image2)
#             z_adv2, text_features, logit_scale = model.forward_features(noise_image_adv2)
#             
#             z3, text_features, logit_scale = model.forward_features(noise_image3)
#             z_adv3, text_features, logit_scale = model.forward_features(noise_image_adv3)
#             
#             z4, text_features, logit_scale = model.forward_features(noise_image4)
#             z_adv4, text_features, logit_scale = model.forward_features(noise_image_adv4)
#             
#             z5, text_features, logit_scale = model.forward_features(noise_image5)
#             z_adv5, text_features, logit_scale = model.forward_features(noise_image_adv5)
#             
#             z_sou, text_features, logit_scale = model.forward_features(images)
#             z_adv_sou, text_features, logit_scale = model.forward_features(adv_image)
            
#             z_defended_cln = -0.2 * z_sou + 1.2 * (z1 + z2 + z3 + z4 +z5) / 5
#             z_defended = -0.2 * z_adv_sou + 1.2 * (z_adv1 + z_adv2 + z_adv3 + z_adv4 +z_adv5) / 5

#             z_defended_cln =  (z1 + z2 + z3 + z4 +z5) / 5
#             z_defended =  (z_adv1 + z_adv2 + z_adv3 + z_adv4 +z_adv5) / 5
            
            
            #0.16 2
#             k = 2
#             alpha=0.3
#             beta=0.9
#             sigmas = generate_sigmas()
#             # Step 1: batch forward
#             z0_flat, z_all = collect_all_embeddings(model.forward_features, adv_image, sigmas, device='cuda')
#             # Step 2: compute curve
#             deltas = compute_delta_curve_per_sample(z0_flat, z_all)
# 
#             # Step 3: search
#             chosen_idx = search_on_curve_per_sample(
#         sigmas, deltas, k=k, alpha=alpha, beta=beta
#     )
#             
#             all_z = []
# 
#             for offset in range(0, k+1):
#                 # 计算窗口内索引
#                 idx = (chosen_idx + offset).clamp(0, len(sigmas)-1)
# 
#                 sigmas_tensor = torch.tensor(sigmas, device=adv_image.device)
#                 chosen_sigmas = sigmas_tensor[idx]          # [B]
#                 
# 
#                 noise = torch.randn_like(adv_image) * chosen_sigmas.view(-1,1,1,1)
# 
#                 z, text_features, logit_scale = model.forward_features(adv_image + noise)
#                 all_z.append(z)
#             # 对窗口内结果取平均
#             z_defended = torch.stack(all_z, dim=0).mean(dim=0)
# 
#             
#             # Step 1: batch forward
#             z0_flat_cln, z_all_cln = collect_all_embeddings(model.forward_features, images, sigmas, device='cuda')
#             # Step 2: compute curve
#             deltas_cln = compute_delta_curve_per_sample(z0_flat_cln, z_all_cln)
#             # Step 3: search
#             chosen_idx_cln = search_on_curve_per_sample(
#         sigmas, deltas_cln, k=k, alpha=alpha, beta=beta
#     )
#             
#             all_z_cln = []
# 
#             for offset in range(-k, k+1):
#                 # 计算窗口内索引
#                 idx = (chosen_idx_cln + offset).clamp(0, len(sigmas)-1)
# 
#                 sigmas_tensor = torch.tensor(sigmas, device=adv_image.device)
#                 chosen_sigmas = sigmas_tensor[idx]          # [B]
# 
#                 noise = torch.randn_like(images) * chosen_sigmas.view(-1,1,1,1)
# 
#                 z_cln, text_features_cln, logit_scale_cln = model.forward_features(images + noise)
#                 all_z_cln.append(z_cln)
#             # 对窗口内结果取平均
#             z_defended_cln = torch.stack(all_z_cln, dim=0).mean(dim=0)
#             
#             










            #clip_output = model(image)
            
            #all_logits.append(clip_output.cpu())
            #feat_mean = consensus_weighted_mean(clip_features)
            
            #clip_outputs = model(images)
            

        #assert args.tta_steps > 0
        #test_time_tuning(mu_dict, clip_output, clip_outputs, clip_features, classnames, sigma_dict, model, images, optimizer, scaler, args)

#         shuffled_image, shuffle_indices = shuffle_image(image, block_size=(32, 32), return_indices=True)
#         
#         #noisy_image = image + noise_image - img_ori
#         noisy_image1, noise1 = add_gaussian_noise(image, noise_std=0.01)
#         noise1 = noisy_image1 - image
#         
#         noisy_image2, noise2 = add_rademacher_noise(image, noise_std=0.02)
#         noise2 = noisy_image2 - image
#         
#         noisy_image3, noise3 = add_gaussian_noise(image, noise_std=0.02)
#         noise3 = noisy_image3 - image
#         
#         noisy_image4, noise4 = add_gaussian_noise(image, noise_std=0.03)
#         noise4 = noisy_image4 - image
#         
#         delta3 = torch.clamp(noise2, min=-16/255, max=16/255)
# 
# #     
#         shuffled_image, shuffle_indices = shuffle_image(noisy_image1, block_size=(32, 32), return_indices=True)
# #         

#         adv_image3 = atk2(noisy_image1, target)
#         adv_image4 = atk2(noisy_image2, target)
        #delta = adv_image2 - shuffled_image
        #delta2 = image - img_ori

        #print(analyze_vector_relationship(delta2, delta))
        #restored_delta = restore_image(delta, shuffle_indices, block_size=(32, 32))
        #adv_image2 = atk2(shuffled_image, target)
        #restored_img = restore_image(adv_image2, shuffle_indices, block_size=(32, 32))
        

#         with torch.no_grad():
#             #sigmas = torch.linspace(0.01, 0.3, 60).tolist()
#             k = 3
#             alpha=0.25
#             beta=0.8
#             sigmas = generate_sigmas()
            
            
                # Step 1: batch forward
#             z0_flat, z_all = collect_all_embeddings(model.forward_features, adv_image, sigmas, device='cuda')
#         
# 
#     # Step 2: compute curve
#             deltas = compute_delta_curve_per_sample(z0_flat, z_all)
# 
# 
#     # Step 3: search
#             chosen_idx = search_on_curve_per_sample(
#         sigmas, deltas, k=k, alpha=alpha, beta=beta
#     )
#            
# 
#             all_z = []
# 
#             for offset in range(-k, k+1):
# 
#     # 计算窗口内索引
#                 idx = (chosen_idx + offset).clamp(0, len(sigmas)-1)
# 
#                 sigmas_tensor = torch.tensor(sigmas, device=adv_image.device)
#                 chosen_sigmas = sigmas_tensor[idx]          # [B]
# 
#                 noise = torch.randn_like(adv_image) * chosen_sigmas.view(-1,1,1,1)
# 
#                 z, _, _ = model.forward_features(adv_image + noise)
# 
#                 all_z.append(z)
# 
#             # 对窗口内结果取平均
#             z_defended = torch.stack(all_z, dim=0).mean(dim=0)
            
            
            
            
    
#             all_labels.append(target.cpu())
#             chosen_idx, curve = adaptive_sigma_search_final(
#     model.forward_features,
#     adv_image,
#     sigmas,
#     k=3,
#     alpha=0.25,
#     beta=0.8
# )
#             
#             left_k  = min(k, chosen_idx)
#             right_k = min(k, len(sigmas) - 1 - chosen_idx)
# 
#             start_idx = chosen_idx - left_k
#             end_idx   = chosen_idx + right_k
# 
#             sigmas_window = sigmas[start_idx:end_idx+1]
#             
#             
#             z_list = []
# 
#             for sigma in sigmas_window:
#                 noise = torch.randn_like(adv_image) * sigma
#                 z, text_features, logit_scale = model.forward_features(adv_image + noise)
#                 z_list.append(z)
# 
#             z_defended = torch.stack(z_list, dim=0).mean(dim=0)
#            
#             chosen_idx_cln, curve = adaptive_sigma_search_final(
#     model.forward_features,
#     images,
#     sigmas,
#     k=3,
#     alpha=0.25,
#     beta=0.8
# )
#             left_k_cln  = min(k, chosen_idx_cln)
#             right_k_cln = min(k, len(sigmas) - 1 - chosen_idx_cln)
# 
#             start_idx_cln = chosen_idx_cln - left_k_cln
#             end_idx_cln   = chosen_idx_cln + right_k_cln
# 
#             sigmas_window_cln = sigmas[start_idx_cln:end_idx_cln+1]
#             
#             
#             z_list_cln = []
# 
#             for sigma in sigmas_window_cln:
#                 noise_cln = torch.randn_like(images) * sigma
#                 z_cln, _, _ = model.forward_features(images + noise)
#                 z_list_cln.append(z_cln)
# 
#             z_defended_cln = torch.stack(z_list_cln, dim=0).mean(dim=0)       

#             z0_flat_cln, z_all_cln = collect_all_embeddings(model.forward_features, images, sigmas, device='cuda')
# 
#     # Step 2: compute curve
#             deltas_cln = compute_delta_curve_per_sample(z0_flat_cln, z_all_cln)
# 
#     # Step 3: search
#             chosen_idx_cln = search_on_curve_per_sample(
#         sigmas, deltas_cln, k=k, alpha=alpha, beta=beta
#     )
#             
# 
#             all_z_cln = []
# 
#             for offset in range(-k, k+1):
# 
#     # 计算窗口内索引
#                 idx = (chosen_idx_cln + offset).clamp(0, len(sigmas)-1)
# 
#                 sigmas_tensor = torch.tensor(sigmas, device=adv_image.device)
#                 chosen_sigmas = sigmas_tensor[idx]          # [B]
# 
#                 noise = torch.randn_like(adv_image) * chosen_sigmas.view(-1,1,1,1)
# 
#                 z_cln, text_features, logit_scale = model.forward_features(images + noise)
# 
#                 all_z_cln.append(z_cln)
# 
#             # 对窗口内结果取平均
#              = torch.stack(all_z_cln, dim=0).mean(dim=0)            
    
            


            #sigmas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3]
#             z_list, delta_norm, safe_sigma_indices, safe_interval = noise_response_converge_interval(model.forward_features, img_ori, sigmas,
#     delta_thresh=1e-3, consec=3, max_interval_frac=0.3)
            #sigmas1 = generate_sigmas()

            
#             z_list, delta_norm, elbow_idx,dis,confs = noise_response_elbow_batch(model.forward_features, images, adv_image, sigmas1)
#             slope = compute_simple_slopes(delta_norm,sigmas1)
#             slope = delta_norm.max(dim=0).values
#             slope1.append(slope)
#             z_listcln, delta_normcln, elbow_idxcln,discln,confscln = noise_response_elbow_batch(model.forward_features, images, images, sigmas1)
#             slopecln = compute_simple_slopes(delta_normcln,sigmas1)
#             slopecln = delta_normcln.max(dim=0).values
#             slope2.append(slopecln)



#             noisy_embeddings = torch.stack(z_list)
#             z_list, delta_norm, elbow_idx,dis,confs = noise_response_elbow_batch(model.forward_features, images, images, sigmas1)
#             result_matrix = analyze_batch_categories(confs, sigmas1, target)
#             class1.append(result_matrix[:, 0].sum().item())
#             class2.append(result_matrix[:, 1].sum().item())
#             class3.append(result_matrix[:, 2].sum().item())
#             class4.append(result_matrix[:, 3].sum().item())
            
            
#             z_listcln, delta_normcln, elbow_idxcln,discln,confscln = noise_response_elbow(model.forward_features, images, adv_image, sigmas1)
#             results = plot_confidence_for_example(confscln, sigmas1, target)
            #slope = plot_noise_response_simple(confs, delta_norm, sigmas1, elbow_idx)
            
            
            #noisy_embeddingscln = torch.stack(z_listcln)
            #slopecln = plot_noise_response_simple(confscln, delta_normcln, sigmas1, elbow_idxcln)

            
#             print(slope[0]+slope[1]-slope[-1]-slope[-2])
#             print(slopecln[0]+slopecln[1]-slopecln[-1]-slopecln[-2])
            
            
     
            

#             slope1.append(slope[0]+slope[1]-slope[2]-slope[3])
#             slope2.append(slopecln[0]+slopecln[1]-slopecln[2]-slopecln[3])


#             
# #             
#             img_features1, text_features, logit_scale = model.forward_features(image)
#             img_features2, text_features, logit_scale = model.forward_features(img_ori)
#             img_features3, text_features, logit_scale = model.forward_features(noisy_image3)
#             img_features4, text_features, logit_scale = model.forward_features(noisy_image4)
            
#             delta = (img_features1.view(1,-1) - img_features2.view(1,-1)).norm(dim=1)
#             print(delta.mean().item())
            

#             img_features2, text_features, logit_scale = model.forward_features(img_ori+0.2*noise2)
#             img_features3, text_features, logit_scale = model.forward_features(img_ori+0.3*noise2)
#             img_features4, text_features, logit_scale = model.forward_features(img_ori+0.4*noise2)
#             img_features5, text_features, logit_scale = model.forward_features(img_ori+0.5*noise2)
#             img_features6, text_features, logit_scale = model.forward_features(img_ori)
#             img_features7, text_features, logit_scale = model.forward_features(img_ori+1*noise1)
#             img_features8, text_features, logit_scale = model.forward_features(image)
#             img_features9, text_features, logit_scale = model.forward_features(img_ori+0.01*noise1)
#             f_x, f_xeps = extract_noise_features(model, img_ori)
#             a, r = top1_pca_energy_ratio(img_features6, img_features8, f_xeps)
#             
#             adv_curve, rand_curve = run_topk_alignment(img_features6, img_features8, f_xeps)
            
#             z_bar  = estimate_semantic_anchor(
#     image,
#     model
# )
#             z_def = mode_control_projection(
#     image,
#     model,z_bar
# )

#             plt.plot(adv_curve, label="adv direction")
#             plt.plot(rand_curve, label="random direction", linestyle="--")
#             plt.xlabel("k (Top-k PCA components)")
#             plt.ylabel("Projection ratio")
#             plt.legend()
#             plt.title("Adv vs Noise PCA Alignment")
#             plt.show()
            
#             adv_vals.append(a)
#             rand_vals.append(r)
#             
#             v_adv = img_features8 - img_features6
#             v_adv = F.normalize(v_adv, dim=-1).squeeze(0)  # [D]
# 
#     # ---- 2. 噪声方向集合 ----
#             V_noise = f_xeps - f_x  # [N, D]
#             V_noise = V_noise - V_noise.mean(dim=0, keepdim=True)
# 
#     # ---- 3. PCA (SVD) ----
#             _, _, Vh = torch.linalg.svd(V_noise, full_matrices=False)
#             U_noise = Vh[:10].T  # [D, K]
# 
#     # ---- 4. 投影比例 ----
#             proj = U_noise.T @ v_adv  # [K]
#             ratio = torch.norm(proj, p=2)
#             
#             r_rand = random_projection_ratio(f_xeps, f_x)
# 
#             energy, singular_vals = adversarial_spectral_energy(
#     f_x=f_x,
#     f_xadv=img_features8,
#     f_xeps=f_xeps
# )
# 
# # 可选：只看前 100 个方向
#             energy_100 = energy[:100]
#             singular_100 = singular_vals[:100]
#             
#             plt.figure(figsize=(16,14))
#             plt.plot(energy_100.numpy())
#             plt.xlabel("Noise PCA direction index")
#             plt.ylabel("Adversarial spectral energy")
#             plt.title("Spectral distribution of adversarial direction")
#             plt.tight_layout()
#             plt.show()


            
#             img_features10, text_features, logit_scale = model.forward_features(image+1.3*noise2)
#             
#             img_features11, text_features, logit_scale = model.forward_features(image+0.9*noise2)
#             img_features12, text_features, logit_scale = model.forward_features(image+1.4*noise2)
            
            #print(F.cosine_similarity((img_features7-img_features8) / torch.norm(img_features7-img_features8, p=2, dim=1, keepdim=True), (img_features9-img_features10)/ torch.norm(img_features9-img_features10, p=2, dim=1, keepdim=True), dim=1, eps=1e-8).mean())
            #print((F.cosine_similarity(img_features_ori, img_features_shuf + 0.50*(img_features-img_features_adv) / torch.norm(img_features-img_features_adv, p=2, dim=1, keepdim=True), dim=1, eps=1e-8).mean()))
            
            #print(F.cosine_similarity(img_features7, img_features10, dim=1, eps=1e-8).mean())
            #dire1 = (img_features7-img_features6) / torch.norm(img_features7-img_features6, p=2, dim=1, keepdim=True)
            #dire2 = (img_features8-img_features6) / torch.norm(img_features8-img_features6, p=2, dim=1, keepdim=True)
            #dire3 = (img_features9-img_features8) / torch.norm(img_features9-img_features8, p=2, dim=1, keepdim=True)
            
            
            
#             
#             img_features3_jia = img_features10 - 0.07 * dire1
#             img_features4_jia = img_features12 - 0.07 * dire2
#             dire3 = (img_features4_jia-img_features3_jia) / torch.norm(img_features4_jia-img_features3_jia, p=2, dim=1, keepdim=True)
#             img_features7_jia = img_features7 - 0.07 * dire3
            #print(F.cosine_similarity(img_features9, img_features7 - 0.07 * dire2 - 0.07 * 0.07 * dire2 - 0.07 * 0.07 * 0.07 * dire2, dim=1, eps=1e-8).mean())
            #print(F.cosine_similarity(dire1, dire2, dim=1, eps=1e-8).mean())
            
#             cos.append(F.cosine_similarity(img_features3_jia, img_features, dim=1, eps=1e-8).mean())
#             cos2.append(F.cosine_similarity(img_features10, img_features, dim=1, eps=1e-8).mean())
            
            #print(F.cosine_similarity(img_features8 - 0.07 * dire2, img_features9, dim=1, eps=1e-8).mean())
            #cos.append(dire1)
            #print(F.cosine_similarity(z_def, img_features6, dim=1, eps=1e-8).mean())
            

            
            
            
            #print(torch.sqrt(torch.sum((img_features - img_features_shuf) ** 2)))


#         confidence_scores_cln = F.softmax(tta_output, dim=1).max(dim=1).values
#         confidence_scores = F.softmax(tta_output1, dim=1).max(dim=1).values
#         class1.append(confidence_scores_cln)
#         class2.append(confidence_scores)
        
        
        
            
            

            
            #confs.append(torch.max(F.softmax(tta_output1, dim=-1)))
            #ents.append(entropy(tta_output1))
            #features_per_class[classnames[target]].append(img_features.squeeze(0).detach().cpu())
            
            #tuned_outputs = model(images)
        
#         sim_matrix_images = torch.bmm(clip_features.unsqueeze(0), clip_features.unsqueeze(0).permute(0, 2, 1))
#         score = get_top_sim(sim_matrix_images)
#         weight = torch.nn.functional.softmax(score/0.01, dim=-1)
#         tta_output = torch.bmm(weight.unsqueeze(-1).transpose(1, 2), tuned_outputs.unsqueeze(0)).squeeze(1)
        

#         sigma_img_dict = update_sigma_img_dict(
#     clip_features, tta_output, classnames, mu_dict, sigma_img_dict,
#     diag_store=False,  # 若 True 则只存对角协方差
#     eps=1e-6
# )
        

        all_logits_tta.append(tta_output.cpu())
        all_logits.append(tta_output1.cpu())



        # measure accuracy and record loss
        acc1, acc5 = accuracy(tta_output, target, topk=(1, 2))
        tpt_acc1, _ = accuracy(tta_output1, target, topk=(1, 2))
        
        acc2, _ = accuracy(tta_output2, target, topk=(1, 2))
        tpt_acc2, _ = accuracy(tta_outputadv2, target, topk=(1, 2))

        acc4, _ = accuracy(tta_output4, target, topk=(1, 2))
        tpt_acc4, _ = accuracy(tta_outputadv4, target, topk=(1, 2))
        
        acc6, _ = accuracy(tta_output6, target, topk=(1, 2))
        tpt_acc6, _ = accuracy(tta_outputadv6, target, topk=(1, 2))
        
        acc8, _ = accuracy(tta_output8, target, topk=(1, 2))
        tpt_acc8, _ = accuracy(tta_outputadv8, target, topk=(1, 2))
        
#         acc10, _ = accuracy(tta_output10, target, topk=(1, 5))
#         tpt_acc10, _ = accuracy(tta_outputadv10, target, topk=(1, 5))
#         
#         acc12, _ = accuracy(tta_output12, target, topk=(1, 5))
#         tpt_acc12, _ = accuracy(tta_outputadv12, target, topk=(1, 5))
#         
#         acc14, _ = accuracy(tta_output14, target, topk=(1, 5))
#         tpt_acc14, _ = accuracy(tta_outputadv14, target, topk=(1, 5))
#         
#         acc16, _ = accuracy(tta_output16, target, topk=(1, 5))
#         tpt_acc16, _ = accuracy(tta_outputadv16, target, topk=(1, 5))
#         
#         
#         acc18, _ = accuracy(tta_output18, target, topk=(1, 5))
#         tpt_acc18, _ = accuracy(tta_outputadv18, target, topk=(1, 5))
#         
#         acc20, _ = accuracy(tta_output20, target, topk=(1, 5))
#         tpt_acc20, _ = accuracy(tta_outputadv20, target, topk=(1, 5))
#         
#         acc22, _ = accuracy(tta_output22, target, topk=(1, 5))
#         tpt_acc22, _ = accuracy(tta_outputadv22, target, topk=(1, 5))
#         
#         acc24, _ = accuracy(tta_output24, target, topk=(1, 5))
#         tpt_acc24, _ = accuracy(tta_outputadv24, target, topk=(1, 5))
#         
#         acc26, _ = accuracy(tta_output26, target, topk=(1, 5))
#         tpt_acc26, _ = accuracy(tta_outputadv26, target, topk=(1, 5))
#         
#         
#         acc28, _ = accuracy(tta_output28, target, topk=(1, 5))
#         tpt_acc28, _ = accuracy(tta_outputadv28, target, topk=(1, 5))
#         
#         acc30, _ = accuracy(tta_output30, target, topk=(1, 5))
#         tpt_acc30, _ = accuracy(tta_outputadv30, target, topk=(1, 5))
       
        top1.update(acc1[0], images.size(0))
        tpt1.update(tpt_acc1[0], images.size(0))
        
        top1002.update(acc2[0], images.size(0))
        tpt1002.update(tpt_acc2[0], images.size(0))
        
        top1004.update(acc4[0], images.size(0))
        tpt1004.update(tpt_acc4[0], images.size(0))
        
        top1006.update(acc6[0], images.size(0))
        tpt1006.update(tpt_acc6[0], images.size(0))
        
        top1008.update(acc8[0], images.size(0))
        tpt1008.update(tpt_acc8[0], images.size(0))
        
#         top101.update(acc10[0], images.size(0))
#         tpt101.update(tpt_acc10[0], images.size(0))
#         
#         top1012.update(acc12[0], images.size(0))
#         tpt1012.update(tpt_acc12[0], images.size(0))
#         
#         top1014.update(acc14[0], images.size(0))
#         tpt1014.update(tpt_acc14[0], images.size(0))
#         
#         top1016.update(acc16[0], images.size(0))
#         tpt1016.update(tpt_acc16[0], images.size(0))
#         
#         top1018.update(acc18[0], images.size(0))
#         tpt1018.update(tpt_acc18[0], images.size(0))
#         
#         top102.update(acc20[0], images.size(0))
#         tpt102.update(tpt_acc20[0], images.size(0))
#         
#         top1022.update(acc22[0], images.size(0))
#         tpt1022.update(tpt_acc22[0], images.size(0))
#         
#         top1024.update(acc24[0], images.size(0))
#         tpt1024.update(tpt_acc24[0], images.size(0))
#         
#         top1026.update(acc26[0], images.size(0))
#         tpt1026.update(tpt_acc26[0], images.size(0))
#         
#         top1028.update(acc28[0], images.size(0))
#         tpt1028.update(tpt_acc28[0], images.size(0))
#         
#         top103.update(acc30[0], images.size(0))
#         tpt103.update(tpt_acc30[0], images.size(0))
    

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0 or (i+1) == len(val_loader):
            if args.eps <= 0:
                print_log = 'iter:{}/{}, clip_acc1={}, tta_acc1={}'.format(i, len(val_loader), top1.avg, tpt1.avg)
            else:
                print_log = 'iter:{}/{}, clip_adv1={}, tta_adv1={}'.format(i, len(val_loader), top1.avg, tpt1.avg)
            args.out_file.write(print_log + '\n')
            args.out_file.flush()
            print(print_log+'\n')
            progress.display(i)

    end_time = time.time()

#     t1 = torch.cat(deltaclns, dim = 0)
#     t2 = torch.cat(deltaadvs, dim = 0)
#     t1 = deltaclns
#     t2 = deltaadvs
#     analyze_tensor_distribution_simple(t1,t2)
# 
#     plot_scatter_for_tensor_lists(t1,t2)
    
#     t1 = t1.detach().cpu()
#     t2 = t2.detach().cpu()
# 
#     # --------- 散点图 ----------
#     plt.figure(figsize=(10,5))
#     plt.scatter(range(len(t1)), t1, label='Tensor 1', alpha=0.7)
#     plt.scatter(range(len(t2)), t2, label='Tensor 2', alpha=0.7)
#     plt.xlabel('Index')
#     plt.ylabel('Value')
#     plt.title('Scatter Plot of Two Tensors')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#     plt.close()
# 
#     # --------- 统计各区间百分比 ----------
#     bins = np.linspace(0, max(t1.max(), t2.max()).item(), 6)  # 0到最大值，分5个区间
#     tensors = {'Tensor 1': t1, 'Tensor 2': t2}
# 
#     for name, t in tensors.items():
#         counts, _ = np.histogram(t.numpy(), bins=bins)
#         percents = counts / counts.sum() * 100
#         print(f'{name} 区间统计:')
#         for i in range(len(bins)-1):
#             print(f'  {bins[i]:.2f} - {bins[i+1]:.2f}: {percents[i]:.2f}%')
#         print()
# 
#     # --------- 区间百分比可视化 ----------
#     plt.figure(figsize=(10,5))
#     width = (bins[1]-bins[0])/3
#     for idx, (name, t) in enumerate(tensors.items()):
#         counts, _ = np.histogram(t.numpy(), bins=bins)
#         percents = counts / counts.sum() * 100
#         plt.bar(bins[:-1]+idx*width, percents, width=width, label=name, alpha=0.7)
# 
#     plt.xlabel('Value Range')
#     plt.ylabel('Percentage (%)')
#     plt.title('Percentage of Points in Each Value Range')
#     plt.xticks(bins)
#     plt.legend()
#     plt.grid(True, axis='y')
#     plt.show()
#     plt.close()
#     print(torch.mean(torch.cat(class1, dim=0)))
#     print(torch.mean(torch.cat(class2, dim=0)))

#     simple_slope_scatter(
#     slope1, 
#     slope2,
#     slope1_name='原始斜率',
#     slope2_name='增强斜率',
#     title='斜率值对比',
#     save_path='/media/cqu/D/FXV/R-TPT-main6/simple_slope_scatter.png'
# )
    print(f"总训练时间: {(end_time - start_time)/60:.2f} 分钟")
    
#     stacked_tensor3 = torch.stack(datasclnvalue2)  # 这会得到一个一维张量，长度为4
#     print(torch.mean(stacked_tensor3))
#     stacked_tensor4 = torch.stack(datasvalue2)  # 这会得到一个一维张量，长度为4
#     print(torch.mean(stacked_tensor4))
#     print(sum(datasclnvalue)/len(datasclnvalue))
    print(sum(datasx)/len(datasx))
    print(sum(datasxx)/len(datasxx))

    stacked_tensor1 = torch.stack(datasx1)  # 这会得到一个一维张量，长度为4
    stacked_tensor2 = torch.stack(datasx2)
    stacked_tensor3 = torch.stack(datasx3)
    stacked_tensor4 = torch.stack(datasx4)
    stacked_tensor5 = torch.stack(datasx5)
    stacked_tensor6 = torch.stack(datasx6)
#     stacked_tensor7 = torch.stack(datasx7)
#     stacked_tensor8 = torch.stack(datasx8)
#     stacked_tensor9 = torch.stack(datasx9)
#     stacked_tensor10 = torch.stack(datasx10)
#     stacked_tensor11 = torch.stack(datasx11)
#     stacked_tensor12 = torch.stack(datasx12)
#     stacked_tensor13 = torch.stack(datasx13)
#     stacked_tensor14 = torch.stack(datasx14)
#     stacked_tensor15 = torch.stack(datasx15)
    

    
    print(torch.mean(stacked_tensor1).item())
    print(torch.mean(stacked_tensor2).item())
    print(torch.mean(stacked_tensor3).item())
    print(torch.mean(stacked_tensor4).item())
    print(torch.mean(stacked_tensor5).item())
    print(torch.mean(stacked_tensor6).item())
#     print(torch.mean(stacked_tensor7).item())
#     print(torch.mean(stacked_tensor8).item())
#     print(torch.mean(stacked_tensor9).item())
#     print(torch.mean(stacked_tensor10).item())
#     print(torch.mean(stacked_tensor11).item())
#     print(torch.mean(stacked_tensor12).item())
#     print(torch.mean(stacked_tensor13).item())
#     print(torch.mean(stacked_tensor14).item())
#     print(torch.mean(stacked_tensor15).item())
    
#     stacked_tensorx = torch.cat(datasx, dim=0)
#     stacked_tensorxx = torch.cat(datasxx, dim=0)
#     torch.save(datasx, '/media/cqu/D/FXV/R-TPT-main6/Caltech101clnbatch.pt')
#     torch.save(datasxx, '/media/cqu/D/FXV/R-TPT-main6/Caltech101batch.pt')
#     print(sum(slope1) / len(slope1))
#     print(sum(slope2) / len(slope2))
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_logits_tta = torch.cat(all_logits_tta, dim=0)
#     cos_tensor = torch.stack(cos)
#     torch.save(cos_tensor, '/media/cqu/D/FXV/R-TPT-main6/cos_tensor.pt')
    #cos_tensor2 = torch.stack(cos2)
    
#     print(cos_tensor.mean())
#     print(cos_tensor2.mean())
    
    #conf_tensor = torch.stack(confs)
    #ent_tensor = torch.stack(ents)
    
    #dis_tensor = torch.stack(distance)
    #torch.save(dis_tensor, '/media/cqu/D/FXV/R-TPT-main6/distanceimgtxt.pt')
#     torch.save(cos_tensor, '/media/cqu/D/FXV/R-TPT-main6/cos_tensor0.01.pt')
#     torch.save(conf_tensor, '/media/cqu/D/FXV/R-TPT-main6/conf_tensor0.01.pt')
#     torch.save(ent_tensor, '/media/cqu/D/FXV/R-TPT-main6/ent_tensor0.01.pt') 

    
    metrics = compute_per_class_stats(all_logits, all_labels, n_bins=15)
    metrics_tta = compute_per_class_stats(all_logits_tta, all_labels, n_bins=15)
#     for c, info in metrics.items():
#         print(f"Class {c}: avg_conf={info['avg_conf']:.3f}, ece={info['ece']:.3f}, n_samples={info['n_samples']}")
#         
#     for c, info in metrics_tta.items():
#         print(f"Class {c}: avg_conf={info['avg_conf']:.3f}, ece={info['ece']:.3f}, n_samples={info['n_samples']}")
        
    
    
#     torch.save(metrics, "/media/cqu/D/FXV/R-TPT-main6/metrics.pth")
#     torch.save(metrics_tta, "/media/cqu/D/FXV/R-TPT-main6/metrics_ttaours.pth")
    
#     torch.save(features_per_class, "/media/cqu/D/FXV/R-TPT-main6/features_per_class.pth")
    

    
#     for cls in features_per_class:
#         features_per_class[cls] = torch.stack(features_per_class[cls], dim=0)  # [N_cls, D]
        
    #sigma_img, center_dict = compute_ensemble_covarianceimg(mu_dict,  features_per_class)
    
    ece = compute_ece(all_logits, all_labels, n_bins=15)
    ece_tta = compute_ece(all_logits_tta, all_labels, n_bins=15)
    print(f"ECE: {ece:.4f}")
    print(f"ECE_tta: {ece_tta:.4f}")
    progress.display_summary()
    progress2.display_summary()
    progress4.display_summary()
#     progress6.display_summary()
#     progress8.display_summary()
#     progress10.display_summary()
#     progress12.display_summary()
#     progress14.display_summary()
#     progress16.display_summary()
#     progress18.display_summary()
#     progress20.display_summary()
#     progress22.display_summary()
#     progress24.display_summary()
#     progress26.display_summary()
#     progress28.display_summary()
#     progress30.display_summary()
    
    

    return [top1.avg, tpt1.avg], all_logits, all_logits_tta, all_labels, sigma_img, center_dict, sigma_img_dict
def process_tensor_data(data):
    """处理tensor数据，转换为numpy数组"""
    if torch.is_tensor(data):
        # 如果是tensor，直接转换
        return data.cpu().numpy().flatten()
    elif isinstance(data, list) and len(data) > 0 and torch.is_tensor(data[0]):
        # 如果是tensor列表，提取数值
        values = [item.item() if torch.is_tensor(item) else item for item in data]
        return np.array(values)
    elif isinstance(data, (list, tuple)):
        # 如果是普通列表
        return np.array(data)
    else:
        # 其他情况尝试直接转换
        try:
            return np.array(data).flatten()
        except:
            print(f"Warning: Could not process data type: {type(data)}")
            return np.array([])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='Caltech101')
    parser.add_argument('--dataset_mode', type=str, default='test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('-p', '--print-freq', default=200, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='output_results/ckps/rtpt')

    parser.add_argument('--eps', default=0.0, type=float)
    parser.add_argument('--alpha', default=0.0, type=float)
    parser.add_argument('--steps', type=int, default=0)

    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')

    parser.add_argument('--load_tecoa', type=str, default='', choices=['', 'RN50-eps1', 'ViT-B/32-eps1', 'ViT-B/32-eps4'])

    # Create directory for saving plots
#     save_dir = "/media/cqu/D/FXV/R-TPT-main6/"
#     os.makedirs(save_dir, exist_ok=True)
# 
#     # Assume you have four tensors, each with about 2000 points
#     # Here we use randomly generated example data, replace with your actual data
#     device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# 
#     # Generate example data (replace with your actual tensors)
#     np.random.seed(42)
#     # 加载6个tensor文件（请根据实际路径替换）
#     tensor1 = torch.load('/media/cqu/D/FXV/R-TPT-main6/Caltech101clnbatch.pt')
#     tensor2 = torch.load('/media/cqu/D/FXV/R-TPT-main6/Caltech101batch.pt')
#     tensor3 = torch.load('/media/cqu/D/FXV/R-TPT-main6/dtdclnbatch.pt')
#     tensor4 = torch.load('/media/cqu/D/FXV/R-TPT-main6/dtd1batch.pt')
#     tensor5 = torch.load('/media/cqu/D/FXV/R-TPT-main6/flowerclnbatch.pt')  # 请替换为实际路径
#     tensor6 = torch.load('/media/cqu/D/FXV/R-TPT-main6/flower1batch.pt')  # 请替换为实际路径
# 
#     # 处理数据
#     data1 = process_tensor_data(tensor1)
#     data2 = process_tensor_data(tensor2)
#     data3 = process_tensor_data(tensor3)
#     data4 = process_tensor_data(tensor4)
#     data5 = process_tensor_data(tensor5)
#     data6 = process_tensor_data(tensor6)
# 
#     # 准备绘图数据（6个）
#     data = [data1, data2, data3, data4, data5, data6]
#     labels = ['Clean Samples of Caltech101', 'Adv Samples of Caltech101', 'Clean Samples of DTD', 'Adv Samples of DTD', 'Clean Samples of Flowers102', 'Adv Samples of Flowers102']  # 可根据实际重命名
#     colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']  # 增加两种颜色
# 
#     # 创建KDE图
#     plt.figure(figsize=(14, 7))  # 稍微调大画布以适应6条曲线
# 
#     # 绘制KDE曲线
#     for i, (d, label, color) in enumerate(zip(data, labels, colors)):
#         if len(d) > 0:  # 确保有数据
#             sns.kdeplot(d, label=label, color=color, linewidth=2, alpha=0.7)
#         else:
#             print(f"Warning: No data for {label}")
# 
#     plt.xlabel('Value', fontsize=16)
#     plt.ylabel('Density', fontsize=16)
#     #plt.title('Comparisons of Kernel Density Estimation (KDE)')
#     plt.legend(fontsize=14)
#     plt.tick_params(axis='both', which='major', labelsize=16)  # 刻度标签
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
# 
#     # 保存图表（可根据需要修改文件名）
#     save_path = '/media/cqu/D/FXV/R-TPT-main6/tensor_kde_6.pdf'
#     plt.savefig(save_path, dpi=300)
#     print(f"Chart saved to {save_path}")




    main()
