"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import torch
import torch.distributed.nn
from torch import distributed as dist, nn as nn
import numpy as np
from torch.nn import functional as F
from torch.nn.functional import cosine_similarity
try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
    image_features,
    text_features,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
    use_horovod=False,
):
    if use_horovod:
        assert hvd is not None, "Please install horovod"
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(
                    all_image_features.chunk(world_size, dim=0)
                )
                gathered_text_features = list(
                    all_text_features.chunk(world_size, dim=0)
                )
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(
                torch.distributed.nn.all_gather(image_features), dim=0
            )
            all_text_features = torch.cat(
                torch.distributed.nn.all_gather(text_features), dim=0
            )
        else:
            gathered_image_features = [
                torch.zeros_like(image_features) for _ in range(world_size)
            ]
            gathered_text_features = [
                torch.zeros_like(text_features) for _ in range(world_size)
            ]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                self.local_loss,
                self.gather_with_grad,
                self.rank,
                self.world_size,
                self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = (
                    logit_scale * all_image_features @ all_text_features.T
                )
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2
        return total_loss

class GroupLoss(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
        top_k = 10,
        cos_batch = 256,
    ):
        """
        top_k: 筛选前top_k个相似度来进行后续计算
        cos_batch: 为了节省显存
        """
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.cos_batch = cos_batch
        self.top_k = top_k

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def _get_similarity(self, text_features):
        """
        text_features: (batch, dim)
        return: (batch, batch) 代表索引之间的相似度关系
        """

        def compute_similarity_in_batches(A, B, batch_size):
            N = A.size(0)
            similarity_matrix = torch.zeros(N, N, device=A.device)

            for i in range(0, N, batch_size):
                end_i = min(i + batch_size, N)
                A_sub = A[i:end_i]

                for j in range(0, N, batch_size):
                    end_j = min(j + batch_size, N)
                    B_sub = B[j:end_j]

                    # 计算子矩阵之间的相似度
                    sub_similarity = cosine_similarity(A_sub.unsqueeze(1), B_sub.unsqueeze(0), dim=2)

                    # 将子矩阵的相似度填充到完整矩阵中
                    similarity_matrix[i:end_i, j:end_j] = sub_similarity

            return similarity_matrix


        cosine_sims = compute_similarity_in_batches(text_features, text_features, batch_size=self.cos_batch)
        return cosine_sims

    def _get_class_mask(self, labels):
        # 为 labels 添加额外的维度以支持广播
        labels_a = labels.unsqueeze(0)  # 形状变为 (1, batch)
        labels_b = labels.unsqueeze(1)  # 形状变为 (batch, 1)

        # 比较每一个标签与所有其他的标签
        mask = labels_a == labels_b

        # 将 bool 类型的掩码转换为 int 类型的矩阵
        similarity_matrix = mask.int()
        return similarity_matrix

    def forward(self, image_features, text_features, logit_scale, img_index):
        """
        todo:
        现在问题：相似都很相似
        """
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                self.local_loss,
                self.gather_with_grad,
                self.rank,
                self.world_size,
                self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = (
                    logit_scale * all_image_features @ all_text_features.T
                )
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        with torch.no_grad():
            # Get similarity matrix
            sim = self._get_similarity(text_features)

            # Make top5 has values, others become 0
            values, indices = torch.topk(sim.clone().fill_diagonal_(0),  # 克隆一个sim 让其对角线元素为0
                                         k=self.top_k if self.top_k < sim.shape[0] else sim.shape[0],
                                         dim=1,
                                         sorted=False)
            mask_sim = torch.zeros_like(sim)
            mask_sim.scatter_(1, indices, 1).fill_diagonal_(1)

            # Make mask_class by labels
            mask_class = self._get_class_mask(img_index)

            # Enable mask
            sim = (sim * mask_sim) * mask_class
            # Normalization
            row_sums = sim.sum(dim=1, keepdim=True)
            labels = sim / row_sums

        # 2: 一个image有多个eeg可对应
        total_loss = (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)# todo 在这加一个转秩？
        ) / 2
        return total_loss
    
def soft_clip_loss(preds, targs, logit_scale):
    clip_clip = (targs @ targs.T)*logit_scale
    brain_clip = (preds @ targs.T)*logit_scale
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss
