import torch
import numpy as np

def get_batch(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从一维 token 数组中随机采样 batch_size 个片段。
    """
    max_start_idx = len(x) - context_length
    ix = np.random.randint(0, max_start_idx, size=(batch_size,))

    # 提取切片
    inputs_list = [x[i : i + context_length] for i in ix]
    targets_list = [x[i + 1 : i + context_length + 1] for i in ix]

    # 先用 .astype(np.int64) 桥接，再转为 PyTorch LongTensor
    inputs_np = np.array(inputs_list).astype(np.int64)
    targets_np = np.array(targets_list).astype(np.int64)

    inputs = torch.from_numpy(inputs_np).to(device)
    targets = torch.from_numpy(targets_np).to(device)

    return inputs, targets