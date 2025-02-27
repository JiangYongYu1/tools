import torch.nn.functional as F
import torch
import os
import matplotlib.pyplot as plt
import glob

# 保存原始的 F.softmax
_orig_softmax = F.softmax

def logging_softmax(x, dim, dtype=torch.float32, filter_mask=True, output_dir="output/deepseek_distill_qwen_7b"):
    # 创建一个用于分析的 tensor，
    if filter_mask:
        mask = torch.isinf(x) if torch.isinf(x).any() else (x == torch.finfo(x.dtype).min)
        x_for_logging = x.masked_fill(mask, 0.0)
    else:
        x_for_logging = x
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_max = x_for_logging.max().item()
    # output_std = x_for_logging.std().item()
    print(f"[monkey-patch] Softmax input -> max: {output_max:.4f}")
    subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    new_folder_name = str(len(subdirs))
    new_folder_path = os.path.join(output_dir, new_folder_name)
    os.makedirs(new_folder_path)
    try:    
        num_heads = x_for_logging.size(1)
        for head in range(num_heads):
            head_data = x_for_logging[0, head].detach().cpu().float().numpy()  # shape: [seq_len, key_len]
            plt.figure(figsize=(6, 4))
            plt.imshow(head_data, aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.title(f"Head {head} Distribution (2D)")
            plt.xlabel("Key Length")
            plt.ylabel("Sequence Length")
            plt.tight_layout()
            plt.savefig(f"{new_folder_path}/head_{head}_dist.png")
            plt.close()
    except Exception as e:
        print(f"Failed to plot attention distribution: {e}")
        
    # 调用原始 softmax
    out = _orig_softmax(x, dim=dim, dtype=dtype).to(x.dtype)
    
    # # 分析输出分布
    # output_mean = out.mean().item()
    # output_std = out.std().item()
    # print(f"[monkey-patch] Softmax output -> mean: {output_mean:.4f}, std: {output_std:.4f}")
    
    return out

# 进行 monkey-patch
F.softmax = logging_softmax