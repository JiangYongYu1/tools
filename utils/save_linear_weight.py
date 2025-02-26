import torch
import torch.nn as nn
import os

def save_linear_weights(model: nn.Module, name_list: list, save_dir: str):
    """
    遍历模型中的所有线性层，如果层名包含name_list中的任意一个名称，则保存该层的权重。

    参数：
    - model (nn.Module): PyTorch 模型。
    - name_list (list): 包含目标层名称的列表。
    - save_path (str): 保存权重的文件路径（例如，'weights.pt'）。
    """
    # 如果保存目录不存在，则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"创建保存目录: {save_dir}")

    # 遍历模型的所有命名模块
    for name, module in model.named_modules():
        # 检查模块是否是线性层
        if isinstance(module, nn.Linear):
            # 检查层名是否包含name_list中的任意一个名称
            if any(sub_name in name for sub_name in name_list):
                # 提取权重并移动到CPU
                weight = module.weight.detach().cpu()
                
                # 处理层名以确保文件名有效（例如，将点替换为下划线）
                safe_name = name.replace('.', '_').replace('/', '_')
                filename = f"{safe_name}_weight.pt"
                
                # 构建完整的保存路径
                file_path = os.path.join(save_dir, filename)
                
                # 保存权重
                torch.save(weight, file_path)
                print(f"已保存权重: {file_path}")

    print(f"所有符合条件的线性层权重已保存到目录 {save_dir}")

# 示例用法
if __name__ == "__main__":
    # 定义一个示例模型
    class ExampleModel(nn.Module):
        def __init__(self):
            super(ExampleModel, self).__init__()
            self.fc1 = nn.Linear(10, 20)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(20, 30)
            self.fc_special = nn.Linear(30, 40)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.fc_special(x)
            return x

    # 实例化模型
    model = ExampleModel()

    # 定义需要匹配的名称列表
    name_list = ["fc1", "special"]

    # 调用保存函数，保存到 'selected_weights.pt'
    save_linear_weights(model, name_list, "selected_weights.pt")