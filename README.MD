# 1. 常用代码整理

## 1.1 samples
不能直接用，参考代码备份
### 1.1.1 function_rewriter
    新写一个函数替换原来类的forward, 使用types.MethodType

## 1.2 常用工具，可以直接用
### 1.2.1 save_linear_weight.py
    
    遍历模型中的所有线性层，如果层名包含name_list中的任意一个名称，则保存该层的权重。

### 1.2.2 

