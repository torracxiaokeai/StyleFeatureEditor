import torch
ckpt = torch.load("/root/avatar_project/StyleFeatureEditor/experiments/fse_inverter_train_003/iteration_29000.pt", map_location='cpu')
print(ckpt.keys())  # 查看顶层键
if 'state_dict' in ckpt:
    print(list(ckpt['state_dict'].keys()))  # 查看encoder中的部分键
