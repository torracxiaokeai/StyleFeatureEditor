import torch
ckpt = torch.load("/root/avatar_project/StyleFeatureEditor/experiments/vivface_fse_editor_train_3_011/iteration_16000.pt", map_location='cpu')
print(ckpt.keys())  # 查看顶层键
if 'vivface_model' in ckpt:
    print(list(ckpt['vivface_model'].keys()))  # 查看encoder中的部分键
