import os
import random
from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.paths = sorted(data_utils.make_dataset(root))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image


class ViVFaceDataset(Dataset):
    """
    用于ViVFace训练的数据集类，从CelebV-HQ格式的数据集中加载：
    1. S: 源图像
    2. D1: 与S相同身份的不同表情/姿态图像
    3. D2: 不同身份的图像
    
    目录结构应为:
    - root/
      - identity_00001/
        - frame_00001.jpg
        - frame_00002.jpg
        - ...
      - identity_00002/
        - frame_00001.jpg
        - ...
    """
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        
        # 获取所有身份文件夹
        self.identity_dirs = []
        for d in os.listdir(root):
            if os.path.isdir(os.path.join(root, d)) and d.startswith('identity_'):
                self.identity_dirs.append(d)
        
        # 根据身份目录创建映射
        self.identity_to_images = {}
        for identity in self.identity_dirs:
            identity_path = os.path.join(root, identity)
            # 筛选frame_XXXXX.jpg格式的文件
            images = []
            for f in os.listdir(identity_path):
                if f.startswith('frame_') and (f.endswith('.jpg') or f.endswith('.png')):
                    images.append(os.path.join(identity_path, f))
            
            if len(images) >= 2:  # 确保每个身份至少有两张图像
                # 按照帧号排序
                images.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
                self.identity_to_images[identity] = images
        
        # 创建数据集索引到身份的映射
        self.valid_identities = list(self.identity_to_images.keys())
        self.dataset_indices = []
        for identity in self.valid_identities:
            images = self.identity_to_images[identity]
            for i in range(len(images)):
                self.dataset_indices.append((identity, i))
        
        print(f"ViVFaceDataset加载完成: {len(self.dataset_indices)}个样本，{len(self.valid_identities)}个不同身份")
    
    def __len__(self):
        return len(self.dataset_indices)
    
    def __getitem__(self, index):
        # 获取S（源图像）
        identity_s, idx_s = self.dataset_indices[index]
        images_s = self.identity_to_images[identity_s]
        path_s = images_s[idx_s]
        
        # 获取D1（同身份不同表情/姿态）
        # 从同一个身份中随机选择另一张图像
        remaining_indices = [i for i in range(len(images_s)) if i != idx_s]
        if not remaining_indices:  # 如果没有其他图像，则重复使用当前图像
            idx_d1 = idx_s
        else:
            idx_d1 = random.choice(remaining_indices)
        path_d1 = images_s[idx_d1]
        
        # 获取D2（不同身份）
        # 随机选择不同的身份
        other_identities = [id for id in self.valid_identities if id != identity_s]
        identity_d2 = random.choice(other_identities)
        images_d2 = self.identity_to_images[identity_d2]
        path_d2 = random.choice(images_d2)
        
        # 加载并转换所有图像
        image_s = Image.open(path_s).convert("RGB")
        image_d1 = Image.open(path_d1).convert("RGB")
        image_d2 = Image.open(path_d2).convert("RGB")
        
        if self.transform:
            image_s = self.transform(image_s)
            image_d1 = self.transform(image_d1)
            image_d2 = self.transform(image_d2)
        
        # 返回字典格式的批次数据
        batch = {
            'source': image_s,        # S: 源图像
            'same_id': image_d1,      # D1: 同身份不同表情
            'diff_id': image_d2,      # D2: 不同身份
            'source_path': path_s,
            'same_id_path': path_d1,
            'diff_id_path': path_d2
        }
        
        return batch


class CelebaAttributeDataset(Dataset):
    def __init__(self, images_root, attr, transform=None, attributes_root="", use_attr=True):
        self.paths = data_utils.make_dataset(images_root)
        self.transform = transform
        with open(attributes_root, "r") as f:
            lines = f.readlines()

        attr_num = -1
        for i, data_attr in enumerate(lines[1].split(" ")):
            if data_attr.strip() == attr.strip():
                attr_num = i
                break
        assert attr_num > -1, f"Can not find attribute {attr}"

        filtred_paths = []
        for path in self.paths:
            pic_num = int(path.split("/")[-1].replace(".jpg", "").replace(".png", "")) 
            pic_attrs = lines[pic_num + 2].strip().split(" ")
            pic_attrs = pic_attrs[2:]
            if use_attr and pic_attrs[attr_num] == "1" or not use_attr and pic_attrs[attr_num] == "-1":
                filtred_paths.append(path)
        self.paths = sorted(filtred_paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from_path = self.paths[index]
        image = Image.open(from_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image


class FIDDataset(Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        file = self.files[i]
        image = file.convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)

        return image


class MetricsPathsDataset(Dataset):
    def __init__(self, root_path, gt_dir=None, transform=None, transform_train=None, return_path=False, ignore=[]):
        self.pairs = []
        self.paths = []
        self.names = []

        for f in os.listdir(root_path):
            if f not in ignore:
                self.names.append(f)
                image_path = os.path.join(root_path, f)
                gt_path = os.path.join(gt_dir, f)
                if f.endswith(".jpg") or f.endswith(".png"):
                    self.pairs.append([image_path, gt_path.replace(".png", ".jpg"), None])
                    self.paths.append(image_path)
        self.transform = transform
        self.transform_train = transform_train
        self.return_path = return_path

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        from_path, to_path, _ = self.pairs[index]
        from_im = Image.open(from_path).convert("RGB")
        to_im = Image.open(to_path).convert("RGB")

        if self.transform:
            to_im = self.transform(to_im)
            from_im = self.transform(from_im)

        if not self.return_path:
            return from_im, to_im
        else:
            return from_im, to_im, self.names[index]


class MetricsDataDataset(Dataset):
    def __init__(
        self, paths, target_data, fake_data, transform=None, transform_train=None
    ):
        self.fake_data = fake_data
        self.target_data = target_data
        self.paths = paths
        self.transform = transform
        self.transform_train = transform_train

    def __len__(self):
        return len(self.fake_data)

    def __getitem__(self, index):

        target_im = self.target_data[index]
        fake_im = self.fake_data[index]

        if self.transform:
            fake_im = self.transform(fake_im)
            target_im = self.transform(target_im)

        return target_im, fake_im
