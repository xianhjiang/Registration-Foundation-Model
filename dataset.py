
import numpy as np
import torch
import os
import torch.nn.functional as F
import torchvision.transforms.functional as Fv
from torchvision import transforms
import tqdm
import itk
import glob
import monai
from monai.transforms import CropForeground, SpatialPad, ResizeWithPadOrCrop

class HCPDataset(torch.utils.data.Dataset):
    def __init__(self, desired_shape=None) -> None:
        super().__init__()
        with open(f"../brain_t1_pipeline/splits/train.txt") as f:
            self.image_paths = f.readlines()
        self.transform = [CropForeground(lambda x: x>0)]
        if desired_shape is not None:
            self.transform.append(SpatialPad(desired_shape))
    
    def __len__(self):
        return len(self.image_paths)

    def process(self, iA, isSeg=False):
        iA = iA[None, None, :, :, :]
        iA = torch.nn.functional.avg_pool3d(iA, 2)[0]
        iA = iA / torch.max(iA)
        for t in self.transform:
            iA = t(iA)
        return iA

    def __getitem__(self, idx):
        img_1 = self.image_paths[idx].split(".nii.gz")[0] + "_restore_brain.nii.gz"
        img_2 = self.image_paths[np.random.randint(0, len(self.image_paths))].split(".nii.gz")[0] + "_restore_brain.nii.gz"
        
        images = []
        for f_name in [img_1, img_2]:
            image = torch.tensor(np.asarray(itk.imread(f_name.replace("playpen-raid2/Data", "playpen-ssd/lin.tian/data_local"))))
            images.append(self.process(image))
        return images
    
class OAIDataset(torch.utils.data.Dataset):
    def __init__(self, desired_shape=None) -> None:
        super().__init__()
        with open(f"../oai_paper_pipeline/splits/train/pair_path_list.txt") as f:
            train_pair_paths = f.readlines()
            knee_image_paths_set = set()
            knee_image_paths = []
            for p in train_pair_paths:
                p_s = p.split()
                if p_s[0] not in knee_image_paths_set:
                    knee_image_paths.append([p_s[0], p_s[2]])
                    knee_image_paths_set.add(p_s[0])
                if p_s[1] not in knee_image_paths_set:
                    knee_image_paths.append([p_s[1], p_s[3]])
                    knee_image_paths_set.add(p_s[1])
            self.img_paths = knee_image_paths
        self.transform = None
        if desired_shape is not None:
            self.transform = ResizeWithPadOrCrop(desired_shape)
    
    
    def __len__(self):
        return len(self.img_paths)

    def process(self, iA):
        iA = iA[None, None, :, :, :]
        iA = torch.nn.functional.avg_pool3d(iA, 2)[0]
        iA = self.transform(iA) if self.transform is not None else iA
        return iA

    def __getitem__(self, idx):
        img_1 = self.img_paths[idx]
        img_2 = self.img_paths[np.random.randint(0, len(self.img_paths))]

        images = []
        for f_name in [img_1, img_2]:
            f_img, f_img_seg = [f.replace("playpen/zhenlinx/Data", "playpen-ssd/lin.tian/data_local") for f in f_name]
            img = torch.tensor(np.asarray(itk.imread(f_img)))
            # img_seg = torch.tensor(np.asarray(itk.imread(f_img_seg)))
            if "RIGHT" in f_img:
                img = torch.flip(img, [0])
                # img_seg = torch.flip(img_seg, [0])
            elif "LEFT" in f_img:
                pass
            else:
                raise AssertionError()

            images.append(self.process(img))
        return images

class COPDDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ids_file="/playpen-raid2/lin.tian/projects/icon_lung/ICON_lung/splits",
        data_path="/playpen-ssd/lin.tian/data_local/Lung_Registration_transposed/",
        data_num=-1,
        desired_shape=None
    ):
        with open(ids_file) as f:
            self.pair_paths = f.readlines()
            self.pair_paths = list(map(lambda x: x[:-1], self.pair_paths))
        self.data_path = data_path
        self.data_num = data_num
        self.transform = [CropForeground(lambda x: x>0)]
        if desired_shape is not None:
            self.transform.append(SpatialPad(desired_shape))

    def __len__(self):
        return len(self.pair_paths) if self.data_num < 0 else self.data_num

    def process(self, iA, isSeg=False):
        iA = iA[None, None, :, :, :]
        # SI flip
        iA = torch.flip(iA, dims=(2,))
        if isSeg:
            iA = iA.float()
            iA[iA > 0] = 1
            iA = torch.nn.functional.avg_pool3d(iA, 2)
        else:
            iA = iA.float()
            iA = torch.clip(iA, -1000, 0) + 1000.0
            iA = iA / 1000.0
            iA = torch.nn.functional.avg_pool3d(iA, 2)
        return iA

    def __getitem__(self, idx):
        case_id = self.pair_paths[idx]
        image_insp = self.process(
            torch.tensor(
                np.asarray(
                    itk.imread(
                        glob.glob(
                            self.data_path
                            + case_id
                            + "/"
                            + case_id
                            + "_INSP_STD*_COPD_img.nii.gz"
                        )[0]
                    )
                )
            )
        )
        image_exp = self.process(
            torch.tensor(
                np.asarray(
                    itk.imread(
                        glob.glob(
                            self.data_path
                            + case_id
                            + "/"
                            + case_id
                            + "_EXP_STD*_COPD_img.nii.gz"
                        )[0]
                    )
                )
            )
        )

        seg_insp = self.process(
            torch.tensor(
                np.asarray(
                    itk.imread(
                        glob.glob(
                            self.data_path
                            + case_id
                            + "/"
                            + case_id
                            + "_INSP_STD*_COPD_label.nii.gz"
                        )[0]
                    )
                )
            ),
            isSeg=True,
        )
        seg_exp = self.process(
            torch.tensor(
                np.asarray(
                    itk.imread(
                        glob.glob(
                            self.data_path
                            + case_id
                            + "/"
                            + case_id
                            + "_EXP_STD*_COPD_label.nii.gz"
                        )[0]
                    )
                )
            ),
            isSeg=True,
        )

        images = [image_insp[0] * seg_insp[0], image_exp[0] * seg_exp[0]]
        for t in self.transform:
            images[0] = t(images[0])
            images[1] = t(images[1])

        return images


class ACDCDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, desired_shape=None):
        """
        初始化 ACDC 数据集类。

        :param data_path: 数据集所在的路径。
        :param desired_shape: 图像的期望形状，如果为 None，则保持原始图像形状。
        """
        self.data_path = data_path
        self.desired_shape = desired_shape
        self.patient_dirs = [os.path.join(data_path, d) for d in os.listdir(data_path) if
                             os.path.isdir(os.path.join(data_path, d))]
        self.transform = [CropForeground(lambda x: x > 0)]
        if desired_shape is not None:
            self.transform.append(SpatialPad(desired_shape))

    def __len__(self):
        """
        返回数据集中的病人数量。
        """
        return len(self.patient_dirs)

    def process(self, image):
        """
        对图像进行预处理。

        :param image: 输入图像。
        :return: 预处理后的图像。
        """
        # print("处理之前的形状-{}".format(image.shape))
        image = image[None, None, :, :, :]
        image = torch.nn.functional.avg_pool3d(image.float(), 2)[0]
        image = image / torch.max(image)
        i = 0
        for t in self.transform:
            i = i + 1
            image = t(image)

        # if image.shape == (1, 128, 128, 128):
        #     print("等于 这时候进入了{}次".format(i))
        # else:
        #     print("这时候不等于了 进入了{}次 路径是-{}--{}".format(i, self.currentPath, image.shape))
        #     # 目标尺寸：希望调整为 [1, 175, 175, 175]。
        #     target_size = (128, 128, 128)
        #
        #     # 使用 interpolate 来调整图像尺寸
        #     image = F.interpolate(image, size=target_size, mode='trilinear', align_corners=False)


        return image

    def __getitem__(self, idx):
        """
        获取给定索引的固定图像和移动图像。

        :param idx: 索引。
        :return: 固定图像和移动图像的元组。
        """
        # 获取病人文件夹路径
        patient_dir = self.patient_dirs[idx]
        self.currentPath = patient_dir

        # 初始化固定图像和移动图像的路径
        fixed_image_path = None
        moving_image_path = None

        # 遍历病人文件夹中的文件
        for file in os.listdir(patient_dir):
            file_path = os.path.join(patient_dir, file)
            if file.endswith(".nii.gz"):
                # 如果文件名包含 "frame01" 作为固定图像
                if "frame01.nii.gz" in file:
                    fixed_image_path = file_path
                # 如果文件名包含 "frame02" 作为移动图像
                elif "frame02.nii.gz" in file:
                    moving_image_path = file_path

        # 读取固定图像和移动图像
        fixed_image = torch.tensor(np.asarray(itk.imread(fixed_image_path)))
        moving_image = torch.tensor(np.asarray(itk.imread(moving_image_path)))

        # 对固定图像和移动图像进行预处理
        fixed_image = self.process(fixed_image)
        moving_image = self.process(moving_image)

        return fixed_image, moving_image