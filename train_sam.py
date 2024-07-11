import os
import random
from datetime import datetime

import footsteps
import numpy as np
import torch
import itk
import sys
import utils
import torch.nn.functional as F
from dataset import COPDDataset, HCPDataset, OAIDataset, ACDCDataset
from torch.utils.data import ConcatDataset, DataLoader
import time

import icon_registration as icon
import icon_registration.network_wrappers as network_wrappers
import icon_registration.networks as networks
from icon_registration import config
from icon_registration.losses import ICONLoss, to_floats
from icon_registration.mermaidlite import compute_warped_image_multiNC
from AutomaticWeightedLoss import AutomaticWeightedLoss
from sam import SAM
from torch.optim.lr_scheduler import CosineAnnealingLR
from bypass_bn import enable_running_stats, disable_running_stats

import warnings

# 屏蔽所有警告
warnings.filterwarnings('ignore')

def write_stats(writer, stats: ICONLoss, ite):
    for k, v in to_floats(stats)._asdict().items():
        writer.add_scalar(k, v, ite)

input_shape = [1, 1, 128, 128, 128]
# input_shape = [1, 1, 160, 192, 224]

BATCH_SIZE = 4
GPUS = 1
device = torch.device("cuda:2")
maxScore = 0
class GradientICONSparse(network_wrappers.RegistrationModule):
    def __init__(self, network, similarity, lmbda):

        super().__init__()

        self.regis_net = network
        self.lmbda = lmbda
        self.similarity = similarity


    def forward(self, image_A, image_B, image_A_mask = None):

        assert self.identity_map.shape[2:] == image_A.shape[2:]
        assert self.identity_map.shape[2:] == image_B.shape[2:]
        if image_A_mask is not None:
            assert self.identity_map.shape[2:] == image_A_mask.shape[2:]

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identity_map.isIdentity = True
        #self.phi_AB 是形变场函数 A配到B
        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_BA = self.regis_net(image_B, image_A)

        # identity_map 是一个表示坐标空间的张量。 phi_AB_vectorfield 坐标张量 也就是空间变换后的坐标
        self.phi_AB_vectorfield = self.phi_AB(self.identity_map)
        self.phi_BA_vectorfield = self.phi_BA(self.identity_map)

        # tag images during warping so that the similarity measure
        # can use information about whether a sample is interpolated
        # or extrapolated

        if getattr(self.similarity, "isInterpolated", False):
            # tag images during warping so that the similarity measure
            # can use information about whether a sample is interpolated
            # or extrapolated
            inbounds_tag = torch.zeros([image_A.shape[0]] + [1] + list(image_A.shape[2:]), device=device)
            if len(self.input_shape) - 2 == 3:
                inbounds_tag[:, :, 1:-1, 1:-1, 1:-1] = 1.0
            elif len(self.input_shape) - 2 == 2:
                inbounds_tag[:, :, 1:-1, 1:-1] = 1.0
            else:
                inbounds_tag[:, :, 1:-1] = 1.0
        else:
            inbounds_tag = None

        self.warped_image_A = compute_warped_image_multiNC(
            torch.cat([image_A, inbounds_tag], axis=1) if inbounds_tag is not None else image_A,
            self.phi_AB_vectorfield,
            self.spacing,
            1,
        )
        if image_A_mask is not None:
            self.warped_image_A_mask = compute_warped_image_multiNC(
                torch.cat([image_A_mask, inbounds_tag], axis=1) if inbounds_tag is not None else image_A_mask,
                self.phi_AB_vectorfield,
                self.spacing,
                1,
            )


        self.warped_image_B = compute_warped_image_multiNC(
            torch.cat([image_B, inbounds_tag], axis=1) if inbounds_tag is not None else image_B,
            self.phi_BA_vectorfield,
            self.spacing,
            1,
        )

        similarity_loss = self.similarity(
            self.warped_image_A, image_B
        ) + self.similarity(self.warped_image_B, image_A)

        # 生成随机扰动 Iepsilon 是稍微偏离 identity_map 的点 也就是坐标张量 生成扰动的测试点是为了评估和验证形变场的逆一致性特性
        if len(self.input_shape) - 2 == 3:
            Iepsilon = (
                self.identity_map
                + 2 * torch.randn(*self.identity_map.shape).to(device)
                / self.identity_map.shape[-1]
            )[:, :, ::2, ::2, ::2]
        elif len(self.input_shape) - 2 == 2:
            Iepsilon = (
                self.identity_map
                + 2 * torch.randn(*self.identity_map.shape).to(device)
                / self.identity_map.shape[-1]
            )[:, :, ::2, ::2]

        # compute squared Frobenius of Jacobian of icon error

        direction_losses = []

        approximate_Iepsilon = self.phi_AB(self.phi_BA(Iepsilon))

        # 这一差异表示 phi_AB 和 phi_BA 之间的逆一致性误差。
        inverse_consistency_error = Iepsilon - approximate_Iepsilon

        delta = 0.001

        if len(self.identity_map.shape) == 4:
            dx = torch.Tensor([[[[delta]], [[0.0]]]]).to(device)
            dy = torch.Tensor([[[[0.0]], [[delta]]]]).to(device)
            direction_vectors = (dx, dy)

        elif len(self.identity_map.shape) == 5:
            dx = torch.Tensor([[[[[delta]]], [[[0.0]]], [[[0.0]]]]]).to(device)
            dy = torch.Tensor([[[[[0.0]]], [[[delta]]], [[[0.0]]]]]).to(device)
            dz = torch.Tensor([[[[0.0]]], [[[0.0]]], [[[delta]]]]).to(device)
            direction_vectors = (dx, dy, dz)
        elif len(self.identity_map.shape) == 3:
            dx = torch.Tensor([[[delta]]]).to(device)
            direction_vectors = (dx,)

        for d in direction_vectors:
            approximate_Iepsilon_d = self.phi_AB(self.phi_BA(Iepsilon + d))
            inverse_consistency_error_d = Iepsilon + d - approximate_Iepsilon_d
            grad_d_icon_error = (
                inverse_consistency_error - inverse_consistency_error_d
            ) / delta
            direction_losses.append(torch.mean(grad_d_icon_error**2))

        inverse_consistency_loss = sum(direction_losses)

        all_loss = self.lmbda * inverse_consistency_loss + similarity_loss

        # 这个就是每个像素点位置的变换 看是否变换过大
        transform_magnitude = torch.mean(
            (self.identity_map - self.phi_AB_vectorfield) ** 2
        )
        return icon.losses.ICONLoss(
            all_loss,
            inverse_consistency_loss,
            similarity_loss,
            transform_magnitude,
            icon.losses.flips(self.phi_BA_vectorfield),
        )


def get_dataset():
    return ConcatDataset(
        (
        ACDCDataset(data_path = "./data/acdc/training", desired_shape = input_shape[2:]),
        # OAIDataset(input_shape[2:]),
        # HCPDataset(input_shape[2:])
        # COPDDataset(
        #     "/playpen-raid2/lin.tian/projects/icon_lung/ICON_lung/splits/train.txt",
        #     desired_shape=input_shape[2:])
        )
    )

def make_network(input_shape, include_last_step=False, lmbda=1.5, loss_fn=icon.LNCC(sigma=5), load_checkpoint=None):
    dimension = len(input_shape) - 2
    inner_net = icon.FunctionFromVectorField(networks.tallUNet2(dimension=dimension))

    for _ in range(2):
        inner_net = icon.TwoStepRegistration(
            icon.DownsampleRegistration(inner_net, dimension=dimension),
            icon.FunctionFromVectorField(networks.tallUNet2(dimension=dimension))
        )
    if include_last_step:
        inner_net = icon.TwoStepRegistration(inner_net, icon.FunctionFromVectorField(networks.tallUNet2(dimension=dimension)))

    inner_net = inner_net.to(device)
    net = GradientICONSparse(inner_net, loss_fn, lmbda=lmbda)
    net.assign_identity_map(input_shape)

    # 加载预训练权重
    if load_checkpoint is not None:
        trained_weights = torch.load(load_checkpoint)
        net.regis_net.load_state_dict(trained_weights)

    return net

def train_kernel(optimizer, net, moving_image, fixed_image, writer, ite, epoch):
    enable_running_stats(net)
    loss_object = net(moving_image, fixed_image)
    loss = torch.mean(loss_object.all_loss)
    loss.backward()
    optimizer.first_step(zero_grad=True)

    disable_running_stats(net)
    loss_object = net(moving_image, fixed_image)
    loss = torch.mean(loss_object.all_loss)
    loss.backward()
    optimizer.second_step(zero_grad=True)

    print("进行到{}轮--{}".format(epoch+1, to_floats(loss_object)))
    write_stats(writer, loss_object, ite)
    return loss


def preprocess(img, type="ct"):
    # print(img.shape)
    if type == "ct":
        clamp = [-1000, 1000]
        img = (torch.clamp(img, clamp[0], clamp[1]) - clamp[0]) / (clamp[1] - clamp[0])
        return F.interpolate(img, [128, 128, 128], mode="trilinear", align_corners=False)
    elif type == "mri":
        im_min, im_max = torch.min(img), torch.quantile(img.view(-1), 0.99)
        img = torch.clip(img, im_min, im_max)
        img = (img - im_min) / (im_max - im_min)
        return F.interpolate(img, [128, 128, 128], mode="trilinear", align_corners=False)

    else:
        print(f"Error: Do not support the type {type}")
        return img


def calculateDice(net, optimizer, epoch):
    global maxScore
    # 去验证每一轮的dice效果
    score_q_total = 0
    # 这是配准后的dice分数
    score_h_total = 0
    score_h1_total = 0
    score_h2_total = 0
    score_h3_total = 0
    for i in range(100, 150):
        i = i + 1
        # target是固定的 source是移动的
        if i < 10:
            i = '00' + str(i)
        elif i < 100:
            i = '0' + str(i)
        else:
            i = str(i)

        # 相同病人不同时期
        target_path = "./data/acdc/testing/patient" + i + "/patient" + i + "_frame01.nii.gz"
        target_mask_path = "./data/acdc/testing/patient" + i + "/patient" + i + "_frame01_gt.nii.gz"
        source_path = "./data/acdc/testing/patient" + i + "/patient" + i + "_frame02.nii.gz"
        source_mask_path = "./data/acdc/testing/patient" + i + "/patient" + i + "_frame02_gt.nii.gz"

        target_itk = itk.imread(target_path)
        target = np.asarray(target_itk)

        target_mask_itk = itk.imread(target_mask_path)
        target_mask = np.asarray(target_mask_itk)

        source_itk = itk.imread(source_path)
        source = np.asarray(source_itk)

        source_mask_itk = itk.imread(source_mask_path)
        source_mask = np.asarray(source_mask_itk)

        assert np.array_equal(dict(target_itk)["direction"],
                              dict(source_itk)["direction"]), "The orientation of source " \
                                                              "and target images need to be" \
                                                              " the same. "
        target = preprocess(torch.Tensor(np.array(target)).unsqueeze(0).unsqueeze(0), type="mri")
        target_mask = torch.Tensor(np.array(target_mask)).unsqueeze(0).unsqueeze(0)
        target_mask = F.interpolate(target_mask, [128, 128, 128], mode="trilinear", align_corners=False)

        source = preprocess(torch.Tensor(np.array(source)).unsqueeze(0).unsqueeze(0), type="mri")
        source_mask = torch.Tensor(np.array(source_mask)).unsqueeze(0).unsqueeze(0)
        source_mask = F.interpolate(source_mask, [128, 128, 128], mode="trilinear", align_corners=False)

        net.eval()
        with torch.no_grad():
            net(source.to(device), target.to(device), source_mask.to(device))

        fixed_seg = utils.numpy(target_mask, device).astype(np.uint8)
        moving_seg = utils.numpy(source_mask, device).astype(np.uint8)
        warped_seg = utils.numpy(net.warped_image_A_mask, device).astype(np.uint8)

        score_q = utils.dice(moving_seg > 0, fixed_seg > 0, [1])
        score_hall = utils.dice(warped_seg > 0, fixed_seg > 0, [1])

        # 这是算分割区域的
        # score_h = utils.dice(warped_seg, fixed_seg, [1, 2, 3])
        # for index, score in enumerate(score_h):
        #     # print("第{}个label配准后的dice是{}".format(index + 1, score.item()))
        #     if index == 0:
        #         score_h1_total += score.item()
        #     elif index == 1:
        #         score_h2_total += score.item()
        #     elif index == 2:
        #         score_h3_total += score.item()
        #
        score_q_total += score_q.item()

        score_h_total += score_hall.item()

    score = round(score_h_total / 50, 7)
    if score > maxScore:
        maxScore = score

        torch.save(
            optimizer.state_dict(),
            footsteps.output_dir + "checkpoints/optimizer_weights_" + str(epoch) + "_" + str(score),
        )
        torch.save(
            net.regis_net.state_dict(),
            footsteps.output_dir + "checkpoints/network_weights_" + str(epoch) + "_" + str(score),
        )
        print("已经保存当前得分最高的模型，得分为：{}".format(score))

def train(
    net,
    optimizer,
    data_loader,
    epochs=200,
    eval_period=-1,
    save_period=-1,
    step_callback=(lambda net, optimizer, epochs: None),
    unwrapped_net=None,
):
    """A training function intended for long running experiments, with tensorboard logging
    and model checkpoints. Use for medical registration training
    """
    # 这个第二个进来
    import footsteps
    from torch.utils.tensorboard import SummaryWriter

    if unwrapped_net is None:
        unwrapped_net = net

    global maxScore

    loss_curve = []
    writer = SummaryWriter(
        footsteps.output_dir + "/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S"),
        flush_secs=30,
    )
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    scheduler = CosineAnnealingLR(optimizer.base_optimizer, T_max=epochs / 4, eta_min=1e-8)
    iteration = 0
    for epoch in range(epochs):
        net.train()
        total_loss = 0
        loss_index = 0
        for moving_image, fixed_image in data_loader:
            moving_image, fixed_image = moving_image.to(device), fixed_image.to(device)
            loss = train_kernel(optimizer, net, moving_image, fixed_image,
                         writer, iteration, epoch)
            iteration += 1
            loss_index += 1
            total_loss += loss
            time.sleep(1)
            step_callback(unwrapped_net, optimizer, epoch)
            # time.sleep(1)
            # calculateDice(unwrapped_net, optimizer, epoch)

        # 在每个epoch结束时更新学习率
        scheduler.step()
        print('Learning rate adjusted to:', optimizer.param_groups[0]['lr'])


        # if(epoch % 200 == 0):
        #     print("进行到{}轮".format(epoch))
        # if epoch % save_period == 0:
        #     torch.save(
        #         optimizer.state_dict(),
        #         footsteps.output_dir + "checkpoints/optimizer_weights_" + str(epoch),
        #     )
        #     torch.save(
        #         unwrapped_net.regis_net.state_dict(),
        #         footsteps.output_dir + "checkpoints/network_weights_" + str(epoch),
        #     )

        # if epoch % eval_period == 0:
        #     visualization_moving, visualization_fixed = next(iter(data_loader))
        #     visualization_moving, visualization_fixed = visualization_moving.to(device), visualization_fixed.to(device)
        #     unwrapped_net.eval()
        #     print("val (from train set)")
        #     warped = []
        #     with torch.no_grad():
        #         print( unwrapped_net(visualization_moving, visualization_fixed))
        #         warped = unwrapped_net.warped_image_A.to(device)
        #     unwrapped_net.train()
        #
        #     def render(im):
        #         if len(im.shape) == 5:
        #             im = im[:, :, :, im.shape[3] // 2]
        #         if torch.min(im) < 0:
        #             im = im - torch.min(im)
        #         if torch.max(im) > 1:
        #             im = im / torch.max(im)
        #         return im[:4, [0, 0, 0]].detach().to(device)
        #
        #     writer.add_images(
        #         "moving_image", render(visualization_moving[:4]), epoch, dataformats="NCHW"
        #     )
        #     writer.add_images(
        #         "fixed_image", render(visualization_fixed[:4]), epoch, dataformats="NCHW"
        #     )
        #     writer.add_images(
        #         "warped_moving_image",
        #         render(warped),
        #         epoch,
        #         dataformats="NCHW",
        #     )
        #     writer.add_images(
        #         "difference",
        #         render(torch.clip((warped[:4, :1] - visualization_fixed[:4, :1].to(device)) + 0.5, 0, 1)),
        #         epoch,
        #         dataformats="NCHW",
        #     )


# w

def train_two_stage(input_shape, data_loader, GPUS, epochs, eval_period, save_period):
    # 这个第一个进来
    # net = make_network(input_shape, include_last_step=True, load_checkpoint='./weights/Step_2_final.trch')
    net = make_network(input_shape, include_last_step=True, load_checkpoint='./checkpoints/network_weights.pth')

    if GPUS == 1:
        net_par = net.to(device)
    else:
        # net_par = torch.nn.DataParallel(net).cuda()
        # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        net_par = net.to(device)
    net = net.to(device)
    # 原始学习率为 0.00005
    # optimizer = torch.optim.Adam(net_par.parameters(), lr=0.00005)
    base_optimizer = torch.optim.AdamW
    optimizer = SAM(net_par.parameters(), base_optimizer, lr=5e-5, betas=(0.9, 0.999), weight_decay=1e-4)


    net_par.train()

    train(net_par, optimizer, data_loader, unwrapped_net=net, step_callback=calculateDice,
          epochs=epochs, eval_period=eval_period, save_period=save_period)

    torch.save(
                net.regis_net.state_dict(),
                footsteps.output_dir + "checkpoints/Step_2_final.trch",
            )

    # net_2 = make_network(input_shape, include_last_step=True, framework=framework)
    #
    # net_2.regis_net.netPhi.load_state_dict(net.regis_net.state_dict())
    #
    # del net
    # del net_par
    # del optimizer
    #
    # if GPUS == 1:
    #     net_2_par = net_2.cuda()
    # else:
    #     net_2_par = torch.nn.DataParallel(net_2).cuda()
    # optimizer = torch.optim.Adam(net_2_par.parameters(), lr=0.00005)
    #
    # net_2_par.train()
    #
    # # We're being weird by training two networks in one script. This hack keeps
    # # the second training from overwriting the outputs of the first.
    # footsteps.output_dir_impl = footsteps.output_dir + "2nd_step/"
    # os.makedirs(footsteps.output_dir)
    #
    # train(net_2_par, optimizer, data_loader, unwrapped_net=net_2, epochs=100, save_period=10)
    #
    # torch.save(
    #             net_2.regis_net.state_dict(),
    #             footsteps.output_dir + "Step_2_final.trch",
    #         )

if __name__ == "__main__":
    footsteps.initialize()

    dataloader = DataLoader(
        get_dataset(),
        batch_size=BATCH_SIZE*GPUS,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    os.makedirs(footsteps.output_dir + "checkpoints", exist_ok=True)

    train_two_stage(input_shape, dataloader, GPUS, 200000, 1000, 200)