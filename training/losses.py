import torch
import torch.nn.functional as F

from torch import nn
from criteria import id_loss, moco_loss, id_vit_loss
from criteria.lpips.lpips import LPIPS
from utils.class_registry import ClassRegistry
from configs.paths import DefaultPaths


losses = ClassRegistry()
adv_losses = ClassRegistry()
disc_losses = ClassRegistry()
other_losses = ClassRegistry()


class LossBuilder:
    def __init__(self, enc_losses_dict, disc_losses_dict, device):
        self.coefs_dict = enc_losses_dict
        self.losses_names = [k for k, v in enc_losses_dict.items() if v > 0]
        self.losses = {}
        self.adv_losses = {}
        self.other_losses = {}
        self.device = device

        for loss in self.losses_names:
            if loss in losses.classes.keys():
                self.losses[loss] = losses[loss]().to(self.device).eval()
            elif loss in adv_losses.classes.keys():
                self.adv_losses[loss] = adv_losses[loss]()
            elif loss in other_losses.classes.keys():
                self.other_losses[loss] = other_losses[loss]()
            else:
                raise ValueError(f'Unexepted loss: {loss}')

        self.disc_losses = []
        for loss_name, loss_args in disc_losses_dict.items():
            if loss_args.coef > 0:
                self.disc_losses.append(disc_losses[loss_name](**loss_args))


    def encoder_loss(self, batch_data):
        loss_dict = {}
        global_loss = 0.0

        for loss_name, loss in self.losses.items():
            loss_val = loss(batch_data["y_hat"], batch_data["x"])
            global_loss += self.coefs_dict[loss_name] * loss_val
            loss_dict[loss_name] = float(loss_val)

        for loss_name, loss in self.other_losses.items():
            loss_val = loss(batch_data)
            assert torch.isfinite(loss_val)
            global_loss += self.coefs_dict[loss_name] * loss_val
            loss_dict[loss_name] = float(loss_val)

        if batch_data["use_adv_loss"]:
            for loss_name, loss in self.adv_losses.items():
                loss_val = loss(batch_data["fake_preds"])
                global_loss += self.coefs_dict[loss_name] * loss_val
                loss_dict[loss_name] = float(loss_val)

        return global_loss, loss_dict

    def disc_loss(self, D, batch_data):
        disc_losses = {}
        total_disc_loss = torch.tensor([0.], device=self.device)

        for loss in self.disc_losses:
            disc_loss, disc_loss_dict = loss(D, batch_data)

            total_disc_loss += disc_loss
            disc_losses.update(disc_loss_dict)

        return total_disc_loss, disc_losses



@losses.add_to_registry(name="l2")
class L2Loss(nn.MSELoss):
    pass


@losses.add_to_registry(name="lpips")
class LPIPSLoss(LPIPS):
    pass


@losses.add_to_registry(name="lpips_scale")
class LPIPSScaleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = LPIPSLoss()

    def forward(self, x, y):
        out = 0
        for res in [256, 128, 64]:
            x_scale = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
            y_scale = F.interpolate(y, size=(res, res), mode="bilinear", align_corners=False)
            out += self.loss_fn.forward(x_scale, y_scale).mean()
        return out


@other_losses.add_to_registry(name="feat_rec")
class FeatReconLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, batch):
        return self.loss_fn(batch["feat_recon"], batch["feat_real"]).mean()


@other_losses.add_to_registry(name="feat_rec_l1")
class FeatReconL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.L1Loss()

    def forward(self, batch):
        return self.loss_fn(batch["feat_recon"], batch["feat_real"]).mean()



@other_losses.add_to_registry(name="l2_latent")
class LatentMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, batch):
        return self.loss_fn(batch["latent"], batch["latent_rec"]).mean()
        


@losses.add_to_registry(name="id")
class IDLoss(id_loss.IDLoss):
    pass


@losses.add_to_registry(name="id_vit")
class IDVitLoss(id_vit_loss.IDVitLoss):
    pass


@losses.add_to_registry(name="moco")
class MocoLoss(moco_loss.MocoLoss):
    pass


@adv_losses.add_to_registry(name="adv")
class EncoderAdvLoss:
    def __call__(self, fake_preds):
        loss_G_adv = F.softplus(-fake_preds).mean().detach()
        return loss_G_adv


@disc_losses.add_to_registry(name="main")
class AdvLoss:
    def __init__(self, coef=0.0):
        self.coef = coef

    def __call__(self, disc, loss_input):
        real_images = loss_input["x"].detach()
        generated_images = loss_input["y_hat"].detach()
        loss_dict = {}

        fake_preds = disc(generated_images, None)
        real_preds = disc(real_images, None)
        loss = self.d_logistic_loss(real_preds, fake_preds)
        loss_dict["disc/main_loss"] = float(loss)

        return loss, loss_dict

    def d_logistic_loss(self, real_preds, fake_preds):
        real_loss = F.softplus(-real_preds)
        fake_loss = F.softplus(fake_preds)

        return (real_loss.mean() + fake_loss.mean()) / 2


@disc_losses.add_to_registry(name="r1")
class R1Loss:
    def __init__(self, coef=0.0, hyper_d_reg_every=16):
        self.coef = coef
        self.hyper_d_reg_every = hyper_d_reg_every

    def __call__(self, disc, loss_input):
        real_images = loss_input["x"]
        step = loss_input["step"]
        if step % self.hyper_d_reg_every != 0:  # use r1 only once per 'hyper_d_reg_every' steps
            return torch.tensor([0.], requires_grad=True, device='cuda'), {}

        real_images.requires_grad = True
        loss_dict = {}

        real_preds = disc(real_images, None)
        real_preds = real_preds.view(real_images.size(0), -1)
        real_preds = real_preds.mean(dim=1).unsqueeze(1)
        r1_loss = self.d_r1_loss(real_preds, real_images)

        loss_D_R1 = self.coef / 2 * r1_loss * self.hyper_d_reg_every + 0 * real_preds[0]
        loss_dict["disc/r1_reg"] = float(loss_D_R1)
        return loss_D_R1, loss_dict

    def d_r1_loss(self, real_pred, real_img):
        (grad_real,) = torch.autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty


# ViVFace自定义损失函数

@other_losses.add_to_registry(name="vivface_self_rec")
class VIVFaceSelfRecLoss(nn.Module):
    """
    ViVFace自重建损失 - 计算源图像S与其重建S_hat之间的重建损失
    包含L2损失、LPIPS损失和梯度方差损失的组合
    """
    def __init__(self):
        super().__init__()
        self.l2_loss = L2Loss()
        self.lpips_loss = LPIPSLoss()
        self.grad_criterion = GradientVarianceLoss(patch_size=8)  # 使用与ViVFace相同的patch_size
        
    def forward(self, batch):
        # 获取源图像和自重建图像
        x_s = batch["source"]
        y_hat_s = batch["y_hat_s"]
        
        # 计算L2、LPIPS和梯度方差损失
        l2_loss = self.l2_loss(y_hat_s, x_s)
        lpips_loss = self.lpips_loss(y_hat_s, x_s).mean()
        gv_loss = self.grad_criterion(y_hat_s, x_s)
        
        # 按照ViVFace的损失组合方式：L2 + LPIPS + GV
        return l2_loss + lpips_loss + 0.1 * gv_loss


@other_losses.add_to_registry(name="vivface_reenact")
class VIVFaceReenactLoss(nn.Module):
    """
    ViVFace表情重演损失 - 计算驱动图像D1与表情迁移结果S_D1之间的重建损失
    包含L2损失、LPIPS损失和梯度方差损失的组合
    """
    def __init__(self):
        super().__init__()
        self.l2_loss = L2Loss()
        self.lpips_loss = LPIPSLoss()
        self.grad_criterion = GradientVarianceLoss(patch_size=8)  # 使用与ViVFace相同的patch_size
        
    def forward(self, batch):
        # 获取驱动图像和表情迁移结果
        x_d1 = batch["same_id"]
        y_hat_s_d1 = batch["y_hat_s_d1"]
        
        # 计算L2、LPIPS和梯度方差损失
        l2_loss = self.l2_loss(y_hat_s_d1, x_d1)
        lpips_loss = self.lpips_loss(y_hat_s_d1, x_d1).mean()
        gv_loss = self.grad_criterion(y_hat_s_d1, x_d1)
        
        # 按照ViVFace的损失组合方式：L2 + LPIPS + GV
        return l2_loss + lpips_loss + 0.1 * gv_loss


@other_losses.add_to_registry(name="vivface_w_consistency")
class VIVFaceWConsistencyLoss(nn.Module):
    """
    ViVFace w_latent一致性损失 - 确保身份编码的一致性
    对比S的w_latent和D2_S的w_latent
    """
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, batch):
        # 获取源图像S的w_latent和跨身份身份迁移D2_S的w_latent
        w_s = batch["w_s"]
        w_d2_s = batch["w_d2_s"]
        
        # 计算MSE损失
        return self.mse_loss(w_s, w_d2_s)


@other_losses.add_to_registry(name="vivface_ss_consistency")
class VIVFaceSSConsistencyLoss(nn.Module):
    """
    ViVFace ss_latent一致性损失 - 确保表情编码的一致性
    对比D2的ss_latent和D2_S的ss_latent
    """
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, batch):
        # 获取不同身份图像D2的ss_latent和跨身份身份迁移结果D2_S的ss_latent
        ss_d2 = batch["ss_d2"]
        ss_d2_s = batch["ss_d2_s"]
        
        # 计算MSE损失
        return self.mse_loss(ss_d2, ss_d2_s)


@other_losses.add_to_registry(name="vivface_ss_regularization")
class VIVFaceSSRegularizationLoss(nn.Module):
    """
    ViVFace ss_latent正则化损失 - 鼓励表情编码稀疏分布
    计算D1的ss_latent与零向量之间的MSE
    """
    def __init__(self, delta_norm_lambda=0.2, s_lambda=1.0):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.delta_norm_lambda = delta_norm_lambda  # 默认权重系数，与ViVFace对齐
        self.s_lambda = s_lambda  # 默认权重系数，与ViVFace对齐
        
    def forward(self, batch):
        # 获取同身份不同表情图像D1的ss_latent
        ss_s = batch["ss_s"]
        
        # 计算与零向量的MSE损失，鼓励稀疏表示
        zero_ss = torch.zeros_like(ss_s)
        
        # 应用权重系数，与原始ViVFace保持一致
        return self.delta_norm_lambda * self.s_lambda * self.mse_loss(ss_s, zero_ss)


@other_losses.add_to_registry(name="vivface_id")
class VIVFaceIDLoss(nn.Module):
    """
    ViVFace身份保持损失 - 使用预训练人脸识别网络确保身份一致性
    """
    def __init__(self):
        super().__init__()
        self.id_loss = id_loss.IDLoss()
        
    def forward(self, batch):
        # 获取源图像、中性表情生成结果和跨身份身份迁移结果
        x_s = batch["source"]
        y_hat_s_neutral = batch["y_hat_s_neutral"]
        y_hat_d2_s = batch["y_hat_d2_s"]
        
        # 计算身份损失
        id_loss_neutral = self.id_loss(y_hat_s_neutral, x_s)
        id_loss_cross = self.id_loss(y_hat_d2_s, x_s)
        
        # 组合损失
        return id_loss_neutral + id_loss_cross


@other_losses.add_to_registry(name="vivface_delta")
class VIVFaceDeltaLoss(nn.Module):
    """
    ViVFace渐进式delta损失 - 控制各层delta幅度
    根据当前progressive_stage，逐层计算w_latent各层delta与w0之间的L2范数
    """
    def __init__(self, delta_norm_lambda=0.0002, p_norm=2):
        super().__init__()
        self.delta_norm_lambda = delta_norm_lambda  # 默认权重系数，与ViVFace对齐
        self.p_norm = p_norm  # 使用L2范数
        
    def forward(self, batch):
        # 获取w_latent和progressive_stage
        w_s = batch["w_s"]
        progressive_stage = batch["progressive_stage"]
        
        # 检查progressive_stage的类型，确保是ProgressiveStage枚举
        if progressive_stage.value == 0 or progressive_stage.value == 18:  # W0阶段或Inference阶段不需要计算delta损失
            return torch.tensor(0.0, device=w_s.device)
        
        # 计算各层delta的p范数，与原始ViVFace保持一致
        first_w = w_s[:, 0, :]  # 使用第一层作为基准(W0)
        
        # 获取需要计算delta的维度，在原始ViVFace中是通过deltas_latent_dims获取的
        # 但在我们的实现中，可以直接使用连续的索引
        delta_loss = 0.0
        for i in range(1, progressive_stage.value + 1):
            delta = w_s[:, i, :] - first_w
            delta_norm = torch.norm(delta, p=self.p_norm, dim=1).mean()
            delta_loss += delta_norm
            
        # 应用权重系数
        return self.delta_norm_lambda * delta_loss


# 添加梯度方差损失
@losses.add_to_registry(name="gradient_variance")
class GradientVarianceLoss(nn.Module):
    """
    梯度方差损失(Gradient Variance Loss) - 来自ViVFace
    用于增强边缘一致性，通过计算两张图像在x和y方向上的梯度差异
    """
    def __init__(self, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        
    def forward(self, y_hat, y):
        # 计算x和y方向的梯度
        grad_y_x = torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])
        grad_y_y = torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])
        grad_y_hat_x = torch.abs(y_hat[:, :, :, :-1] - y_hat[:, :, :, 1:])
        grad_y_hat_y = torch.abs(y_hat[:, :, :-1, :] - y_hat[:, :, 1:, :])
        
        # 计算梯度差异的MSE损失
        grad_diff_x = torch.nn.functional.mse_loss(grad_y_hat_x, grad_y_x)
        grad_diff_y = torch.nn.functional.mse_loss(grad_y_hat_y, grad_y_y)
        
        # 返回两个方向梯度差异的平均值
        return (grad_diff_x + grad_diff_y) / 2.0
