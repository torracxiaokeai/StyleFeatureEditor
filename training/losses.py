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
        # 修改损失函数识别逻辑，支持嵌套配置
        self.losses_names = []
        for k, v in enc_losses_dict.items():
            # 检查是否是字典类型配置（包括OmegaConf的DictConfig）
            if hasattr(v, 'get') or isinstance(v, dict):
                # 如果是字典类型，检查是否有lambda权重
                lambda_val = v.get('lambda', 0) if hasattr(v, 'get') else v.get('lambda', 0)
                if lambda_val > 0:
                    self.losses_names.append(k)
            else:
                # 如果是简单数值，直接比较
                try:
                    if float(v) > 0:
                        self.losses_names.append(k)
                except (ValueError, TypeError):
                    # 如果无法转换为数值，跳过
                    continue
        
        self.losses = {}
        self.adv_losses = {}
        self.other_losses = {}
        self.device = device

        for loss in self.losses_names:
            loss_config = enc_losses_dict[loss]
            
            # 提取损失函数参数
            if hasattr(loss_config, 'get') or isinstance(loss_config, dict):
                loss_kwargs = {k: v for k, v in loss_config.items() if k != 'lambda'}
            else:
                loss_kwargs = {}
            
            if loss in losses.classes.keys():
                try:
                    # 尝试传递参数
                    self.losses[loss] = losses[loss](**loss_kwargs).to(self.device).eval()
                except TypeError:
                    # 如果失败，回退到不传参数的方式（向后兼容）
                    if loss_kwargs:  # 只有当有参数时才提示
                        print(f"警告：损失函数 {loss} 不支持参数 {loss_kwargs}，使用默认参数")
                self.losses[loss] = losses[loss]().to(self.device).eval()
            elif loss in adv_losses.classes.keys():
                try:
                    self.adv_losses[loss] = adv_losses[loss](**loss_kwargs)
                except TypeError:
                    if loss_kwargs:
                        print(f"警告：对抗损失函数 {loss} 不支持参数 {loss_kwargs}，使用默认参数")
                self.adv_losses[loss] = adv_losses[loss]()
            elif loss in other_losses.classes.keys():
                try:
                    self.other_losses[loss] = other_losses[loss](**loss_kwargs)
                except TypeError:
                    if loss_kwargs:
                        print(f"警告：其他损失函数 {loss} 不支持参数 {loss_kwargs}，使用默认参数")
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
            # 获取权重系数
            loss_config = self.coefs_dict[loss_name]
            if hasattr(loss_config, 'get') or isinstance(loss_config, dict):
                weight = loss_config.get('lambda', 1.0) if hasattr(loss_config, 'get') else loss_config.get('lambda', 1.0)
            else:
                weight = float(loss_config)
            
            global_loss += weight * loss_val
            loss_dict[loss_name] = float(loss_val)

        for loss_name, loss in self.other_losses.items():
            loss_val = loss(batch_data)
            assert torch.isfinite(loss_val)
            # 获取权重系数
            loss_config = self.coefs_dict[loss_name]
            if hasattr(loss_config, 'get') or isinstance(loss_config, dict):
                weight = loss_config.get('lambda', 1.0) if hasattr(loss_config, 'get') else loss_config.get('lambda', 1.0)
            else:
                weight = float(loss_config)
            
            global_loss += weight * loss_val
            loss_dict[loss_name] = float(loss_val)

        if batch_data["use_adv_loss"]:
            for loss_name, loss in self.adv_losses.items():
                loss_val = loss(batch_data["fake_preds"])
                # 获取权重系数
                loss_config = self.coefs_dict[loss_name]
                if hasattr(loss_config, 'get') or isinstance(loss_config, dict):
                    weight = loss_config.get('lambda', 1.0) if hasattr(loss_config, 'get') else loss_config.get('lambda', 1.0)
                else:
                    weight = float(loss_config)
                
                global_loss += weight * loss_val
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


# ==========================================
# VivFace特定损失函数
# ==========================================

@other_losses.add_to_registry(name="L_self")
class VivFaceSelfReconLoss(nn.Module):
    """
    VivFace自重建损失 (S→S_hat)
    
    结合MSE、LPIPS和梯度方差损失的重建损失
    支持可配置的权重参数
    """
    def __init__(self, l2_lambda=1.0, lpips_lambda=0.8, gv_lambda=0.1):
        super().__init__()
        self.l2_lambda = l2_lambda
        self.lpips_lambda = lpips_lambda 
        self.gv_lambda = gv_lambda
        
        self.mse_loss = nn.MSELoss()
        # 延迟初始化LPIPS和梯度方差损失
        self.lpips_loss = None
        self.grad_criterion = None
        self.initialized = False
    
    def _initialize_losses(self, device):
        """延迟初始化损失函数"""
        if not self.initialized:
            self.lpips_loss = LPIPS(net_type='alex').to(device).eval()
            # 导入梯度方差损失
            try:
                from criteria.gradient_variance_loss import GradientVariance
                self.grad_criterion = GradientVariance(patch_size=None)
            except ImportError:
                print("警告：无法导入梯度方差损失，使用MSE替代")
                self.grad_criterion = None
            self.initialized = True
    
    def forward(self, batch):
        S = batch["S"]
        S_hat = batch["S_hat"]
        
        # 延迟初始化
        if not self.initialized:
            self._initialize_losses(S.device)
        
        # MSE损失
        l2_loss = self.mse_loss(S_hat, S)
        
        # LPIPS感知损失
        lpips_loss = self.lpips_loss(S_hat, S).mean()
        
        # 梯度方差损失
        if self.grad_criterion is not None:
            gv_loss = self.grad_criterion(S_hat, S)
        else:
            gv_loss = 0.0
        
        # 应用权重组合
        total_loss = self.l2_lambda * l2_loss + self.lpips_lambda * lpips_loss + self.gv_lambda * gv_loss
        
        return total_loss

@other_losses.add_to_registry(name="L_reenact")
class VivFaceReenactLoss(nn.Module):
    """
    VivFace重演损失 (D1→S_D1)
    
    同样结合MSE、LPIPS和梯度方差损失
    支持可配置的权重参数
    """
    def __init__(self, l2_lambda=1.0, lpips_lambda=0.8, gv_lambda=0.1):
        super().__init__()
        self.l2_lambda = l2_lambda
        self.lpips_lambda = lpips_lambda 
        self.gv_lambda = gv_lambda
        
        self.mse_loss = nn.MSELoss()
        # 延迟初始化LPIPS和梯度方差损失
        self.lpips_loss = None
        self.grad_criterion = None
        self.initialized = False
    
    def _initialize_losses(self, device):
        """延迟初始化损失函数"""
        if not self.initialized:
            self.lpips_loss = LPIPS(net_type='alex').to(device).eval()
            # 导入梯度方差损失
            try:
                from criteria.gradient_variance_loss import GradientVariance
                self.grad_criterion = GradientVariance(patch_size=None)
            except ImportError:
                print("警告：无法导入梯度方差损失，使用MSE替代")
                self.grad_criterion = None
            self.initialized = True
    
    def forward(self, batch):
        D1 = batch["D1"]
        S_D1 = batch["S_D1"]
        
        # 延迟初始化
        if not self.initialized:
            self._initialize_losses(D1.device)
        
        # MSE损失
        l2_loss = self.mse_loss(S_D1, D1)
        
        # LPIPS感知损失
        lpips_loss = self.lpips_loss(S_D1, D1).mean()
        
        # 梯度方差损失
        if self.grad_criterion is not None:
            gv_loss = self.grad_criterion(S_D1, D1)
        else:
            gv_loss = 0.0
        
        # 应用权重组合
        total_loss = self.l2_lambda * l2_loss + self.lpips_lambda * lpips_loss + self.gv_lambda * gv_loss
        
        return total_loss

@other_losses.add_to_registry(name="L_latent_consistency")
class VivFaceLatentConsistencyLoss(nn.Module):
    """
    VivFace潜在编码一致性损失
    
    确保S的w_latent与D2_S的w_latent一致
    """
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, batch):
        S_latent = batch["S_latent"]
        D2_S_latent = batch["D2_S_latent"]
        
        if S_latent is None or D2_S_latent is None:
            return torch.tensor(0.0, device=batch["S"].device, requires_grad=True)
        
        return self.mse_loss(S_latent['w_latent'], D2_S_latent['w_latent'])

@other_losses.add_to_registry(name="L_ss_latent_consistency")
class VivFaceSSLatentConsistencyLoss(nn.Module):
    """
    VivFace StyleSpace潜在编码一致性损失
    
    确保D2的ss_generic_latent与D2_S的ss_generic_latent一致
    """
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, batch):
        D2_latent = batch["D2_latent"]
        D2_S_latent = batch["D2_S_latent"]
        
        if D2_latent is None or D2_S_latent is None:
            return torch.tensor(0.0, device=batch["S"].device, requires_grad=True)
        
        return self.mse_loss(D2_latent['ss_generic_latent'], D2_S_latent['ss_generic_latent'])

@other_losses.add_to_registry(name="L_ss_latent_regularization")
class VivFaceSSLatentRegularizationLoss(nn.Module):
    """
    VivFace StyleSpace潜在编码正则化损失
    
    基于ViVFace原始实现：
    loss_dict['L_ss_latent_regularization'] = self.opts.delta_norm_lambda*self.opts.s_lambda*self.mse_loss(D1_latent['ss_generic_latent'], torch.zeros_like(D1_latent['ss_generic_latent']).cuda())
    
    鼓励D1的ss_generic_latent接近零（稀疏性）
    """
    def __init__(self, delta_norm_lambda=2e-4, s_lambda=0.2):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.delta_norm_lambda = delta_norm_lambda
        self.s_lambda = s_lambda
    
    def forward(self, batch):
        D1_latent = batch["D1_latent"]
        
        if D1_latent is None:
            return torch.tensor(0.0, device=batch["S"].device, requires_grad=True)
        
        # 使用ss_generic_latent而不是ss_latent
        ss_generic_latent = D1_latent['ss_generic_latent']
        zero_ss_latent = torch.zeros_like(ss_generic_latent).to(ss_generic_latent.device)
        
        # 应用ViVFace的权重组合：delta_norm_lambda * s_lambda
        loss = self.delta_norm_lambda * self.s_lambda * self.mse_loss(ss_generic_latent, zero_ss_latent)
        
        return loss

@other_losses.add_to_registry(name="loss_id")
class VivFaceIdentityLoss(nn.Module):
    """
    VivFace身份保持损失
    
    使用预训练的人脸识别网络确保身份一致性
    """
    def __init__(self):
        super().__init__()
        self.id_loss = None
        self.initialized = False
    
    def _initialize_loss(self, device):
        """延迟初始化身份损失"""
        if not self.initialized:
            try:
                from criteria import id_loss
                self.id_loss = id_loss.IDLoss().to(device).eval()
                print("成功加载身份损失函数")
            except ImportError:
                print("警告：无法导入身份损失，跳过身份损失计算")
                self.id_loss = None
            self.initialized = True
    
    def forward(self, batch):
        S = batch["S"]
        S_neutral = batch["S_neutral"]
        D2_S = batch["D2_S"]
        
        # 延迟初始化
        if not self.initialized:
            self._initialize_loss(S.device)
        
        if self.id_loss is None:
            return torch.tensor(0.0, device=S.device, requires_grad=True)
        
        if S_neutral is None or D2_S is None:
            return torch.tensor(0.0, device=S.device, requires_grad=True)
        
        # 计算两个身份损失
        loss_id_a, _, _ = self.id_loss(S_neutral, S, S_neutral)
        loss_id_b, _, _ = self.id_loss(D2_S, S, D2_S)
        
        return loss_id_a + loss_id_b

@other_losses.add_to_registry(name="delta_losses")
class VivFaceDeltaLoss(nn.Module):
    """
    VivFace渐进式delta损失
    
    控制各层delta的幅度，支持渐进式训练
    """
    def __init__(self, delta_norm=2, delta_norm_lambda=1.0):
        super().__init__()
        self.delta_norm = delta_norm
        self.delta_norm_lambda = delta_norm_lambda
    
    def forward(self, batch):
        S_latent = batch["S_latent"]
        progressive_stage = batch["progressive_stage"]
        
        if S_latent is None or progressive_stage is None:
            return torch.tensor(0.0, device=batch["S"].device, requires_grad=True)
        
        # 如果是推理阶段（18），不计算delta损失
        if progressive_stage.value == 18:
            return torch.tensor(0.0, device=batch["S"].device, requires_grad=True)
        
        total_delta_loss = torch.tensor(0.0, device=batch["S"].device, requires_grad=True)
        
        # 获取w_latent
        w_latent = S_latent['w_latent']  # [batch, 18, 512]
        first_w = w_latent[:, 0, :]  # [batch, 512]
        
        # 计算从第1层到当前训练阶段的delta损失
        for i in range(1, progressive_stage.value + 1):
            delta = w_latent[:, i, :] - first_w
            delta_loss = torch.norm(delta, self.delta_norm, dim=1).mean()
            total_delta_loss = total_delta_loss + self.delta_norm_lambda * delta_loss
        
        return total_delta_loss

@adv_losses.add_to_registry(name="encoder_discriminator_loss")
class VivFaceEncoderDiscriminatorLoss:
    """
    VivFace编码器判别器对抗损失
    
    用于训练编码器欺骗W判别器
    """
    def __call__(self, fake_preds):
        # fake_preds实际上是w_latent，需要进一步处理
        # 这个会在LossBuilder中处理判别器的调用
        if isinstance(fake_preds, torch.Tensor):
            return F.softplus(-fake_preds).mean()
        else:
            # 如果传入的是w_latent，返回0（会在train_step中单独处理）
            return torch.tensor(0.0, device=fake_preds['w_latent'].device if isinstance(fake_preds, dict) else fake_preds.device)

# ==========================================
# VivFace W判别器损失
# ==========================================

@disc_losses.add_to_registry(name="w_discriminator")
class VivFaceWDiscriminatorLoss:
    """
    VivFace W判别器损失
    
    专门用于训练W空间的判别器，区分真实和生成的w编码
    """
    def __init__(self, coef=1.0, r1_gamma=10.0):
        self.coef = coef
        self.r1_gamma = r1_gamma

    def __call__(self, w_disc, loss_input):
        """
        计算W判别器损失
        
        Args:
            w_disc: W判别器
            loss_input: 包含真实和虚假w编码的输入
        """
        loss_dict = {}
        
        # 获取真实和虚假的w编码
        fake_w = loss_input.get("fake_w", None)
        real_w = loss_input.get("real_w", None)
        
        if fake_w is None or real_w is None:
            return torch.tensor(0.0), loss_dict
        
        # 判别器预测
        fake_preds = w_disc(fake_w)
        real_preds = w_disc(real_w)
        
        # 计算判别器损失
        real_loss = F.softplus(-real_preds).mean()
        fake_loss = F.softplus(fake_preds).mean()
        
        disc_loss = real_loss + fake_loss
        
        loss_dict["w_disc/real_loss"] = float(real_loss)
        loss_dict["w_disc/fake_loss"] = float(fake_loss)
        loss_dict["w_disc/total_loss"] = float(disc_loss)
        
        # R1正则化（如果启用）
        if self.r1_gamma > 0:
            real_w.requires_grad_(True)
            real_preds_for_grad = w_disc(real_w)
            r1_grads = torch.autograd.grad(
                outputs=real_preds_for_grad.sum(),
                inputs=real_w,
                create_graph=True
            )[0]
            r1_penalty = r1_grads.pow(2).sum([1]).mean()
            
            disc_loss = disc_loss + self.r1_gamma * r1_penalty
            loss_dict["w_disc/r1_penalty"] = float(r1_penalty)
        
        return self.coef * disc_loss, loss_dict
