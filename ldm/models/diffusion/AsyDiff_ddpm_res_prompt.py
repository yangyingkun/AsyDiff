import torch

import numpy as np

import pytorch_lightning as pl

from torch.optim.lr_scheduler import LambdaLR

from einops import rearrange

from contextlib import nullcontext

from functools import partial

import itertools

from tqdm import tqdm

from ldm.util import exists, default, instantiate_from_config

from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor

from ldm.models.diffusion.ddpm import LatentDiffusion

class AsyDiff_train_stage1_trans_model_Gencase_Loss(LatentDiffusion):
    def __init__(self, style_stage_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = DiffusionWrapper_AsyDiff_train_stage1_trans_model_Gencase(
            kwargs['unet_config'], kwargs['conditioning_key'])
        self.instantiate_style_stage(style_stage_config)
        # 配置梯度裁剪参数 (Lightning推荐方式)
        # self.gradient_clip_val = 0.03  # 梯度裁剪的范数值
        # self.gradient_clip_algorithm = "norm"  # 裁剪算法: "norm" 或 "value"

        # 添加日志输出，方便调试
        print(f"Training steps set to: {self.training_steps}")

    def instantiate_style_stage(self, config):
        model = instantiate_from_config(config)
        self.style_stage_model = model

    @torch.no_grad()
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())

        # # ignore unet parameters, training from scratch for diffuser
        ignore_keys = []
        for k in keys:
            if 'diffusion_model' in k:
                ignore_keys.append(k)
        print('ignore_keys', ignore_keys)
        # #################################################

        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        if self.make_it_fit:
            n_params = len([name for name, _ in
                            itertools.chain(self.named_parameters(),
                                            self.named_buffers())])
            for name, param in tqdm(
                    itertools.chain(self.named_parameters(),
                                    self.named_buffers()),
                    desc="Fitting old weights to new weights",
                    total=n_params
            ):
                if not name in sd:
                    continue
                old_shape = sd[name].shape
                new_shape = param.shape
                assert len(old_shape) == len(new_shape)
                if len(new_shape) > 2:
                    # we only modify first two axes
                    assert new_shape[2:] == old_shape[2:]
                # assumes first axis corresponds to output dim
                if not new_shape == old_shape:
                    new_param = param.clone()
                    old_param = sd[name]
                    if len(new_shape) == 1:
                        for i in range(new_param.shape[0]):
                            new_param[i] = old_param[i % old_shape[0]]
                    elif len(new_shape) >= 2:
                        for i in range(new_param.shape[0]):
                            for j in range(new_param.shape[1]):
                                new_param[i, j] = old_param[i %
                                                            old_shape[0], j % old_shape[1]]

                        n_used_old = torch.ones(old_shape[1])
                        for j in range(new_param.shape[1]):
                            n_used_old[j % old_shape[1]] += 1
                        n_used_new = torch.zeros(new_shape[1])
                        for j in range(new_param.shape[1]):
                            n_used_new[j] = n_used_old[j % old_shape[1]]

                        n_used_new = n_used_new[None, :]
                        while len(n_used_new.shape) < len(new_shape):
                            n_used_new = n_used_new.unsqueeze(-1)
                        new_param /= n_used_new

                    sd[name] = new_param

        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys:\n {missing}")
        if len(unexpected) > 0:
            print(f"\nUnexpected Keys:\n {unexpected}")

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        # params = params + list(self.style_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        print("~~~~~~~~~~~~~~~~")
        print(beta_schedule)
        print("~~~~~~~~~~~~~~~~")
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        # print('alphas_cumprod', alphas_cumprod)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas', to_torch(alphas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        # print('sqrt_one_minus_alphas_cumprod', np.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
            1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        ###################################################
        # coefficients for the forward change of DDPM
        ###################################################
        self.register_buffer('forward_chage_posterior_mean_coef1', to_torch(
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod_prev + betas)))
        self.register_buffer('forward_chage_posterior_mean_coef2', to_torch(
            betas / (1. - alphas_cumprod_prev + betas)))
        forward_chage_posterior_variance = (
            betas * (1. - alphas_cumprod_prev)) / (1. - alphas_cumprod_prev + betas)
        self.register_buffer('forward_chage_posterior_variance', to_torch(
            forward_chage_posterior_variance))
        self.register_buffer('forward_chage_posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(forward_chage_posterior_variance, 1e-20))))
        ###################################################
        # Diffusion Transformation Model for General Case
        print('***********************************************')
        print('Font Transfer Stage for General Case')
        print('self.edit_t1', self.edit_t1)
        print('self.edit_t2', self.edit_t2)
        temp_Ht = torch.zeros_like(to_torch(alphas)).numpy()
        sqrt_alphas = np.sqrt(alphas)
        for ii in range(self.edit_t1, self.edit_t2):  # Ht in paper
            temp_Ht[ii] = 1.0 + sqrt_alphas[ii] * temp_Ht[ii - 1]
        Ht = temp_Ht
        Ht_prev = np.append(0., Ht[:-1])
        self.register_buffer('forward_chage_Ht', to_torch(Ht))  # Ht in paper
        self.register_buffer('forward_chage_Ht_prev', to_torch(Ht_prev))
        temp_Ft = torch.zeros_like(to_torch(alphas)).numpy()
        for ii in range(self.edit_t1, self.edit_t2):
            temp_Ft[ii] = 1.0 + alphas[ii] * temp_Ft[ii - 1]
        Ft = temp_Ft
        Ft_prev = np.append(0., Ft[:-1])
        self.register_buffer('forward_chage_Ft', to_torch(Ft))
        self.register_buffer('forward_chage_Ft_prev', to_torch(Ft_prev))

        trans_coeff_1 = Ht[int(self.edit_t2 - 1)]
        self.register_buffer('forward_chage_trans_coeff_1',
                             to_torch(trans_coeff_1))
        trans_coeff_2 = Ft[int(self.edit_t2 - 1)]
        self.register_buffer('forward_chage_trans_coeff_2',
                             to_torch(trans_coeff_2))
        trans_coeff_3 = alphas_cumprod[int(self.edit_t2 - 1)]
        self.register_buffer('forward_chage_trans_coeff_3',
                             to_torch(trans_coeff_3))
        trans_coeff_4 = alphas_cumprod[int(self.edit_t2 - self.edit_t1 - 1)]
        self.register_buffer("forward_chage_trans_coeff_4",
                             to_torch(trans_coeff_4))

        ###################################################
        if self.parameterization == "x0":
            lvlb_weights = 0.5 * \
                np.sqrt(torch.Tensor(alphas_cumprod)) / \
                (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    def get_input_ori(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def get_input_ori_flat(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]

        x = rearrange(x, 'b s h w c -> b s c h w')

        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None, return_x=False):
        x = self.get_input_ori(batch, "gt_branch")

        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)

        xf = self.get_input_ori(batch, "pan_branch")
        xf = xf.to(self.device)

        if bs is not None:
            xf = xf[:bs]

        xc = None
        out = [x, xf]

        if return_first_stage_outputs:
            xrec = x
            out.extend([x, xrec])
        if return_x:
            out.extend([x])
        if return_original_cond:
            out.append(xc)
        return out

    def q_sample(self, x_start, c, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        # x_start 表示 残差图
        # cond[zf] 表示 pan 高通部分
        # cond[zcs] 表示 lrms 部分
        # cond[zcf] 表示 pan 部分
        zf = torch.cat(c['zf'], 0)

        # temp1 = t < self.edit_t1
        # temp1 = temp1.view(-1, 1, 1, 1)
        # temp2 = t >= self.edit_t2
        # temp2 = temp2.view(-1, 1, 1, 1)
        # params_1 = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * (x_start * temp1 + zf * temp2)
        # params_2 = torch.sqrt(1. - extract_into_tensor(self.alphas_cumprod, t, x_start.shape)) * noise

        # return params_1 + params_2

        stage1_temp1 = t >= self.edit_t2
        stage1_temp1 = stage1_temp1.long()
        stage1_inf1 = x_start * stage1_temp1.view(-1, 1, 1, 1)
        stage1_params_1 = extract_into_tensor(self.sqrt_alphas_cumprod, torch.clamp(t - self.edit_t1, 0, self.num_timesteps - 1), x_start.shape)
        stage1_params_2 = torch.sqrt(1. - extract_into_tensor(self.alphas_cumprod, torch.clamp(t - self.edit_t1, 0, self.num_timesteps - 1), x_start.shape))
        stage1 = stage1_params_1 * stage1_inf1 + stage1_params_2 * noise

        stage2_temp1 = t < self.edit_t2
        stage2_temp2 = t >= self.edit_t1
        stage2_temp = stage2_temp1 * stage2_temp2
        stage2_temp = stage2_temp.long()
        stage2_inf1 = x_start * stage2_temp.view(-1, 1, 1, 1)
        stage2_inf2 = zf * stage2_temp.view(-1, 1, 1, 1)
        stage2_params_1 = extract_into_tensor(self.sqrt_alphas_cumprod, torch.clamp(t, 0, self.num_timesteps - 1), x_start.shape) * stage2_inf1
        stage2_params_2 = extract_into_tensor(self.forward_chage_Ht, torch.clamp(t, 0, self.num_timesteps - 1), x_start.shape) * \
            (torch.sqrt(self.forward_chage_trans_coeff_3) * stage2_inf1 - torch.sqrt(self.forward_chage_trans_coeff_4) * stage2_inf2) / \
            self.forward_chage_trans_coeff_1
        stage2_params_3 = torch.sqrt(1. - extract_into_tensor(self.alphas_cumprod, torch.clamp(t, 0, self.num_timesteps - 1), x_start.shape) +
                              extract_into_tensor(self.forward_chage_Ft, torch.clamp(t, 0, self.num_timesteps - 1), x_start.shape) / self.forward_chage_trans_coeff_2 *
                              (self.forward_chage_trans_coeff_3 - self.forward_chage_trans_coeff_4)) * noise
        stage2 = stage2_params_1 - stage2_params_2 + stage2_params_3

        stage3_temp1 = t < self.edit_t1
        stage3_temp1 = stage3_temp1.long()
        stage3_inf1 = x_start * stage3_temp1.view(-1, 1, 1, 1)
        stage3_params_1 = extract_into_tensor(self.sqrt_alphas_cumprod, torch.clamp(t, 0, self.num_timesteps - 1), x_start.shape)
        stage3_params_2 = torch.sqrt(1. - extract_into_tensor(self.alphas_cumprod, torch.clamp(t, 0, self.num_timesteps - 1), x_start.shape))
        stage3 = stage3_params_1 * stage3_inf1 + stage3_params_2 * noise

        return stage1 + stage2 + stage3

    def p_losses(self, x_start, cond, t, noise=None):
        # x_start 表示 残差图
        # cond[zf] 表示 pan 高通部分
        # cond[zcs] 表示 lrms 部分
        # cond[zcf] 表示 pan 部分
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, c=cond, t=t, noise=noise)

        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        temp1 = t >= self.edit_t2
        temp1 = temp1.long()
        temp2 = t < self.edit_t2
        temp2 = temp2.long()

        zf = torch.cat(cond['zf'], 0)
        target = temp1.view(-1, 1, 1, 1) * zf + temp2.view(-1, 1, 1, 1) * x_start

        # target = x_start
        loss_simple = self.get_loss(
            model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})
        loss = loss_simple.mean()

        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def shared_step(self, batch, **kwargs):
        # x 表示 残差图
        # zf 表示 pan 高通部分
        x, zf = self.get_input(batch, self.first_stage_key)

        # xcs 表示 lrms 部分
        xcs = self.get_input_ori_flat(batch, "style_imgs")
        xcs = xcs.to(self.device)
        # zcs 表示 lrms 部分
        zcs = torch.mean(xcs, dim=1)

        # xcf 表示 pan 部分
        xcf = self.get_input_ori(batch, "grey_char_img")
        # zcf 表示 pan 部分
        zcf = xcf

        prompt = batch["prompt"]
        # we just combine them into dict
        c = dict(zf=[zf], zcs=[zcs], zcf=[zcf], prompt=[prompt])

        loss = self(x, c)
        return loss

    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.edit_t2,
                          (x.shape[0],), device=self.device).long()

        return self.p_losses(x, c, t, *args, **kwargs)

    def apply_model(self, x_noisy, t, cond, return_ids=False):

        x_recon = self.model(x_noisy, t, cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=50, ddim_eta=0., return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, unconditional_guidance_scale=1., unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        ema_scope = self.ema_scope if use_ema_scope else nullcontext
        use_ddim = ddim_steps is not None

        log = dict()
        # x 表示 残差图
        # zf 表示 pan 高通部分
        z, zf, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                            return_first_stage_outputs=True,
                                            force_c_encode=True,
                                            return_original_cond=True,
                                            bs=N)

        xs = self.get_input_ori_flat(batch, "style_imgs")
        xs = xs.to(self.device)
        # zcs 表示 lrms 部分
        zcs = torch.mean(xs, dim=1)

        if N is not None:
            zcs = zcs[:N]

        xcf = self.get_input_ori(batch, "grey_char_img")

        # zcf, _ = self.style_stage_model.encode(xcf)
        # zcf 表示 pan 部分
        zcf = xcf
        if N is not None:
            zcf = zcf[:N]

        prompt = batch["prompt"]

        c = dict(zf=[zf], zcs=[zcs], zcf=[zcf], prompt=[prompt])

        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

class DiffusionWrapper_AsyDiff_train_stage1_trans_model_Gencase(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.sequential_cross_attn = diff_model_config.pop(
            "sequential_crossattn", False)
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key

    def forward(self, x, t, cond):
        device = x.device
        # cond[zf] 表示 pan 高通部分
        # cond[zcs] 表示 lrms 部分
        # cond[zcf] 表示 pan 部分
        zcs = torch.cat(cond['zcs'], 1).to(device)
        zcf = torch.cat(cond['zcf'], 1).to(device)
        prompt = cond['prompt'][0]
        t = t.to(device)

        out = self.diffusion_model(lrms=zcs, pan=zcf, prompts=prompt, xt=x, timesteps=t)

        return out
