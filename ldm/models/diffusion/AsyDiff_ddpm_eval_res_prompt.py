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

from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like

from ldm.models.diffusion.ddpm import LatentDiffusion

class AsyDiff_model_eval_Gencase_Loss(LatentDiffusion):
    def __init__(self, style_stage_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # print('1')
        self.model = DiffusionWrapper_AsyDiff_model_eval_Gencase(
            kwargs['unet_config'], kwargs['conditioning_key'])
        self.instantiate_style_stage(style_stage_config)

    def instantiate_style_stage(self, config):
        model = instantiate_from_config(config)
        self.style_stage_model = model

    @torch.no_grad()
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        # print('2')
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())

        # #################################################
        # # ignore attn2 parameters, load pretrained diffuser model (e.g., SD released model)
        # ignore_keys = []
        # for k in keys:
        #     if 'attn2' in k:
        #         ignore_keys.append(k)

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
        params = params + list(self.style_stage_model.parameters())
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
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)
        # to_torch = partial(torch.tensor, dtype=torch.float16)

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

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # stage 3
        stage3_posterior_variance = (
            1. - alphas) * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer("stage3_posterior_variance",
                             to_torch(stage3_posterior_variance))
        self.register_buffer("stage3_posterior_log_variance_clipped", to_torch(
            np.log(np.maximum(stage3_posterior_variance, 1e-20))))
        self.register_buffer("stage3_posterior_mean_coef1", to_torch(
            np.sqrt(alphas) * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer("stage3_posterior_mean_coef2", to_torch(
            np.sqrt(alphas_cumprod_prev) * (1. - alphas) / (1. - alphas_cumprod)))
        # stage 2
        stage2_posterior_variance = (1. - alphas + (trans_coeff_3 - trans_coeff_4) / trans_coeff_2) \
            * (1. - alphas_cumprod_prev + Ft_prev * (trans_coeff_3 - trans_coeff_4) / trans_coeff_2) \
            / (1. - alphas_cumprod + Ft * (trans_coeff_3 - trans_coeff_4) / trans_coeff_2)
        self.register_buffer("stage2_posterior_variance",
                             to_torch(stage2_posterior_variance))
        self.register_buffer("stage2_posterior_log_variance_clipped", to_torch(
            np.log(np.maximum(stage2_posterior_variance, 1e-20))))
        self.register_buffer("stage2_posterior_mean_coef1", to_torch(
            (1. - alphas_cumprod_prev + Ft_prev *
             (trans_coeff_3 - trans_coeff_4) / trans_coeff_2)
            / (1. - alphas_cumprod + Ft * (trans_coeff_3 - trans_coeff_4) / trans_coeff_2)
            * np.sqrt(alphas)))
        self.register_buffer("stage2_posterior_mean_coef2", to_torch(
            (1. - alphas + (trans_coeff_3 - trans_coeff_4) / trans_coeff_2)
            / (1. - alphas_cumprod + Ft * (trans_coeff_3 - trans_coeff_4) / trans_coeff_2)
            * np.sqrt(alphas_cumprod_prev)))
        self.register_buffer("stage2_posterior_mean_coef3", to_torch(
            (1. - alphas_cumprod_prev + Ft_prev *
             (trans_coeff_3 - trans_coeff_4) / trans_coeff_2)
            / (1. - alphas_cumprod + Ft * (trans_coeff_3 - trans_coeff_4) / trans_coeff_2)
            * np.sqrt(alphas) / trans_coeff_1))
        self.register_buffer("stage2_posterior_mean_coef4", to_torch(
            (1. - alphas + (trans_coeff_3 - trans_coeff_4) / trans_coeff_2)
            / (1. - alphas_cumprod + Ft * (trans_coeff_3 - trans_coeff_4) / trans_coeff_2)
            * Ht_prev / trans_coeff_1))
        # stage 1
        stage1_posterior_variance = (
            1. - alphas) * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer("stage1_posterior_variance",
                             to_torch(stage1_posterior_variance))
        self.register_buffer("stage1_posterior_log_variance_clipped", to_torch(
            np.log(np.maximum(stage1_posterior_variance, 1e-20))))
        self.register_buffer("stage1_posterior_mean_coef1", to_torch(
            np.sqrt(alphas) * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer("stage1_posterior_mean_coef2", to_torch(
            np.sqrt(alphas_cumprod_prev) * (1. - alphas) / (1. - alphas_cumprod)))
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

        x = rearrange(x, 'b x h w c -> b x c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None, return_x=False):
        x = self.get_input_ori(batch, "gt_branch")
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)

        z = x

        xf = self.get_input_ori(batch, "pan_branch")
        xf = xf.to(self.device)

        zf = xf

        if bs is not None:
            zf = zf[:bs]

        c = zf
        xc = None
        out = [z, c]
        if return_first_stage_outputs:
            xrec = z
            out.extend([x, xrec])
        if return_x:
            out.extend([x])
        if return_original_cond:
            out.append(xc)
        return out

    def q_sample(self, x_start, c, t, noise=None):
        # x_start 表示 残差图
        # cond[zf] 表示 pan 高通部分
        # cond[zcs] 表示 lrms 部分
        # cond[zcf] 表示 pan 部分
        noise = default(noise, lambda: torch.randn_like(x_start))

        zf = torch.cat(c['zf'], 1)
        temp1 = t >= self.edit_t1
        temp2 = t <= self.edit_t2
        temp3 = temp1 * temp2
        temp3 = temp3.long()

        params_1 = extract_into_tensor(
            self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        params_2 = temp3.view(-1, 1, 1, 1) * extract_into_tensor(self.forward_chage_Ht, t, x_start.shape) * \
            (torch.sqrt(self.forward_chage_trans_coeff_3) * x_start - torch.sqrt(self.forward_chage_trans_coeff_4) * zf) / \
            self.forward_chage_trans_coeff_1
        params_3 = torch.sqrt(1. - extract_into_tensor(self.alphas_cumprod, t, x_start.shape) + temp3.view(-1, 1, 1, 1) *
                              extract_into_tensor(self.forward_chage_Ft, t, x_start.shape) / self.forward_chage_trans_coeff_2 *
                              (self.forward_chage_trans_coeff_3 - self.forward_chage_trans_coeff_4)) * noise

        return params_1 - params_2 + params_3

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, c=cond, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        target = x_start

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
        zcs = torch.mean(xcs, dim=1)
        # xcf 表示 pan 部分
        xcf = self.get_input_ori(batch, "grey_char_img")
        zcf = xcf

        c = dict(zf=[zf], zcs=[zcs], zcf=[zcf])

        loss = self(x, c)
        return loss

    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(self.edit_t1, self.edit_t2,
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
        z, zf, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                            return_first_stage_outputs=True,
                                            force_c_encode=True,
                                            return_original_cond=True,
                                            bs=N)

        xs = self.get_input_ori_flat(batch, "style_imgs")
        xs = xs.to(self.device)
        zcs = torch.mean(xs, dim=1)
        if N is not None:
            zcs = zcs[:N]

        xcf = self.get_input_ori(batch, "grey_char_img")
        if N is not None:
            zcf = zcf[:N]

        c = dict(zf=[zf], zcs=[zcs], zcf=[zcf])

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

    def q_posterior(self, x_start, x_t, z2, t):
        temp_time = t[0]
        if temp_time >= self.edit_t2:
            posterior_mean = (
                extract_into_tensor(self.stage3_posterior_mean_coef1, t - self.edit_t1, x_t.shape) * x_t +
                extract_into_tensor(
                    self.stage3_posterior_mean_coef2, t - self.edit_t1, x_t.shape) * z2
            )
            posterior_variance = extract_into_tensor(
                self.stage3_posterior_variance, t - self.edit_t1, x_t.shape)
            posterior_log_variance_clipped = extract_into_tensor(
                self.stage3_posterior_log_variance_clipped, t - self.edit_t1, x_t.shape)

        elif temp_time < self.edit_t2 and temp_time >= self.edit_t1:
            posterior_mean = (
                extract_into_tensor(self.stage2_posterior_mean_coef1, t, x_t.shape) * x_t +
                extract_into_tensor(self.stage2_posterior_mean_coef2, t, x_t.shape) * x_start +
                extract_into_tensor(self.stage2_posterior_mean_coef3, t, x_t.shape) *
                (torch.sqrt(self.forward_chage_trans_coeff_3) * x_start - torch.sqrt(self.forward_chage_trans_coeff_4) * z2) -
                extract_into_tensor(self.stage2_posterior_mean_coef4, t, x_t.shape) *
                (torch.sqrt(self.forward_chage_trans_coeff_3) * x_start -
                 torch.sqrt(self.forward_chage_trans_coeff_4) * z2)
            )
            posterior_variance = extract_into_tensor(
                self.stage2_posterior_variance, t, x_t.shape)
            posterior_log_variance_clipped = extract_into_tensor(
                self.stage2_posterior_log_variance_clipped, t, x_t.shape)

        else:
            posterior_mean = (
                extract_into_tensor(self.stage1_posterior_mean_coef1, t, x_t.shape) * x_t +
                extract_into_tensor(
                    self.stage1_posterior_mean_coef2, t, x_t.shape) * x_start
            )
            posterior_variance = extract_into_tensor(
                self.stage1_posterior_variance, t, x_t.shape)
            posterior_log_variance_clipped = extract_into_tensor(
                self.stage1_posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, rec_model, x, cond1, cond2, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        t_in = t
        temp_time = t[0]
        # z2 表示 pan 高通部分
        # x 表示 预测的残差图
        z2 = torch.cat(cond1['zf'], 1)

        if temp_time < self.edit_t2 and temp_time >= self.edit_t1:
            model_out = self.apply_model(
                x, t_in, cond2, return_ids=return_codebook_ids)
        elif temp_time >= self.edit_t2:
            model_out = z2
        else:
            model_out = rec_model.apply_model(
                x, t_in, cond1, return_ids=return_codebook_ids)

        model_out = model_out.clamp_(-1., 1.)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, z2=z2, t=t)

        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, rec_model, x, cond1, cond2, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        # print('16')
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(rec_model=rec_model, x=x, cond1=cond1, cond2=cond2, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        # noise = default(noise, lambda: torch.randn_like(x))
        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(self, rec_model, cond1, cond2, shape, verbose=True, callback=None, quantize_denoised=False,
                              img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps + self.edit_t1
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []

        # 初始化情况下image就是一个服从正态分布的噪声图
        self.noise = img
        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
            range(0, timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)

            img, x0_partial = self.p_sample(rec_model, img, cond1, cond2, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            # print(i, torch.max(img), torch.min(img))
            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def ddim_p_sample(self, x, cond1, cond2, t, t_prev, clip_denoised=False, repeat_noise=False,
                      return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        # print('16')
        b, *_, device = *x.shape, x.device
        t_in = t
        temp_time = t[0]

        z2 = torch.cat(cond1['zf'], 1)
        x_t = x

        if temp_time >= self.edit_t2:
            model_out = z2
        elif temp_time < self.edit_t2 and temp_time >= self.edit_t1:
            model_out = self.apply_model(
                x, t_in, cond2, return_ids=return_codebook_ids)
        else:
            model_out = self.apply_model(
                x, t_in, cond1, return_ids=return_codebook_ids)

        model_out = model_out.clamp_(-1., 1.)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if temp_time >= self.edit_t2:
            middle_params1 = extract_into_tensor(
                self.alphas_cumprod, t - self.edit_t1, x_t.shape)
            middle_params2 = extract_into_tensor(
                self.alphas_cumprod_prev, t_prev - self.edit_t1 + 1, x_t.shape)

            k = torch.sqrt(1. - middle_params2) / \
                torch.sqrt(1. - middle_params1)
            m = torch.sqrt(middle_params2) - k * torch.sqrt(middle_params1)
            xt_prev = k * x_t + m * x_recon

        elif temp_time < self.edit_t2 and temp_time >= self.edit_t1:
            middle_params1 = extract_into_tensor(
                self.alphas_cumprod, t, x_t.shape)
            middle_params2 = extract_into_tensor(
                self.alphas_cumprod_prev, t_prev + 1, x_t.shape)
            middle_params3 = extract_into_tensor(
                self.forward_chage_Ht, t, x_t.shape)
            middle_params4 = extract_into_tensor(
                self.forward_chage_Ht_prev, t_prev + 1, x_t.shape)
            middle_params5 = extract_into_tensor(
                self.forward_chage_Ft, t, x_t.shape)
            middle_params6 = extract_into_tensor(
                self.forward_chage_Ft_prev, t_prev + 1, x_t.shape)

            k = torch.sqrt(1. - middle_params2 + middle_params6 * (self.forward_chage_trans_coeff_3 - self.forward_chage_trans_coeff_4) / self.forward_chage_trans_coeff_2) / \
                torch.sqrt(1. - middle_params1 + middle_params5 * (self.forward_chage_trans_coeff_3 - self.forward_chage_trans_coeff_4) / self.forward_chage_trans_coeff_2)
            m = (torch.sqrt(middle_params2) - middle_params4 * torch.sqrt(self.forward_chage_trans_coeff_3) / self.forward_chage_trans_coeff_1) - \
                k * (torch.sqrt(middle_params1) - middle_params3 * torch.sqrt(self.forward_chage_trans_coeff_3) / self.forward_chage_trans_coeff_1)
            n = (middle_params4 * torch.sqrt(self.forward_chage_trans_coeff_4) / self.forward_chage_trans_coeff_1) - \
                k * (middle_params3 * torch.sqrt(self.forward_chage_trans_coeff_4) / self.forward_chage_trans_coeff_1)

            xt_prev = k * x_t + m * x_recon + n * z2
            # xt_prev = x_t

        else:
            middle_params1 = extract_into_tensor(
                self.alphas_cumprod, t, x_t.shape)
            middle_params2 = extract_into_tensor(
                self.alphas_cumprod_prev, t_prev + 1, x_t.shape)

            k = torch.sqrt(1. - middle_params2) / \
                torch.sqrt(1. - middle_params1)
            m = torch.sqrt(middle_params2) - k * torch.sqrt(middle_params1)

            xt_prev = k * x_t + m * x_recon


        if return_codebook_ids:
            return xt_prev, logits.argmax(dim=1)
        if return_x0:
            return xt_prev, x_recon
        else:
            return xt_prev

    @torch.no_grad()
    def ddim_progressive_denoising(self, cond1, cond2, shape, skip=1, verbose=True, callback=None, quantize_denoised=False,
                                   img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                                   score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                                   log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps + self.edit_t1
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []

        # 初始化情况下image就是一个服从正态分布的噪声图
        self.noise = img
        if start_T is not None:
            timesteps = min(timesteps, start_T)
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        main_timesteps = list(range(skip-1, timesteps, skip))
        main_timesteps_prev = [-1] + main_timesteps[:-1]

        # 创建迭代器（保持原有风格）
        iterator = tqdm(
            reversed(list(zip(main_timesteps, main_timesteps_prev))),
            desc='Progressive Generation',
            total=len(main_timesteps)
        ) if verbose else reversed(list(zip(main_timesteps, main_timesteps_prev)))

        for i, i_prev in iterator:
            # print(i)
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            ts_prev = torch.full(
                (b,), i_prev, device=self.device, dtype=torch.long)

            img, x0_partial = self.ddim_p_sample(img, cond1, cond2, ts, ts_prev,
                                                 clip_denoised=self.clip_denoised,
                                                 quantize_denoised=quantize_denoised, return_x0=True,
                                                 temperature=temperature[i], noise_dropout=noise_dropout,
                                                 score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def single_p_sample(self, x, cond1, cond2, t, t_prev, clip_denoised=False, repeat_noise=False,
                      return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        # print('16')
        b, *_, device = *x.shape, x.device
        t_in = t
        temp_time = t[0]

        z2 = torch.cat(cond1['zf'], 1)
        x_t = x

        model_out = self.apply_model(
            x, t_in, cond2, return_ids=return_codebook_ids)

        model_out = model_out.clamp_(-1., 1.)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        middle_params1 = extract_into_tensor(
            self.alphas_cumprod, t, x_t.shape)
        middle_params2 = extract_into_tensor(
            self.alphas_cumprod_prev, t_prev + 1, x_t.shape)

        k = torch.sqrt(1. - middle_params2) / \
            torch.sqrt(1. - middle_params1)
        m = torch.sqrt(middle_params2) - k * torch.sqrt(middle_params1)

        xt_prev = k * x_t + m * x_recon

        if return_codebook_ids:
            return xt_prev, logits.argmax(dim=1)
        if return_x0:
            return xt_prev, x_recon
        else:
            return xt_prev


    @torch.no_grad()
    def single_progressive_denoising(self, cond1, cond2, shape, skip=1, verbose=True, callback=None, quantize_denoised=False,
                                   img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                                   score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                                   log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        # timesteps = self.num_timesteps + self.edit_t1
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []

        # 初始化情况下image就是一个服从正态分布的噪声图
        self.noise = img
        if start_T is not None:
            timesteps = min(timesteps, start_T)
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        main_timesteps = list(range(skip-1, timesteps, skip))
        main_timesteps_prev = [-1] + main_timesteps[:-1]
        # main_timesteps = [self.edit_t1 - 1, self.edit_t2 - 1, timesteps - 1]
        # main_timesteps_prev = [-1] + main_timesteps[:-1]

        # 创建迭代器（保持原有风格）
        iterator = tqdm(
            reversed(list(zip(main_timesteps, main_timesteps_prev))),
            desc='Progressive Generation',
            total=len(main_timesteps)
        ) if verbose else reversed(list(zip(main_timesteps, main_timesteps_prev)))

        # iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
        #                 total=timesteps) if verbose else reversed(
        #     range(0, timesteps))

        for i, i_prev in iterator:
            # print(i)
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            ts_prev = torch.full(
                (b,), i_prev, device=self.device, dtype=torch.long)

            img, x0_partial = self.single_p_sample(img, cond1, cond2, ts, ts_prev,
                                                 clip_denoised=self.clip_denoised,
                                                 quantize_denoised=quantize_denoised, return_x0=True,
                                                 temperature=temperature[i], noise_dropout=noise_dropout,
                                                 score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            # print(i, torch.max(img), torch.min(img))
            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def ddim_p_sample_without_trans(self, x, cond1, cond2, t, t_prev, clip_denoised=False, repeat_noise=False,
                      return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        # print('16')
        b, *_, device = *x.shape, x.device
        t_in = t
        temp_time = t[0]

        z2 = torch.cat(cond1['zf'], 1)
        x_t = x

        if temp_time >= self.edit_t2:
            model_out = z2
        elif temp_time < self.edit_t2 and temp_time >= self.edit_t1:
            # model_out = self.apply_model(
            #     x, t_in, cond2, return_ids=return_codebook_ids)
            model_out = self.apply_model(
                x, t_in, cond1, return_ids=return_codebook_ids)
        else:
            model_out = self.apply_model(
                x, t_in, cond1, return_ids=return_codebook_ids)

        model_out = model_out.clamp_(-1., 1.)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if temp_time >= self.edit_t2:
            middle_params1 = extract_into_tensor(
                self.alphas_cumprod, t - self.edit_t1, x_t.shape)
            middle_params2 = extract_into_tensor(
                self.alphas_cumprod_prev, t_prev - self.edit_t1 + 1, x_t.shape)

            k = torch.sqrt(1. - middle_params2) / \
                torch.sqrt(1. - middle_params1)
            m = torch.sqrt(middle_params2) - k * torch.sqrt(middle_params1)
            xt_prev = k * x_t + m * x_recon

        elif temp_time < self.edit_t2 and temp_time >= self.edit_t1:
            # middle_params1 = extract_into_tensor(
            #     self.alphas_cumprod, t, x_t.shape)
            # middle_params2 = extract_into_tensor(
            #     self.alphas_cumprod_prev, t_prev + 1, x_t.shape)
            # middle_params3 = extract_into_tensor(
            #     self.forward_chage_Ht, t, x_t.shape)
            # middle_params4 = extract_into_tensor(
            #     self.forward_chage_Ht_prev, t_prev + 1, x_t.shape)
            # middle_params5 = extract_into_tensor(
            #     self.forward_chage_Ft, t, x_t.shape)
            # middle_params6 = extract_into_tensor(
            #     self.forward_chage_Ft_prev, t_prev + 1, x_t.shape)

            # k = torch.sqrt(1. - middle_params2 + middle_params6 * (self.forward_chage_trans_coeff_3 - self.forward_chage_trans_coeff_4) / self.forward_chage_trans_coeff_2) / \
            #     torch.sqrt(1. - middle_params1 + middle_params5 * (self.forward_chage_trans_coeff_3 -
            #                self.forward_chage_trans_coeff_4) / self.forward_chage_trans_coeff_2)
            # m = (torch.sqrt(middle_params2) - middle_params4 * torch.sqrt(self.forward_chage_trans_coeff_3) / self.forward_chage_trans_coeff_1) - \
            #     k * (torch.sqrt(middle_params1) - middle_params3 * torch.sqrt(
            #         self.forward_chage_trans_coeff_3) / self.forward_chage_trans_coeff_1)
            # n = (middle_params4 * torch.sqrt(self.forward_chage_trans_coeff_4) / self.forward_chage_trans_coeff_1) - \
            #     k * (middle_params3 * torch.sqrt(self.forward_chage_trans_coeff_4) /
            #          self.forward_chage_trans_coeff_1)

            # xt_prev = k * x_t + m * x_recon + n * z2
            xt_prev = x_t

        else:
            middle_params1 = extract_into_tensor(
                self.alphas_cumprod, t, x_t.shape)
            middle_params2 = extract_into_tensor(
                self.alphas_cumprod_prev, t_prev + 1, x_t.shape)

            k = torch.sqrt(1. - middle_params2) / \
                torch.sqrt(1. - middle_params1)
            m = torch.sqrt(middle_params2) - k * torch.sqrt(middle_params1)

            xt_prev = k * x_t + m * x_recon

        if return_codebook_ids:
            return xt_prev, logits.argmax(dim=1)
        if return_x0:
            return xt_prev, x_recon
        else:
            return xt_prev

    @torch.no_grad()
    def ddim_progressive_denoising_without_trans(self, cond1, cond2, shape, skip=1, verbose=True, callback=None, quantize_denoised=False,
                                   img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                                   score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                                   log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps + self.edit_t1
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []

        # 初始化情况下image就是一个服从正态分布的噪声图
        self.noise = img
        if start_T is not None:
            timesteps = min(timesteps, start_T)
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        main_timesteps = list(range(skip-1, timesteps, skip))
        main_timesteps_prev = [-1] + main_timesteps[:-1]

        # 创建迭代器（保持原有风格）
        iterator = tqdm(
            reversed(list(zip(main_timesteps, main_timesteps_prev))),
            desc='Progressive Generation',
            total=len(main_timesteps)
        ) if verbose else reversed(list(zip(main_timesteps, main_timesteps_prev)))

        # iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
        #                 total=timesteps) if verbose else reversed(
        #     range(0, timesteps))

        for i, i_prev in iterator:
            # print(i)
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            ts_prev = torch.full(
                (b,), i_prev, device=self.device, dtype=torch.long)

            img, x0_partial = self.ddim_p_sample_without_trans(img, cond1, cond2, ts, ts_prev,
                                                 clip_denoised=self.clip_denoised,
                                                 quantize_denoised=quantize_denoised, return_x0=True,
                                                 temperature=temperature[i], noise_dropout=noise_dropout,
                                                 score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)

        return img, intermediates

class DiffusionWrapper_AsyDiff_model_eval_Gencase(pl.LightningModule):
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
        # t = t.to(torch.float32)
        t = t.to(device)

        out = self.diffusion_model(lrms=zcs, pan=zcf, xt=x, timesteps=t)

        return out
