#!/usr/bin/env python3
import gc
import glob
import math
import os
import random
import shutil
import time
import warnings
from pathlib import Path

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import psutil
import tensorboardX
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
import tqdm
from packaging import version as pver
from rich.console import Console
from torch_ema import ExponentialMovingAverage
from torchmetrics import PearsonCorrCoef


def get_pos_neg_text_embeddings(embeddings, azimuth_val, opt):
    if azimuth_val >= -90 and azimuth_val < 90:
        if azimuth_val >= 0:
            r = 1 - azimuth_val / 90
        else:
            r = 1 + azimuth_val / 90
        start_z = embeddings["front"]
        end_z = embeddings["side"]
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings["front"], embeddings["side"]], dim=0)
        if r > 0.8:
            front_neg_w = 0.0
        else:
            front_neg_w = math.exp(-r * opt.front_decay_factor) * opt.negative_w
        if r < 0.2:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-(1 - r) * opt.side_decay_factor) * opt.negative_w

        weights = torch.tensor([1.0, front_neg_w, side_neg_w])
    else:
        if azimuth_val >= 0:
            r = 1 - (azimuth_val - 90) / 90
        else:
            r = 1 + (azimuth_val + 90) / 90
        start_z = embeddings["side"]
        end_z = embeddings["back"]
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings["side"], embeddings["front"]], dim=0)
        front_neg_w = opt.negative_w
        if r > 0.8:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-r * opt.side_decay_factor) * opt.negative_w / 2

        weights = torch.tensor([1.0, side_neg_w, front_neg_w])
    return text_z, weights.to(text_z.device)


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse("1.10"):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing="ij")


def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


@torch.amp.autocast("cuda", enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None):
    """get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    """

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(
        torch.linspace(0, W - 1, W, device=device),
        torch.linspace(0, H - 1, H, device=device),
    )
    i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H * W)

        if error_map is None:
            inds = torch.randint(0, H * W, size=[N], device=device)  # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(
                error_map.to(device), N, replacement=False
            )  # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = (
                inds_coarse // 128,
                inds_coarse % 128,
            )  # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (
                (inds_x * sx + torch.rand(B, N, device=device) * sx)
                .long()
                .clamp(max=H - 1)
            )
            inds_y = (
                (inds_y * sy + torch.rand(B, N, device=device) * sy)
                .long()
                .clamp(max=W - 1)
            )
            inds = inds_x * W + inds_y

            results["inds_coarse"] = inds_coarse  # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results["inds"] = inds

    else:
        inds = torch.arange(H * W, device=device).expand([B, H * W])

    zs = -torch.ones_like(i)
    xs = -(i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    # directions = safe_normalize(directions)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)  # (B, N, 3)

    rays_o = poses[..., :3, 3]  # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)  # [B, N, 3]

    results["rays_o"] = rays_o
    results["rays_d"] = rays_d

    return results


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x**0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


class Trainer(object):
    def __init__(
        self,
        argv,  # command line args
        name,  # name of this experiment
        opt,  # extra conf
        model,  # network
        guidance,  # guidance network
        criterion=None,  # loss function, if None, assume inline implementation in train_step
        optimizer=None,  # optimizer
        ema_decay=None,  # if use EMA, set the decay
        lr_scheduler=None,  # scheduler
        metrics=[],  # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
        local_rank=0,  # which GPU am I
        world_size=1,  # total num of GPUs
        device=None,  # device to use, usually setting to None is OK. (auto choose device)
        mute=False,  # whether to mute all print
        fp16=False,  # amp optimize level
        max_keep_ckpt=2,  # max num of saved ckpts in disk
        workspace="workspace",  # workspace to save logs & ckpts
        best_mode="min",  # the smaller/larger result, the better
        use_loss_as_metric=True,  # use loss as the first metric
        report_metric_at_train=False,  # also report metrics at training
        use_checkpoint="latest",  # which ckpt to use at init time
        use_tensorboardX=True,  # whether to use tensorboard for logging
        scheduler_update_every_step=False,  # whether to call scheduler.step() after every train step
    ):

        self.argv = argv
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = (
            device
            if device is not None
            else torch.device(
                f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
            )
        )
        self.console = Console()

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank]
            )
        self.model = model

        # guide model
        self.guidance = guidance
        self.embeddings = {}

        # text prompt / images
        if self.guidance is not None:
            for key in self.guidance:
                for p in self.guidance[key].parameters():
                    p.requires_grad = False
                self.embeddings[key] = {}
            self.prepare_embeddings()

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if self.opt.images is not None:
            self.pearson = PearsonCorrCoef().to(self.device)

        self.optimizer = optimizer(self.model)

        self.lr_scheduler = lr_scheduler(self.optimizer)

        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.total_train_t = 0
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = "min"

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, "checkpoints")
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

            # Save a copy of image_config in the experiment workspace
            if opt.image_config is not None:
                shutil.copyfile(
                    opt.image_config,
                    os.path.join(self.workspace, os.path.basename(opt.image_config)),
                )

            # Save a copy of images in the experiment workspace
            if opt.images is not None:
                for image_file in opt.images:
                    shutil.copyfile(
                        image_file,
                        os.path.join(self.workspace, os.path.basename(image_file)),
                    )

        self.log(f"[INFO] Cmdline: {self.argv}")
        self.log(f"[INFO] opt: {self.opt}")
        self.log(
            f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}'
        )
        self.log(
            f"[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}"
        )

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else:  # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    # calculate the text embs.
    @torch.no_grad()
    def prepare_embeddings(self):

        # text embeddings (stable-diffusion)
        if self.opt.text is not None:
            self.embeddings["SD"]["default"] = self.guidance["SD"].get_text_embeds(
                [self.opt.text]
            )
            self.embeddings["SD"]["uncond"] = self.guidance["SD"].get_text_embeds(
                [self.opt.negative]
            )

            for d in ["front", "side", "back"]:
                self.embeddings["SD"][d] = self.guidance["SD"].get_text_embeds(
                    [f"{self.opt.text}, {d} view"]
                )

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                # print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush()  # write immediately to file

    ### ------------------------------

    def train_step(self, data, save_guidance_path: Path = None):
        """
        Args:
            save_guidance_path: an image that combines the NeRF render, the added latent noise,
                the denoised result and optionally the fully-denoised image.
        """

        # experiment iterations ratio
        # i.e. what proportion of this experiment have we completed (in terms of iterations) so far?
        exp_iter_ratio = (self.global_step - self.opt.exp_start_iter) / (
            self.opt.exp_end_iter - self.opt.exp_start_iter
        )

        # progressively relaxing view range
        if self.opt.progressive_view:
            r = min(1.0, self.opt.progressive_view_init_ratio + 2.0 * exp_iter_ratio)
            self.opt.phi_range = [
                self.opt.default_azimuth * (1 - r) + self.opt.full_phi_range[0] * r,
                self.opt.default_azimuth * (1 - r) + self.opt.full_phi_range[1] * r,
            ]
            self.opt.theta_range = [
                self.opt.default_polar * (1 - r) + self.opt.full_theta_range[0] * r,
                self.opt.default_polar * (1 - r) + self.opt.full_theta_range[1] * r,
            ]
            self.opt.radius_range = [
                self.opt.default_radius * (1 - r) + self.opt.full_radius_range[0] * r,
                self.opt.default_radius * (1 - r) + self.opt.full_radius_range[1] * r,
            ]
            self.opt.fovy_range = [
                self.opt.default_fovy * (1 - r) + self.opt.full_fovy_range[0] * r,
                self.opt.default_fovy * (1 - r) + self.opt.full_fovy_range[1] * r,
            ]

        # progressively increase max_level
        if self.opt.progressive_level:
            self.model.max_level = min(1.0, 0.25 + 2.0 * exp_iter_ratio)

        rays_o = data["rays_o"]  # [B, N, 3]
        rays_d = data["rays_d"]  # [B, N, 3]
        mvp = data["mvp"]  # [B, 4, 4]

        B, N = rays_o.shape[:2]
        H, W = data["H"], data["W"]

        # When ref_data has B images > opt.batch_size
        if B > self.opt.batch_size:
            # choose batch_size images out of those B images
            choice = torch.randperm(B)[: self.opt.batch_size]
            B = self.opt.batch_size
            rays_o = rays_o[choice]
            rays_d = rays_d[choice]
            mvp = mvp[choice]

        if exp_iter_ratio <= self.opt.latent_iter_ratio:
            ambient_ratio = 1.0
            shading = "normal"
            as_latent = True
            binarize = False
            bg_color = None

        else:
            if exp_iter_ratio <= self.opt.albedo_iter_ratio:
                ambient_ratio = 1.0
                shading = "albedo"
            else:
                # random shading
                ambient_ratio = (
                    self.opt.min_ambient_ratio
                    + (1.0 - self.opt.min_ambient_ratio) * random.random()
                )
                rand = random.random()
                if rand >= (1.0 - self.opt.textureless_ratio):
                    shading = "textureless"
                else:
                    shading = "lambertian"

            as_latent = False

            # random weights binarization (like mobile-nerf) [NOT WORKING NOW]
            # binarize_thresh = min(0.5, -0.5 + self.global_step / self.opt.iters)
            # binarize = random.random() < binarize_thresh
            binarize = False

            # random background
            rand = random.random()
            if self.opt.bg_radius > 0 and rand > 0.5:
                bg_color = None  # use bg_net
            else:
                bg_color = torch.rand(3).to(self.device)  # single color random bg

        outputs = self.model.render(
            rays_o,
            rays_d,
            mvp,
            H,
            W,
            staged=False,
            perturb=True,
            bg_color=bg_color,
            ambient_ratio=ambient_ratio,
            shading=shading,
            binarize=binarize,
        )
        pred_depth = outputs["depth"].reshape(B, 1, H, W)
        pred_mask = outputs["weights_sum"].reshape(B, 1, H, W)
        if "normal_image" in outputs:
            pred_normal = outputs["normal_image"].reshape(B, H, W, 3)

        if as_latent:
            # abuse normal & mask as latent code for faster geometry initialization (ref: fantasia3D)
            pred_rgb = (
                torch.cat(
                    [outputs["image"], outputs["weights_sum"].unsqueeze(-1)], dim=-1
                )
                .reshape(B, H, W, 4)
                .permute(0, 3, 1, 2)
                .contiguous()
            )  # [B, 4, H, W]
        else:
            pred_rgb = (
                outputs["image"].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
            )  # [B, 3, H, W]

        # novel view loss
        loss = 0
        # interpolate text_z
        azimuth = data["azimuth"]  # [-180, 180]

        # ENHANCE: remove loop to handle batch size > 1
        text_z = [self.embeddings["SD"]["uncond"]] * azimuth.shape[0]
        self.opt.perpneg = False

        for b in range(azimuth.shape[0]):
            if azimuth[b] >= -90 and azimuth[b] < 90:
                if azimuth[b] >= 0:
                    r = 1 - azimuth[b] / 90
                else:
                    r = 1 + azimuth[b] / 90
                start_z = self.embeddings["SD"]["front"]
                end_z = self.embeddings["SD"]["side"]
            else:
                if azimuth[b] >= 0:
                    r = 1 - (azimuth[b] - 90) / 90
                else:
                    r = 1 + (azimuth[b] + 90) / 90
                start_z = self.embeddings["SD"]["side"]
                end_z = self.embeddings["SD"]["back"]
            text_z.append(r * start_z + (1 - r) * end_z)

        text_z = torch.cat(text_z, dim=0)
        if self.opt.perpneg:
            loss = loss + self.guidance["SD"].train_step_perpneg(
                text_z,
                weights,
                pred_rgb,
                as_latent=as_latent,
                guidance_scale=self.opt.guidance_scale,
                grad_scale=self.opt.lambda_guidance,
                save_guidance_path=save_guidance_path,
            )
        else:
            loss = loss + self.guidance["SD"].train_step(
                text_z,
                pred_rgb,
                as_latent=as_latent,
                guidance_scale=self.opt.guidance_scale,
                grad_scale=self.opt.lambda_guidance,
                save_guidance_path=save_guidance_path,
            )

        # regularizations
        if self.opt.lambda_opacity > 0:
            loss_opacity = (outputs["weights_sum"] ** 2).mean()
            loss = loss + self.opt.lambda_opacity * loss_opacity

        if self.opt.lambda_entropy > 0:
            alphas = outputs["weights"].clamp(1e-5, 1 - 1e-5)
            # alphas = alphas ** 2 # skewed entropy, favors 0 over 1
            loss_entropy = (
                -alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)
            ).mean()
            lambda_entropy = self.opt.lambda_entropy * min(
                1, 2 * self.global_step / self.opt.iters
            )
            loss = loss + lambda_entropy * loss_entropy

        if self.opt.lambda_2d_normal_smooth > 0 and "normal_image" in outputs:
            # pred_vals = outputs['normal_image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
            # smoothed_vals = TF.gaussian_blur(pred_vals.detach(), kernel_size=9)
            # loss_smooth = F.mse_loss(pred_vals, smoothed_vals)
            # total-variation
            loss_smooth = (
                pred_normal[:, 1:, :, :] - pred_normal[:, :-1, :, :]
            ).square().mean() + (
                pred_normal[:, :, 1:, :] - pred_normal[:, :, :-1, :]
            ).square().mean()
            loss = loss + self.opt.lambda_2d_normal_smooth * loss_smooth

        if self.opt.lambda_orient > 0 and "loss_orient" in outputs:
            loss_orient = outputs["loss_orient"]
            loss = loss + self.opt.lambda_orient * loss_orient

        if self.opt.lambda_3d_normal_smooth > 0 and "loss_normal_perturb" in outputs:
            loss_normal_perturb = outputs["loss_normal_perturb"]
            loss = loss + self.opt.lambda_3d_normal_smooth * loss_normal_perturb

        return pred_rgb, pred_depth, loss

    def post_train_step(self):

        # unscale grad before modifying it!
        # ref: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
        self.scaler.unscale_(self.optimizer)

        # clip grad
        if self.opt.grad_clip >= 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.opt.grad_clip)

        if not self.opt.dmtet and self.opt.backbone == "grid":

            if self.opt.lambda_tv > 0:
                lambda_tv = (
                    min(1.0, self.global_step / (0.5 * self.opt.iters))
                    * self.opt.lambda_tv
                )
                self.model.encoder.grad_total_variation(
                    lambda_tv, None, self.model.bound
                )
            if self.opt.lambda_wd > 0:
                self.model.encoder.grad_weight_decay(self.opt.lambda_wd)

    def eval_step(self, data):

        rays_o = data["rays_o"]  # [B, N, 3]
        rays_d = data["rays_d"]  # [B, N, 3]
        mvp = data["mvp"]

        B, N = rays_o.shape[:2]
        H, W = data["H"], data["W"]

        shading = data["shading"] if "shading" in data else "albedo"
        ambient_ratio = data["ambient_ratio"] if "ambient_ratio" in data else 1.0
        light_d = data["light_d"] if "light_d" in data else None

        outputs = self.model.render(
            rays_o,
            rays_d,
            mvp,
            H,
            W,
            staged=True,
            perturb=False,
            bg_color=None,
            light_d=light_d,
            ambient_ratio=ambient_ratio,
            shading=shading,
        )
        pred_rgb = outputs["image"].reshape(B, H, W, 3)
        pred_depth = outputs["depth"].reshape(B, H, W)

        # dummy
        loss = torch.zeros([1], device=pred_rgb.device, dtype=pred_rgb.dtype)

        return pred_rgb, pred_depth, loss

    def test_step(self, data, bg_color=None, perturb=False):
        rays_o = data["rays_o"]  # [B, N, 3]
        rays_d = data["rays_d"]  # [B, N, 3]
        mvp = data["mvp"]

        B, N = rays_o.shape[:2]
        H, W = data["H"], data["W"]

        if bg_color is not None:
            bg_color = bg_color.to(rays_o.device)

        shading = data["shading"] if "shading" in data else "albedo"
        ambient_ratio = data["ambient_ratio"] if "ambient_ratio" in data else 1.0
        light_d = data["light_d"] if "light_d" in data else None

        outputs = self.model.render(
            rays_o,
            rays_d,
            mvp,
            H,
            W,
            staged=True,
            perturb=perturb,
            light_d=light_d,
            ambient_ratio=ambient_ratio,
            shading=shading,
            bg_color=bg_color,
        )

        pred_rgb = outputs["image"].reshape(B, H, W, 3)
        pred_depth = outputs["depth"].reshape(B, H, W)

        return pred_rgb, pred_depth, None

    def save_mesh(self, loader=None, save_path=None):

        if save_path is None:
            save_path = os.path.join(self.workspace, "mesh")

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(save_path, exist_ok=True)

        self.model.export_mesh(
            save_path,
            resolution=self.opt.mcubes_resolution,
            decimate_target=self.opt.decimate_target,
        )

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, test_loader, max_epochs):

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(
                os.path.join(self.workspace, "run", self.name)
            )

        start_t = time.time()

        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader, max_epochs)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.opt.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

            if self.epoch % self.opt.test_interval == 0 or self.epoch == max_epochs:
                self.test(test_loader)

        end_t = time.time()

        self.total_train_t = end_t - start_t + self.total_train_t

        self.log(f"[INFO] training takes {(self.total_train_t)/ 60:.4f} minutes.")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True):

        if save_path is None:
            save_path = os.path.join(self.workspace, "results")

        if name is None:
            name = f"{self.name}_ep{self.epoch:04d}"

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(
            total=len(loader) * loader.batch_size,
            bar_format="{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []

        with torch.no_grad():

            for i, data in enumerate(loader):

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, _ = self.test_step(data)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth - pred_depth.min()) / (
                    pred_depth.max() - pred_depth.min() + 1e-6
                )
                pred_depth = (pred_depth * 255).astype(np.uint8)

                if write_video:
                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
                else:
                    cv2.imwrite(
                        os.path.join(save_path, f"{name}_{i:04d}_rgb.png"),
                        cv2.cvtColor(pred, cv2.COLOR_RGB2BGR),
                    )
                    cv2.imwrite(
                        os.path.join(save_path, f"{name}_{i:04d}_depth.png"), pred_depth
                    )

                pbar.update(loader.batch_size)

        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)

            imageio.mimwrite(
                os.path.join(save_path, f"{name}_rgb.mp4"),
                all_preds,
                fps=25,
                quality=8,
                macro_block_size=1,
            )
            imageio.mimwrite(
                os.path.join(save_path, f"{name}_depth.mp4"),
                all_preds_depth,
                fps=25,
                quality=8,
                macro_block_size=1,
            )

        self.log(f"==> Finished Test.")

    def train_one_epoch(self, loader, max_epochs):
        self.log(
            f"==> [{time.strftime('%Y-%m-%d_%H-%M-%S')}] Start Training {self.workspace} Epoch {self.epoch}/{max_epochs}, lr={self.optimizer.param_groups[0]['lr']:.6f} ..."
        )

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(
                total=len(loader) * loader.batch_size,
                bar_format="{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )

        self.local_step = 0

        if self.opt.save_guidance:
            save_guidance_folder = Path(self.workspace) / "guidance"
            save_guidance_folder.mkdir(parents=True, exist_ok=True)

        for data in loader:

            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                if self.opt.save_guidance and (
                    self.global_step % self.opt.save_guidance_interval == 0
                ):
                    save_guidance_path = (
                        save_guidance_folder / f"step_{self.global_step:07d}.png"
                    )
                else:
                    save_guidance_path = None
                pred_rgbs, pred_depths, loss = self.train_step(
                    data, save_guidance_path=save_guidance_path
                )

            # hooked grad clipping for RGB space
            if self.opt.grad_clip_rgb >= 0:

                def _hook(grad):
                    if self.opt.fp16:
                        # correctly handle the scale
                        grad_scale = self.scaler._get_scale_async()
                        return grad.clamp(
                            grad_scale * -self.opt.grad_clip_rgb,
                            grad_scale * self.opt.grad_clip_rgb,
                        )
                    else:
                        return grad.clamp(
                            -self.opt.grad_clip_rgb, self.opt.grad_clip_rgb
                        )

                pred_rgbs.register_hook(_hook)
                # pred_rgbs.retain_grad()

            self.scaler.scale(loss).backward()

            self.post_train_step()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                # if self.report_metric_at_train:
                #     for metric in self.metrics:
                #         metric.update(preds, truths)

                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar(
                        "train/lr",
                        self.optimizer.param_groups[0]["lr"],
                        self.global_step,
                    )

                if self.scheduler_update_every_step:
                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}"
                    )
                else:
                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})"
                    )
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        cpu_mem, gpu_mem = get_CPU_mem(), get_GPU_mem()[0]
        self.log(
            f"==> [{time.strftime('%Y-%m-%d_%H-%M-%S')}] Finished Epoch {self.epoch}/{max_epochs}. CPU={cpu_mem:.1f}GB, GPU={gpu_mem:.1f}GB."
        )

    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate {self.workspace} at epoch {self.epoch} ...")

        if name is None:
            name = f"{self.name}_ep{self.epoch:04d}"

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(
                total=len(loader) * loader.batch_size,
                bar_format="{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )

        with torch.no_grad():
            self.local_step = 0

            for data in loader:
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, loss = self.eval_step(data)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size

                    preds_list = [
                        torch.zeros_like(preds).to(self.device)
                        for _ in range(self.world_size)
                    ]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [
                        torch.zeros_like(preds_depth).to(self.device)
                        for _ in range(self.world_size)
                    ]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:

                    # save image
                    save_path = os.path.join(
                        self.workspace,
                        "validation",
                        f"{name}_{self.local_step:04d}_rgb.png",
                    )
                    save_path_depth = os.path.join(
                        self.workspace,
                        "validation",
                        f"{name}_{self.local_step:04d}_depth.png",
                    )

                    # self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    pred = preds[0].detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)

                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    pred_depth = (pred_depth - pred_depth.min()) / (
                        pred_depth.max() - pred_depth.min() + 1e-6
                    )
                    pred_depth = (pred_depth * 255).astype(np.uint8)

                    cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(save_path_depth, pred_depth)

                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})"
                    )
                    pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(
                    result if self.best_mode == "min" else -result
                )  # if max mode, use -result
            else:
                self.stats["results"].append(
                    average_loss
                )  # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False):

        if name is None:
            name = f"{self.name}_ep{self.epoch:04d}"

        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "stats": self.stats,
        }

        if self.model.cuda_ray:
            state["mean_density"] = self.model.mean_density

        if self.opt.dmtet:
            state["tet_scale"] = self.model.tet_scale.cpu().numpy()

        if full:
            state["optimizer"] = self.optimizer.state_dict()
            state["lr_scheduler"] = self.lr_scheduler.state_dict()
            state["scaler"] = self.scaler.state_dict()
            if self.ema is not None:
                state["ema"] = self.ema.state_dict()

        if not best:

            state["model"] = self.model.state_dict()

            file_path = f"{name}.pth"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = os.path.join(
                    self.ckpt_path, self.stats["checkpoints"].pop(0)
                )
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, os.path.join(self.ckpt_path, file_path))

        else:
            if len(self.stats["results"]) > 0:
                # always save best since loss cannot reflect performance.
                if True:
                    # self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    # self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state["model"] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.log(
                    f"[WARN] no evaluated results found, skip saving best checkpoint."
                )

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f"{self.ckpt_path}/*.pth"))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if "model" not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(
            checkpoint_dict["model"], strict=False
        )
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and "ema" in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict["ema"])
                self.log("[INFO] loaded EMA.")
            except:
                self.log("[WARN] failed to loaded EMA.")

        if self.model.cuda_ray:
            if "mean_density" in checkpoint_dict:
                self.model.mean_density = checkpoint_dict["mean_density"]

        if self.opt.dmtet:
            if "tet_scale" in checkpoint_dict:
                new_scale = torch.from_numpy(checkpoint_dict["tet_scale"]).to(
                    self.device
                )
                self.model.verts *= new_scale / self.model.tet_scale
                self.model.tet_scale = new_scale

        if model_only:
            return

        self.stats = checkpoint_dict["stats"]
        self.epoch = checkpoint_dict["epoch"]
        self.global_step = checkpoint_dict["global_step"]
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer and "optimizer" in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and "lr_scheduler" in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict["lr_scheduler"])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.scaler and "scaler" in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict["scaler"])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")


def get_CPU_mem():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3


def get_GPU_mem():
    num = torch.cuda.device_count()
    mem, mems = 0, []
    for i in range(num):
        mem_free, mem_total = torch.cuda.mem_get_info(i)
        mems.append(int(((mem_total - mem_free) / 1024**3) * 1000) / 1000)
        mem += mems[-1]
    return mem, mems
