#!/usr/bin/env python3
import math
from typing import Optional
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from data_utils import colors_from_spherical_harmonics
from data_utils import load_gaussians_from_ply
from pytorch3d.ops.knn import knn_points
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.transforms import quaternion_to_matrix


class Gaussians:

    def __init__(
        self,
        init_type: str,
        device: str,
        load_path: Optional[str] = None,
        num_points: Optional[int] = None,
        isotropic: Optional[bool] = None,
    ):

        self.device = device
        if self.device not in ("cpu", "cuda"):
            raise ValueError(f"Unsupported device: {self.device}")

        if init_type == "gaussians":
            if isotropic is not None:
                raise ValueError(
                    (
                        "Isotropy/Anisotropy will be determined from pre-trained gaussians. "
                        "Please set isotropic to None."
                    )
                )
            if load_path is None:
                raise ValueError

            data, is_isotropic = self._load_gaussians(load_path)
            self.is_isotropic = is_isotropic

        elif init_type == "points":
            if isotropic is not None and type(isotropic) is not bool:
                raise TypeError("isotropic must be either None or True or False.")
            if load_path is None:
                raise ValueError

            if isotropic is None:
                self.is_isotropic = False
            else:
                self.is_isotropic = isotropic

            data = self._load_points(load_path)

        elif init_type == "random":
            if isotropic is not None and type(isotropic) is not bool:
                raise TypeError("isotropic must be either None or True or False.")
            if num_points is None:
                raise ValueError

            if isotropic is None:
                self.is_isotropic = False
            else:
                self.is_isotropic = isotropic

            data = self._load_random(num_points)

        else:
            raise ValueError(f"Invalid init_type: {init_type}")

        self.pre_act_quats: torch.Tensor = data["pre_act_quats"]
        self.means: torch.Tensor = data["means"]
        self.pre_act_scales: torch.Tensor = data["pre_act_scales"]
        self.colors: torch.Tensor = data["colors"]
        self.pre_act_opacities: torch.Tensor = data["pre_act_opacities"]

        # [Q 1.3.1] NOTE: Uncomment spherical harmonics code for question 1.3.1
        if data.get("spherical_harmonics") is not None:
            self.spherical_harmonics = data["spherical_harmonics"]

        if self.device == "cuda":
            self.to_cuda()

    def __len__(self) -> int:
        return len(self.means)

    def _load_gaussians(self, ply_path: str) -> Tuple[dict[str, torch.Tensor], bool]:

        data = dict()
        ply_gaussians = load_gaussians_from_ply(ply_path)

        data["means"] = torch.tensor(ply_gaussians["xyz"])
        data["pre_act_quats"] = torch.tensor(ply_gaussians["rot"])
        data["pre_act_scales"] = torch.tensor(ply_gaussians["scale"])
        data["pre_act_opacities"] = torch.tensor(ply_gaussians["opacity"]).squeeze()
        data["colors"] = torch.tensor(ply_gaussians["dc_colors"])

        # [Q 1.3.1] NOTE: Uncomment spherical harmonics code for question 1.3.1
        data["spherical_harmonics"] = torch.tensor(ply_gaussians["sh"])

        if data["pre_act_scales"].shape[1] != 3:
            raise NotImplementedError("Currently does not support isotropic")

        is_isotropic = False

        return data, is_isotropic

    def _load_points(self, path: str) -> dict[str, torch.Tensor]:

        data = dict()
        means = np.load(path)

        # Initializing means using the provided point cloud
        data["means"] = torch.tensor(means.astype(np.float32))  # (N, 3)

        # Initializing opacities such that all when sigmoid is applied to pre_act_opacities,
        # we will have a opacity value close to (but less than) 1.0
        data["pre_act_opacities"] = 8.0 * torch.ones(
            (len(means),), dtype=torch.float32
        )  # (N,)

        # Initializing colors randomly
        data["colors"] = torch.rand((len(means), 3), dtype=torch.float32)  # (N, 3)

        # Initializing quaternions to be the identity quaternion
        quats = torch.zeros((len(means), 4), dtype=torch.float32)  # (N, 4)
        quats[:, 0] = 1.0
        data["pre_act_quats"] = quats  # (N, 4)

        # Initializing scales using the mean distance of each point to its 50 nearest points
        dists, _, _ = knn_points(
            data["means"].unsqueeze(0), data["means"].unsqueeze(0), K=50
        )
        data["pre_act_scales"] = torch.log(torch.mean(dists[0], dim=1)).unsqueeze(
            1
        )  # (N, 1)

        if not self.is_isotropic:
            data["pre_act_scales"] = data["pre_act_scales"].repeat(1, 3)  # (N, 3)

        return data

    def _load_random(self, num_points: int) -> dict[str, torch.Tensor]:

        data = dict()

        # Initializing means randomly
        data["means"] = torch.randn((num_points, 3)).to(torch.float32) * 0.2  # (N, 3)

        # Initializing opacities such that all when sigmoid is applied to pre_act_opacities,
        # we will have a opacity value close to (but less than) 1.0
        data["pre_act_opacities"] = 8.0 * torch.ones(
            (num_points,), dtype=torch.float32
        )  # (N,)

        # Initializing colors randomly
        data["colors"] = torch.rand((num_points, 3), dtype=torch.float32)  # (N, 3)

        # Initializing quaternions to be the identity quaternion
        quats = torch.zeros((num_points, 4), dtype=torch.float32)  # (N, 4)
        quats[:, 0] = 1.0
        data["pre_act_quats"] = quats  # (N, 4)

        # Initializing scales randomly
        data["pre_act_scales"] = torch.log(
            (torch.rand((num_points, 1), dtype=torch.float32) + 1e-6) * 0.01
        )

        if not self.is_isotropic:
            data["pre_act_scales"] = data["pre_act_scales"].repeat(1, 3)  # (N, 3)

        return data

    def _compute_jacobian(
        self, means_3D: torch.Tensor, camera: PerspectiveCameras, img_size: Tuple
    ) -> torch.Tensor:

        if camera.in_ndc():
            raise RuntimeError

        fx, fy = camera.focal_length.flatten()
        W, H = img_size

        half_tan_fov_x = 0.5 * W / fx
        half_tan_fov_y = 0.5 * H / fy

        view_transform = camera.get_world_to_view_transform()
        means_view_space = view_transform.transform_points(means_3D)

        tx = means_view_space[:, 0]
        ty = means_view_space[:, 1]
        tz = means_view_space[:, 2]
        tz2 = tz * tz

        lim_x = 1.3 * half_tan_fov_x
        lim_y = 1.3 * half_tan_fov_y

        tx = torch.clamp(tx / tz, -lim_x, lim_x) * tz
        ty = torch.clamp(ty / tz, -lim_y, lim_y) * tz

        J = torch.zeros((len(tx), 2, 3))  # (N, 2, 3)
        J = J.to(self.device)

        J[:, 0, 0] = fx / tz
        J[:, 1, 1] = fy / tz
        J[:, 0, 2] = -(fx * tx) / tz2
        J[:, 1, 2] = -(fy * ty) / tz2

        return J  # (N, 2, 3)

    def check_if_trainable(self) -> None:

        attrs = ["means", "pre_act_scales", "colors", "pre_act_opacities"]
        if not self.is_isotropic:
            attrs += ["pre_act_quats"]

        for attr in attrs:
            param = getattr(self, attr)
            if not getattr(param, "requires_grad", False):
                raise Exception(
                    "Please use function make_trainable to make parameters trainable"
                )

        if self.is_isotropic and self.pre_act_quats.requires_grad:
            raise RuntimeError(
                "You do not need to optimize quaternions in isotropic mode."
            )

    def to_cuda(self) -> None:

        self.pre_act_quats = self.pre_act_quats.cuda()
        self.means = self.means.cuda()
        self.pre_act_scales = self.pre_act_scales.cuda()
        self.colors = self.colors.cuda()
        self.pre_act_opacities = self.pre_act_opacities.cuda()

        # [Q 1.3.1] NOTE: Uncomment spherical harmonics code for question 1.3.1
        self.spherical_harmonics = self.spherical_harmonics.cuda()

    def compute_cov_3D(self, quats: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """
        Computes the covariance matrices of 3D Gaussians using equation (6) of the 3D
        Gaussian Splatting paper.

        Link: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_low.pdf

        Args:
            quats   :   A torch.Tensor of shape (N, 4) representing the rotation
                        components of 3D Gaussians in quaternion form.
            scales  :   If self.is_isotropic is True, scales is will be a torch.Tensor of shape (N, 1)
                        If self.is_isotropic is False, scales is will be a torch.Tensor of shape (N, 3).
                        Represents the scaling components of the 3D Gaussians.

        Returns:
            cov_3D  :   A torch.Tensor of shape (N, 3, 3)
        """
        # NOTE: While technically you can use (almost) the same code for the
        # isotropic and anisotropic case, can you think of a more efficient
        # code for the isotropic case?

        # HINT: Are quats ever used or optimized for isotropic gaussians? What will their value be?
        # Based on your answers, can you write a more efficient code for the isotropic case?
        if self.is_isotropic:
            ### YOUR CODE HERE ###
            cov_3D = (scales**2).view(-1, 1, 1) * torch.eye(
                3,
                device=scales.device,
                dtype=scales.dtype,
            )  # (N, 3, 3)

        else:
            # HINT: You can use a function from pytorch3d to convert quaternions to rotation matrices.
            ### YOUR CODE HERE ###
            SST = torch.diag_embed(scales**2).to(
                device=scales.device,
                dtype=scales.dtype,
            )

            # q = (w, x, y, z)
            # SO(3) = [[1 - 2y^2 - 2z^2, 2xy - 2wz,       2xz + 2wy],
            #          [2xy + 2wz,       1 - 2x^2 - 2z^2, 2yz - 2wx],
            #          [2xz - 2wy,       2yz + 2wx,       1 - 2x^2 - 2y^2]]
            R = quaternion_to_matrix(quats)
            cov_3D = R @ SST @ R.mT  # (N, 3, 3)

        return cov_3D

    def compute_cov_2D(
        self,
        means_3D: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        camera: PerspectiveCameras,
        img_size: Tuple,
    ) -> torch.Tensor:
        """
        Computes the covariance matrices of 2D Gaussians using equation (5) of the 3D
        Gaussian Splatting paper.

        Link: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_low.pdf

        Args:
            quats       :   A torch.Tensor of shape (N, 4) representing the rotation
                            components of 3D Gaussians in quaternion form.
            scales      :   If self.is_isotropic is True, scales is will be a torch.Tensor of shape (N, 1)
                            If self.is_isotropic is False, scales is will be a torch.Tensor of shape (N, 3)
            camera      :   A pytorch3d PerspectiveCameras object
            img_size    :   A tuple representing the (width, height) of the image

        Returns:
            cov_3D  :   A torch.Tensor of shape (N, 3, 3)
        """
        ### YOUR CODE HERE ###
        N = means_3D.shape[0]

        # HINT: For computing the jacobian J, can you find a function in this file that can help?
        J = self._compute_jacobian(means_3D, camera, img_size)  # (N, 2, 3)

        ### YOUR CODE HERE ###
        # HINT: Can you extract the world to camera rotation matrix (W) from one of the inputs
        # of this function?
        W = (
            camera.get_world_to_view_transform()
            .get_matrix()[..., :3, :3]
            .expand(N, 3, 3)
        )  # (N, 3, 3)

        ### YOUR CODE HERE ###
        # HINT: Can you find a function in this file that can help?
        cov_3D = self.compute_cov_3D(quats, scales)  # (N, 3, 3)

        ### YOUR CODE HERE ###
        # HINT: Use the above three variables to compute cov_2D
        cov_2D = (J @ W @ cov_3D @ W.mT @ J.mT)[..., :2, :2]  # (N, 2, 2)

        # Post processing to make sure that each 2D Gaussian covers at least approximately 1 pixel
        cov_2D[:, 0, 0] += 0.3
        cov_2D[:, 1, 1] += 0.3

        return cov_2D

    @staticmethod
    def compute_means_2D(
        means_3D: torch.Tensor, camera: PerspectiveCameras
    ) -> torch.Tensor:
        """
        Computes the means of the projected 2D Gaussians given the means of the 3D Gaussians.

        Args:
            means_3D    :   A torch.Tensor of shape (N, 3) representing the means of
                            3D Gaussians.
            camera      :   A pytorch3d PerspectiveCameras object.

        Returns:
            means_2D    :   A torch.Tensor of shape (N, 2) representing the means of
                            2D Gaussians.
        """
        ### YOUR CODE HERE ###
        # HINT: Do note that means_2D have units of pixels. Hence, you must apply a
        # transformation that moves points in the world space to screen space.
        means_2D = camera.transform_points_screen(means_3D)[..., :2]  # (N, 2)
        return means_2D

    @staticmethod
    def invert_cov_2D(cov_2D: torch.Tensor) -> torch.Tensor:
        """
        Using the formula for inverse of a 2D matrix to invert the cov_2D matrix

        Args:
            cov_2D          :   A torch.Tensor of shape (N, 2, 2)

        Returns:
            cov_2D_inverse  :   A torch.Tensor of shape (N, 2, 2)
        """
        determinants = (
            cov_2D[:, 0, 0] * cov_2D[:, 1, 1] - cov_2D[:, 1, 0] * cov_2D[:, 0, 1]
        )
        determinants = determinants[:, None, None]  # (N, 1, 1)

        cov_2D_inverse = torch.zeros_like(cov_2D)  # (N, 2, 2)
        cov_2D_inverse[:, 0, 0] = cov_2D[:, 1, 1]
        cov_2D_inverse[:, 1, 1] = cov_2D[:, 0, 0]
        cov_2D_inverse[:, 0, 1] = -1.0 * cov_2D[:, 0, 1]
        cov_2D_inverse[:, 1, 0] = -1.0 * cov_2D[:, 1, 0]

        cov_2D_inverse = (1.0 / determinants) * cov_2D_inverse

        return cov_2D_inverse

    @staticmethod
    def evaluate_gaussian_2D(
        points_2D: torch.Tensor, means_2D: torch.Tensor, cov_2D_inverse: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the exponent (power) of 2D Gaussians.

        Args:
            points_2D       :   A torch.Tensor of shape (1, H*W, 2) containing the x, y points
                                corresponding to every pixel in an image. See function
                                compute_alphas in the class Scene to get more information
                                about how points_2D is created.
            means_2D        :   A torch.Tensor of shape (N, 1, 2) representing the means of
                                N 2D Gaussians.
            cov_2D_inverse  :   A torch.Tensor of shape (N, 2, 2) representing the
                                inverse of the covariance matrices of N 2D Gaussians.

        Returns:
            power           :   A torch.Tensor of shape (N, H*W) representing the computed
                                power of the N 2D Gaussians at every pixel location in an image.
        """
        ### YOUR CODE HERE ###
        # HINT: Refer to README for a relevant equation
        x_minus_mu = (points_2D - means_2D).unsqueeze(-1)  # (N, H*W, 2, 1)
        power = (
            -0.5 * x_minus_mu.mT @ cov_2D_inverse.unsqueeze(1) @ x_minus_mu
        )  # (N, H*W, 1, 1)

        return power.squeeze()  # (N, H*W)

    @staticmethod
    def apply_activations(
        pre_act_quats: torch.Tensor,
        pre_act_scales: torch.Tensor,
        pre_act_opacities: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Convert log scales to scales
        scales = torch.exp(pre_act_scales)

        # Normalize quaternions
        quats = torch.nn.functional.normalize(pre_act_quats)

        # Bound opacities between (0, 1)
        opacities = torch.sigmoid(pre_act_opacities)

        return quats, scales, opacities


class Scene:

    def __init__(self, gaussians: Gaussians):
        self.gaussians = gaussians
        self.device = self.gaussians.device

    def __repr__(self) -> str:
        return f"<Scene with {len(self.gaussians)} Gaussians>"

    def compute_depth_values(self, camera: PerspectiveCameras) -> torch.Tensor:
        """
        Computes the depth value of each 3D Gaussian.

        Args:
            camera  :   A pytorch3d PerspectiveCameras object.

        Returns:
            z_vals  :   A torch.Tensor of shape (N,) with the depth of each 3D Gaussian.
        """
        ### YOUR CODE HERE ###
        # HINT: You can use get the means of 3D Gaussians self.gaussians and calculate
        # the depth using the means and the camera
        means_3D = self.gaussians.means  # (N, 3)

        z_vals = camera.get_world_to_view_transform().transform_points(means_3D)[..., 2]

        return z_vals

    def get_idxs_to_filter_and_sort(self, z_vals: torch.Tensor) -> torch.Tensor:
        """
        Given depth values of Gaussians, return the indices to depth-wise sort
        Gaussians and at the same time remove invalid Gaussians.

        You can see the function render to see how the returned indices will be used.
        You are required to create a torch.Tensor idxs such that by using them in the
        function render we can arrange Gaussians (or equivalently their attributes such as
        the mean) in ascending order of depth. You should also make sure to not include indices
        that correspond to Gaussians with depth value less than 0.

        idxs should be torch.Tensor of dtype int64 with length N (N <= M, where M is the
        total number of Gaussians before filtering)

        Please refer to the README file for more details.
        """
        ### YOUR CODE HERE ###
        valid_idxs = torch.nonzero(z_vals >= 0).squeeze()

        sorted_order = torch.argsort(z_vals[valid_idxs], descending=False)
        idxs = valid_idxs[sorted_order]

        return idxs.to(dtype=torch.int64, device=z_vals.device)

    def compute_alphas(
        self,
        opacities: torch.Tensor,
        means_2D: torch.Tensor,
        cov_2D: torch.Tensor,
        img_size: Tuple,
    ) -> torch.Tensor:
        """
        Given some parameters of N ordered Gaussians, this function computes
        the alpha values.

        Args:
            opacities   :   A torch.Tensor of shape (N,) with the opacity value
                            of each Gaussian.
            means_2D    :   A torch.Tensor of shape (N, 2) with the means
                            of the 2D Gaussians.
            cov_2D      :   A torch.Tensor of shape (N, 2, 2) with the covariances
                            of the 2D Gaussians.
            img_size    :   The (width, height) of the image to be rendered.


        Returns:
            alphas      :   A torch.Tensor of shape (N, H, W) with the computed alpha
                            values for each of the N ordered Gaussians at every
                            pixel location.
        """
        W, H = img_size

        # point_2D contains all possible pixel locations in an image
        xs, ys = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
        points_2D = torch.stack((xs.flatten(), ys.flatten()), dim=1)  # (H*W, 2)
        points_2D = points_2D.to(self.device)

        points_2D = points_2D.unsqueeze(0)  # (1, H*W, 2)
        means_2D = means_2D.unsqueeze(1)  # (N, 1, 2)

        ### YOUR CODE HERE ###
        # HINT: Can you find a function in this file that can help?
        cov_2D_inverse = self.gaussians.invert_cov_2D(
            cov_2D
        )  # (N, 2, 2) TODO: Verify shape

        ### YOUR CODE HERE ###
        # HINT: Can you find a function in this file that can help?
        power = self.gaussians.evaluate_gaussian_2D(
            points_2D,
            means_2D,
            cov_2D_inverse,
        )  # (N, H*W)

        # Computing exp(power) with some post processing for numerical stability
        exp_power = torch.where(power > 0.0, 0.0, torch.exp(power))  # (N, H*W)

        ### YOUR CODE HERE ###
        # HINT: Refer to README for a relevant equation.
        alphas = opacities.unsqueeze(-1) * exp_power  # (N, H*W)
        alphas = torch.reshape(alphas, (-1, H, W))  # (N, H, W)

        # Post processing for numerical stability
        alphas = torch.minimum(alphas, torch.full_like(alphas, 0.99))
        alphas = torch.where(alphas < 1 / 255.0, 0.0, alphas)

        return alphas

    def compute_transmittance(
        self, alphas: torch.Tensor, start_transmittance: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Given the alpha values of N ordered Gaussians, this function computes
        the transmittance.

        The variable start_transmittance contains information about the transmittance
        at each pixel location BEFORE encountering the first Gaussian in the input.
        This variable is useful when computing transmittance in mini-batches because
        we would require information about the transmittance accumulated until the
        previous mini-batch to begin computing the transmittance for the current mini-batch.

        In case there were no previous mini-batches (or we are splatting in one-shot
        without using mini-batches), then start_transmittance will be None (since no Gaussians
        have been encountered so far). In this case, the code will use a starting
        transmittance value of 1.

        Args:
            alphas                  :   A torch.Tensor of shape (N, H, W) with the computed alpha
                                        values for each of the N ordered Gaussians at every
                                        pixel location.
            start_transmittance     :   Can be None or a torch.Tensor of shape (1, H, W). Please
                                        see the docstring for more information.

        Returns:
            transmittance           :   A torch.Tensor of shape (N, H, W) with the computed transmittance
                                        values for each of the N ordered Gaussians at every
                                        pixel location.
        """
        _, H, W = alphas.shape

        if start_transmittance is None:
            S = torch.ones((1, H, W), device=alphas.device, dtype=alphas.dtype)
        else:
            S = start_transmittance

        one_minus_alphas = 1.0 - alphas
        one_minus_alphas = torch.concat((S, one_minus_alphas), dim=0)  # (N+1, H, W)

        ### YOUR CODE HERE ###
        # HINT: Refer to README for a relevant equation.
        transmittance = torch.cumprod(one_minus_alphas, dim=0)[:-1]  # (N, H, W)

        # Post processing for numerical stability
        transmittance = torch.where(
            transmittance < 1e-4, 0.0, transmittance
        )  # (N, H, W)

        return transmittance

    def splat(
        self,
        camera: PerspectiveCameras,
        means_3D: torch.Tensor,
        z_vals: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        img_size: Tuple = (256, 256),
        start_transmittance: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given N ordered (depth-sorted) 3D Gaussians (or equivalently in our case,
        the parameters of the 3D Gaussians like means, quats etc.), this function splats
        them to the image plane to render an RGB image, depth map and a silhouette map.

        Args:
            camera                  :   A pytorch3d PerspectiveCameras object.
            means_3D                :   A torch.Tensor of shape (N, 3) with the means
                                        of the 3D Gaussians.
            z_vals                  :   A torch.Tensor of shape (N,) with the depths
                                        of the 3D Gaussians. # TODO: Verify shape
            quats                   :   A torch.Tensor of shape (N, 4) representing the rotation
                                        components of 3D Gaussians in quaternion form.
            scales                  :   A torch.Tensor of shape (N, 1) (if isotropic) or
                                        (N, 3) (if anisotropic) representing the scaling
                                        components of 3D Gaussians.
            colors                 :   A torch.Tensor of shape (N, 3) with the color contribution
                                        of each Gaussian.
            opacities               :   A torch.Tensor of shape (N,) with the opacity of each Gaussian.
            img_size                :   The (width, height) of the image.
            start_transmittance     :   Please see the docstring of the function compute_transmittance
                                        for information about this argument.

        Returns:
            image                   :   A torch.Tensor of shape (H, W, 3) with the rendered RGB color image.
            depth                   :   A torch.Tensor of shape (H, W, 1) with the rendered depth map.
            mask                    :   A torch.Tensor of shape (H, W, 1) with the rendered silhouette map.
            final_transmittance     :   A torch.Tensor of shape (1, H, W) representing the transmittance at
                                        each pixel computed using the N ordered Gaussians. This will be useful
                                        for mini-batch splatting in the next iteration.
        """
        # Step 1: Compute 2D gaussian parameters

        ### YOUR CODE HERE ###
        # HINT: Can you find a function in this file that can help?
        means_2D = self.gaussians.compute_means_2D(means_3D, camera)  # (N, 2)

        ### YOUR CODE HERE ###
        # HINT: Can you find a function in this file that can help?
        cov_2D = self.gaussians.compute_cov_2D(
            means_3D, quats, scales, camera, img_size
        )  # (N, 2, 2)

        # Step 2: Compute alpha maps for each gaussian

        ### YOUR CODE HERE ###
        # HINT: Can you find a function in this file that can help?
        alphas = self.compute_alphas(opacities, means_2D, cov_2D, img_size)  # (N, H, W)

        # Step 3: Compute transmittance maps for each gaussian

        ### YOUR CODE HERE ###
        # HINT: Can you find a function in this file that can help?
        transmittance = self.compute_transmittance(
            alphas, start_transmittance
        )  # (N, H, W)

        # Some unsqueezing to set up broadcasting for vectorized implementation.
        # You can selectively comment these out if you want to compute things
        # in a different way.
        z_vals = z_vals[:, None, None, None]  # (N, 1, 1, 1)
        alphas = alphas[..., None]  # (N, H, W, 1)
        colors = colors[:, None, None, :]  # (N, 1, 1, 3)
        transmittance = transmittance[..., None]  # (N, H, W, 1)

        # Step 4: Create image, depth and mask by computing the colors for each pixel.

        ### YOUR CODE HERE ###
        # HINT: Refer to README for a relevant equation
        image = (colors * alphas * transmittance).sum(dim=0)  # (H, W, 3)

        ### YOUR CODE HERE ###
        # HINT: Can you implement an equation inspired by the equation for color?
        depth = (z_vals * alphas * transmittance).sum(dim=0)  # (H, W, 1)

        ### YOUR CODE HERE ###
        # HINT: Can you implement an equation inspired by the equation for color?
        mask = (alphas * transmittance).sum(dim=0)  # (H, W, 1)

        final_transmittance = transmittance[-1, ..., 0].unsqueeze(0)  # (1, H, W)
        return image, depth, mask, final_transmittance

    def render(
        self,
        camera: PerspectiveCameras,
        per_splat: int = -1,
        img_size: Tuple = (256, 256),
        bg_color: Tuple = (0.0, 0.0, 0.0),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a scene represented by N 3D Gaussians, this function renders the RGB
        color image, the depth map and the silhouette map that can be observed
        from a given pytorch 3D camera.

        Args:
            camera      :   A pytorch3d PerspectiveCameras object.
            per_splat   :   Number of gaussians to splat in one function call. If set to -1,
                            then all gaussians in the scene are splat in a single function call.
                            If set to any other positive integer, then it determines the number of
                            gaussians to splat per function call (the last function call might splat
                            lesser number of gaussians). In general, the algorithm can run faster
                            if more gaussians are splat per function call, but at the cost of higher GPU
                            memory consumption.
            img_size    :   The (width, height) of the image to be rendered.
            bg_color    :   A tuple indicating the RGB color that the background should have.

        Returns:
            image       :   A torch.Tensor of shape (H, W, 3) with the rendered RGB color image.
            depth       :   A torch.Tensor of shape (H, W, 1) with the rendered depth map.
            mask        :   A torch.Tensor of shape (H, W, 1) with the rendered silhouette map.
        """
        bg_color_ = torch.tensor(bg_color)[None, None, :]  # (1, 1, 3)
        bg_color_ = bg_color_.to(self.device)

        # Globally sort gaussians according to their depth value
        z_vals = self.compute_depth_values(camera)
        idxs = self.get_idxs_to_filter_and_sort(z_vals)

        pre_act_quats = self.gaussians.pre_act_quats[idxs]
        pre_act_scales = self.gaussians.pre_act_scales[idxs]
        pre_act_opacities = self.gaussians.pre_act_opacities[idxs]
        z_vals = z_vals[idxs]
        means_3D = self.gaussians.means[idxs]

        if getattr(self.gaussians, "spherical_harmonics", None) is None:
            # For questions 1.1, 1.2 and 1.3.2, use the below line of code for colors.
            colors = self.gaussians.colors[idxs]
        else:
            # [Q 1.3.1] For question 1.3.1, uncomment the below three lines to calculate the
            # colors instead of using self.gaussians.colors[idxs]. You may also comment
            # out the above line of code since it will be overwritten anyway.
            spherical_harmonics = self.gaussians.spherical_harmonics[idxs]
            gaussian_dirs = self.calculate_gaussian_directions(means_3D, camera)
            colors = colors_from_spherical_harmonics(spherical_harmonics, gaussian_dirs)

        # Apply activations
        quats, scales, opacities = self.gaussians.apply_activations(
            pre_act_quats, pre_act_scales, pre_act_opacities
        )

        if per_splat == -1:
            num_mini_batches = 1
        elif per_splat > 0:
            num_mini_batches = math.ceil(len(means_3D) / per_splat)
        else:
            raise ValueError("Invalid setting of per_splat")

        # In this case we can directly splat all gaussians onto the image
        if num_mini_batches == 1:

            # Get image, depth and mask via splatting
            image, depth, mask, _ = self.splat(
                camera, means_3D, z_vals, quats, scales, colors, opacities, img_size
            )

        # In this case we splat per_splat number of gaussians per iteration. This makes
        # the implementation more memory efficient but at the same time makes it slower.
        else:

            W, H = img_size
            D = means_3D.device
            start_transmittance = torch.ones((1, H, W), dtype=torch.float32).to(D)
            image = torch.zeros((H, W, 3), dtype=torch.float32).to(D)
            depth = torch.zeros((H, W, 1), dtype=torch.float32).to(D)
            mask = torch.zeros((H, W, 1), dtype=torch.float32).to(D)

            for b_idx in range(num_mini_batches):

                quats_ = quats[b_idx * per_splat : (b_idx + 1) * per_splat]
                scales_ = scales[b_idx * per_splat : (b_idx + 1) * per_splat]
                z_vals_ = z_vals[b_idx * per_splat : (b_idx + 1) * per_splat]
                colors_ = colors[b_idx * per_splat : (b_idx + 1) * per_splat]
                means_3D_ = means_3D[b_idx * per_splat : (b_idx + 1) * per_splat]
                opacities_ = opacities[b_idx * per_splat : (b_idx + 1) * per_splat]

                # Get image, depth and mask via splatting
                image_, depth_, mask_, start_transmittance = self.splat(
                    camera,
                    means_3D_,
                    z_vals_,
                    quats_,
                    scales_,
                    colors_,
                    opacities_,
                    img_size,
                    start_transmittance,
                )

                image = image + image_
                depth = depth + depth_
                mask = mask + mask_

        image = mask * image + (1.0 - mask) * bg_color_

        return image, depth, mask

    def calculate_gaussian_directions(
        self,
        means_3D: torch.Tensor,
        camera: PerspectiveCameras,
    ) -> torch.Tensor:
        """
        [Q 1.3.1] Calculates the world frame direction vectors that point from the
        camera's origin to each 3D Gaussian.

        Args:
            means_3D        :   A torch.Tensor of shape (N, 3) with the means
                                of the 3D Gaussians.
            camera          :   A pytorch3d PerspectiveCameras object.

        Returns:
            gaussian_dirs   :   A torch.Tensor of shape (N, 3) representing the direction vector
                                that points from the camera's origin to each 3D Gaussian.
        """
        ### YOUR CODE HERE ###
        # HINT: Think about how to get the camera origin in the world frame.
        # HINT: Do not forget to normalize the computed directions.
        origins = camera.get_camera_center()
        gaussian_dirs = F.normalize(means_3D - origins, p=2, dim=-1)  # (N, 3)
        return gaussian_dirs
