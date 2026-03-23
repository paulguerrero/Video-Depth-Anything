# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import Compose
import cv2
from tqdm import tqdm
import numpy as np
import gc
from einops import rearrange

from .dinov2 import DINOv2
from .dpt_temporal import DPTHeadTemporal
from .util.transform import Resize, NormalizeImage, PrepareForNet

from .utils.util import compute_scale_and_shift, get_interpolate_frames

# infer settings, do not change
INFER_LEN = 32
OVERLAP = 10
KEYFRAMES = [0,12,24,25,26,27,28,29,30,31]
INTERP_LEN = 8

class VideoDepthAnything(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        num_frames=32,
        pe='ape',
        metric=False,
        ckpt=None,
    ):
        super(VideoDepthAnything, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23]
        }

        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)

        self.head = DPTHeadTemporal(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, num_frames=num_frames, pe=pe)
        self.metric = metric
        if ckpt is not None:
            self.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)

    def forward(self, x):
        B, T, C, H, W = x.shape
        patch_h, patch_w = H // 14, W // 14
        features = self.pretrained.get_intermediate_layers(x.flatten(0,1), self.intermediate_layer_idx[self.encoder], return_class_token=True)
        depth = self.head(features, patch_h, patch_w, T)[0]
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        return depth.squeeze(1).unflatten(0, (B, T)) # return shape [B, T, H, W]

    def infer_video_depth(self, frames, target_fps, input_size=518, device='cuda', fp32=False):
        frame_height, frame_width = frames[0].shape[:2]
        ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
        if ratio > 1.78:  # we recommend to process video with ratio smaller than 16:9 due to memory limitation
            input_size = int(input_size * 1.777 / ratio)
            input_size = round(input_size / 14) * 14

        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        frame_list = [frames[i] for i in range(frames.shape[0])]
        frame_step = INFER_LEN - OVERLAP
        org_video_len = len(frame_list)
        append_frame_len = (frame_step - (org_video_len % frame_step)) % frame_step + (INFER_LEN - frame_step)
        frame_list = frame_list + [frame_list[-1].copy()] * append_frame_len

        depth_list = []
        pre_input = None
        for frame_id in tqdm(range(0, org_video_len, frame_step)):
            cur_list = []
            for i in range(INFER_LEN):
                cur_list.append(torch.from_numpy(transform({'image': frame_list[frame_id+i].astype(np.float32) / 255.0})['image']).unsqueeze(0).unsqueeze(0))
            cur_input = torch.cat(cur_list, dim=1).to(device)
            if pre_input is not None:
                cur_input[:, :OVERLAP, ...] = pre_input[:, KEYFRAMES, ...]

            with torch.no_grad():
                with torch.autocast(device_type=torch.device(device).type, enabled=(not fp32)):
                    depth = self.forward(cur_input) # depth shape: [1, T, H, W]

            depth = depth.to(cur_input.dtype)
            depth = F.interpolate(depth.flatten(0,1).unsqueeze(1), size=(frame_height, frame_width), mode='bilinear', align_corners=True)
            depth_list += [depth[i][0].cpu().numpy() for i in range(depth.shape[0])]

            pre_input = cur_input

        del frame_list
        gc.collect()

        depth_list_aligned = []
        ref_align = []
        align_len = OVERLAP - INTERP_LEN
        kf_align_list = KEYFRAMES[:align_len]

        for frame_id in range(0, len(depth_list), INFER_LEN):
            if len(depth_list_aligned) == 0:
                depth_list_aligned += depth_list[:INFER_LEN]
                for kf_id in kf_align_list:
                    ref_align.append(depth_list[frame_id+kf_id])
            else:
                curr_align = []
                for i in range(len(kf_align_list)):
                    curr_align.append(depth_list[frame_id+i])

                if self.metric:
                    scale, shift = 1.0, 0.0
                else:
                    scale, shift = compute_scale_and_shift(np.concatenate(curr_align),
                                                           np.concatenate(ref_align),
                                                           np.concatenate(np.ones_like(ref_align)==1))

                pre_depth_list = depth_list_aligned[-INTERP_LEN:]
                post_depth_list = depth_list[frame_id+align_len:frame_id+OVERLAP]
                for i in range(len(post_depth_list)):
                    post_depth_list[i] = post_depth_list[i] * scale + shift
                    post_depth_list[i][post_depth_list[i]<0] = 0
                depth_list_aligned[-INTERP_LEN:] = get_interpolate_frames(pre_depth_list, post_depth_list)

                for i in range(OVERLAP, INFER_LEN):
                    new_depth = depth_list[frame_id+i] * scale + shift
                    new_depth[new_depth<0] = 0
                    depth_list_aligned.append(new_depth)

                ref_align = ref_align[:1]
                for kf_id in kf_align_list[1:]:
                    new_depth = depth_list[frame_id+kf_id] * scale + shift
                    new_depth[new_depth<0] = 0
                    ref_align.append(new_depth)

        depth_list = depth_list_aligned

        return np.stack(depth_list[:org_video_len], axis=0), target_fps

    @staticmethod
    def compute_scale_and_shift_torch(prediction, target, mask, scale_only=False):
        # Ensure scale_only is a boolean scalar, not a tensor
        scale_only = bool(scale_only)

        if scale_only:
            # For scale only computation
            a_00 = torch.sum(mask * prediction * prediction, dim=1)
            b_0 = torch.sum(mask * prediction * target, dim=1)
            scale = b_0 / (a_00 + 1e-6)
            return scale, torch.zeros_like(scale)
        else:
            # For both scale and shift computation
            a_00 = torch.sum(mask * prediction * prediction, dim=1)
            a_01 = torch.sum(mask * prediction, dim=1)
            a_11 = torch.sum(mask, dim=1)

            b_0 = torch.sum(mask * prediction * target, dim=1)
            b_1 = torch.sum(mask * target, dim=1)

            det = a_00 * a_11 - a_01 * a_01

            scale = torch.ones_like(det)
            shift = torch.zeros_like(det)

            valid_det = det != 0
            scale[valid_det] = (a_11[valid_det] * b_0[valid_det] - a_01[valid_det] * b_1[valid_det]) / det[valid_det]
            shift[valid_det] = (-a_01[valid_det] * b_0[valid_det] + a_00[valid_det] * b_1[valid_det]) / det[valid_det]

            return scale, shift

    def get_interpolate_frames_torch(self, frame_list_pre, frame_list_post):
        n = len(frame_list_pre)
        device = frame_list_pre[0].device
        weights = torch.linspace(0, 1, n, device=device)
        return [(1 - w) * pre + w * post for pre, post, w in zip(frame_list_pre, frame_list_post, weights)]

    def infer_multi_videos_depth_tensor(self, frames, target_fps, input_size=518, device="cuda"):
        # frames should be of shape [B, F, C, H, W]
        frame_height, frame_width = frames.shape[-2], frames.shape[-1]
        batch_size, num_frames = frames.shape[0], frames.shape[1]
        ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
        if ratio > 1.78:  # we recommend to process video with ratio smaller than 16:9 due to memory limitation
            input_size = int(input_size * 1.777 / ratio)
            input_size = round(input_size / 14) * 14

        # Calculate new dimensions while preserving aspect ratio
        scale_height = input_size / frame_height
        scale_width = input_size / frame_width

        if scale_width > scale_height:
            new_height = int(frame_height * scale_width)
            new_width = input_size
        else:
            new_height = input_size
            new_width = int(frame_width * scale_height)

        # Ensure dimensions are multiples of 14
        new_height = (new_height // 14) * 14
        new_width = (new_width // 14) * 14

        frames = rearrange(frames, "b f c h w -> (b f) c h w") / 255.0
        frames = F.interpolate(frames, size=(new_height, new_width), mode="bicubic", align_corners=True)
        frames = (
            frames - torch.tensor([0.485, 0.456, 0.406], dtype=frames.dtype, device=frames.device).view(1, 3, 1, 1)
        ) / torch.tensor([0.229, 0.224, 0.225], dtype=frames.dtype, device=frames.device).view(1, 3, 1, 1)
        frames = rearrange(frames, "(b f) c h w -> b f c h w", b=batch_size, f=num_frames)

        frame_step = INFER_LEN - OVERLAP
        org_video_len = frames.shape[1]
        append_frame_len = (frame_step - (org_video_len % frame_step)) % frame_step + (INFER_LEN - frame_step)
        frames = torch.cat([frames, frames[:, -1:, :, :, :].expand(-1, append_frame_len, -1, -1, -1)], dim=1)

        # Process frames in chunks
        depth_list = []
        depth_aligned = []
        pre_input = None
        for frame_id in range(0, org_video_len, frame_step):
            cur_input = frames[:, frame_id : frame_id + INFER_LEN, ...].to(device)
            if pre_input is not None:
                cur_input[:, :OVERLAP, ...] = pre_input[:, KEYFRAMES, ...]

            with torch.no_grad():
                with torch.autocast(device_type=torch.device(device).type, enabled=True):
                    depth = self.forward(cur_input)  # depth shape: [B, T, H, W]
                depth = depth.to(cur_input.dtype)
                depth = F.interpolate(
                    depth.flatten(0, 1).unsqueeze(1),
                    size=(frame_height, frame_width),
                    mode="bilinear",
                    align_corners=True,
                )
                depth = depth.squeeze(1).reshape(batch_size, -1, frame_height, frame_width)
                depth_list.append(depth)  # Store as torch tensor

            pre_input = cur_input

        del frames
        gc.collect()

        # Depth alignment using torch tensors
        depth_aligned = []
        ref_align = None
        align_len = OVERLAP - INTERP_LEN
        kf_align_list = KEYFRAMES[:align_len]

        for chunk_idx, depth_chunk in enumerate(depth_list):
            if chunk_idx == 0:
                depth_aligned.append(depth_chunk)  # First chunk remains unchanged
                ref_align = depth_chunk[:, kf_align_list, ...]
            else:
                curr_align = depth_chunk[:, : len(kf_align_list), ...]

                # Compute scale and shift using torch tensors
                mask = torch.ones_like(curr_align, device=device)
                scale, shift = self.compute_scale_and_shift_torch(
                    curr_align.reshape(batch_size, -1), ref_align.reshape(batch_size, -1), mask.reshape(batch_size, -1)
                )

                # Apply scale and shift
                depth_chunk = depth_chunk * scale.view(-1, 1, 1, 1) + shift.view(-1, 1, 1, 1)
                depth_chunk = torch.clamp(depth_chunk, min=0)

                # Interpolate overlapping frames
                pre_depth = depth_aligned[-1][:, -INTERP_LEN:, ...]
                post_depth = depth_chunk[:, align_len:OVERLAP, ...]
                interpolated = self.get_interpolate_frames_torch(
                    [pre_depth[:, i] for i in range(INTERP_LEN)], [post_depth[:, i] for i in range(INTERP_LEN)]
                )

                # Update last INTERP_LEN frames of previous chunk
                for i, interp in enumerate(interpolated):
                    depth_aligned[-1][:, -INTERP_LEN + i] = interp

                # Append non-overlapping frames
                depth_aligned.append(depth_chunk[:, OVERLAP:, ...])

                # Update reference alignment
                ref_align = torch.cat([ref_align[:, :1], depth_chunk[:, kf_align_list[1:], ...]], dim=1)

        # Concatenate all aligned depths
        depth_all = torch.cat([chunk for chunk in depth_aligned], dim=1)
        depth_all = depth_all[:, :org_video_len, ...]
        return depth_all, target_fps
    