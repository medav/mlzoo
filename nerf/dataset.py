import torch
import numpy as np
import json
import imageio.v2 as imageio
from dataclasses import dataclass

def get_rays_np(h, w, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    indexing='xy')

    dirs = np.stack([
        (i - w / 2) / focal,
        -(j - h / 2) / focal,
        -np.ones_like(i)
    ], -1)

    rays_d = dirs @ c2w[:3, :3].T
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


class BlenderDataset(torch.utils.data.Dataset):
    @dataclass
    class Frame:
        file : str
        rot : float
        pose : np.ndarray

    def __init__(self, dirpath, split='train', white_bg=False):
        super().__init__()
        self.dirpath = dirpath
        self.split = split

        jd = json.load(open(dirpath + f'/transforms_{split}.json'))


        self.frame_data = [
            BlenderDataset.Frame(
                file=f'{dirpath}/{f["file_path"]}.png',
                rot=float(f["rotation"]),
                pose=np.array(f["transform_matrix"]))
            for f in jd['frames']
        ]

        imgs = np.array(
            [imageio.imread(fd.file) for fd in self.frame_data],
            dtype=np.float32) / 255

        # TODO: Not entirely sure what this does -- figure it out?
        if white_bg:
            imgs = imgs[..., :3] * imgs[..., -1:] + (1 - imgs[..., -1:])
        else:
            imgs = imgs[..., :3]

        poses = np.array([fd.pose for fd in self.frame_data], dtype=np.float32)

        self.imgs = torch.from_numpy(imgs)
        self.poses = torch.from_numpy(poses)

        [_, h, w, _] = imgs.shape
        self.h = h
        self.w = w

        self.camera_angle_x = jd['camera_angle_x']
        self.focal = 0.5 * w / np.tan(0.5 * self.camera_angle_x)

        rays = np.stack(
            [get_rays_np(h, w, self.focal, p) for p in poses[:, :3, :4]],
            axis=0)

        rays_rgb = np.concatenate([rays, imgs[:, np.newaxis, ...]], axis=1) \
            .transpose([0, 2, 3, 1, 4]) \
            .reshape([-1, 3, 3])

        np.random.shuffle(rays_rgb)
        self.rays_rgb = torch.from_numpy(rays_rgb)


    def __len__(self): return self.rays_rgb.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, int):
            batch = self.rays_rgb[idx:idx + 1].permute((1, 0, 2))
        else:
            batch = self.rays_rgb[idx].permute((1, 0, 2))
        return batch[:2], batch[2]



if __name__ == '__main__':
    ds = BlenderDataset('/research/data/nerf/lego', split='train')

    rays, target = ds[0]
    print(rays.shape, target.shape)


