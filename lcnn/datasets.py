import glob
import random

import numpy as np
import numpy.linalg as LA
from skimage import io
from tensorflow.keras.utils import Sequence

from lcnn.config import M


class WireframeDataset(Sequence):
    def __init__(self, rootdir, split, batch_size=1):
        self.rootdir = rootdir
        print(f"root dir:{rootdir}/{split}/")
        filelist = glob.glob(f"{rootdir}/{split}/*_label.npz")
        filelist.sort()

        print(f"n{split}:", len(filelist))
        self.split = split
        self.filelist = filelist
        self.batch_size = batch_size

    def __len__(self):
        return len(self.filelist) // self.batch_size

    def __getitem__(self, idx):
        # take one batch of files
        batch_filelist = self.filelist[idx*self.batch_size: (idx+1) * self.batch_size]

        # init batch image, meta and target
        batch_image = [None] * self.batch_size
        batch_meta = [None] * self.batch_size
        batch_target = [None] * self.batch_size

        for bi, file_ in enumerate(batch_filelist):
            iname = file_[:-10].replace("_a0", "").replace("_a1", "") + ".png"
            print(f"iname: {iname}")
            # take upto three channels
            image = io.imread(iname).astype(float)[:, :, :3]
            if "a1" in file_:
                print("a1 fond in file list")
                image = image[:, ::-1, :]
            image = (image - M.image.mean) / M.image.stddev
            # npz["jmap"]: [J, H, W]    Junction heat map
            # npz["joff"]: [J, 2, H, W] Junction offset within each pixel
            # npz["lmap"]: [H, W]       Line heat map with anti-aliasing
            # npz["junc"]: [Na, 3]      Junction coordinates
            # npz["Lpos"]: [M, 2]       Positive lines represented with junction indices
            # npz["Lneg"]: [M, 2]       Negative lines represented with junction indices
            # npz["lpos"]: [Np, 2, 3]   Positive lines represented with junction coordinates
            # npz["lneg"]: [Nn, 2, 3]   Negative lines represented with junction coordinates
            #
            # For junc, lpos, and lneg that stores the junction coordinates, the last
            # dimension is (y, x, t), where t represents the type of that junction.
            with np.load(file_) as npz:
                target = {
                    name: npz[name]
                    for name in ["jmap", "joff", "lmap"]
                }
                lpos = np.random.permutation(npz["lpos"])[: M.n_stc_posl]
                lneg = np.random.permutation(npz["lneg"])[: M.n_stc_negl]
                npos, nneg = len(lpos), len(lneg)
                lpre = np.concatenate([lpos, lneg], 0)
                for i in range(len(lpre)):
                    if random.random() > 0.5:
                        lpre[i] = lpre[i, ::-1]
                ldir = lpre[:, 0, :2] - lpre[:, 1, :2]
                ldir /= np.clip(LA.norm(ldir, axis=1, keepdims=True), 1e-6, None)
                feat = [
                    lpre[:, :, :2].reshape(-1, 4) / 128 * M.use_cood,
                    ldir * M.use_slop,
                    lpre[:, :, 2],
                ]
                feat = np.concatenate(feat, 1)
                meta = {
                    "junc": npz["junc"][:, :2],
                    "jtyp": npz["junc"][:, 2],
                    "Lpos": self.adjacency_matrix(len(npz["junc"]), npz["Lpos"]),
                    "Lneg": self.adjacency_matrix(len(npz["junc"]), npz["Lneg"]),
                    "lpre": lpre[:, :, :2],
                    "lpre_label": np.concatenate([np.ones(npos), np.zeros(nneg)]),
                    "lpre_feat": feat,
                }
            batch_image[bi] = image
            batch_meta[bi] = meta
            batch_target[bi] = target

        return batch_image, batch_meta, batch_target

    def adjacency_matrix(self, n, link):
        mat = np.zeros([n + 1, n + 1], dtype='uint8')
        if len(link) > 0:
            mat[link[:, 0], link[:, 1]] = 1
            mat[link[:, 1], link[:, 0]] = 1
        return mat
    
    def on_epoch_end(self):
        random.shuffle(self.filelist)
    
    def inspect_batch(self, batch_data):
        '''Inspect batch of data'''
        batch_image, batch_meta, batch_target = batch_data
        for image, meta, target in zip(batch_image, batch_meta, batch_target):
            print('-' * 30)
            print("Image ...")
            print(f"shape: {image.shape}")
            print('Meta ...')
            for k, v in meta.items():
                print(f'{k}: {v.shape}')
            print('Target ...')
            for k, v in target.items():
                print(f'{k}: {v.shape}')
            print('-' * 30)

