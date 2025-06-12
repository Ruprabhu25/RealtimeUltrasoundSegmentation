import os
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from tqdm import tqdm
import nibabel as nib

# 1. Dataset
class JpegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size):
        self.image_paths = sorted(
            glob.glob(os.path.join(image_dir, '*.png'))
        )
        self.mask_paths = sorted(
            glob.glob(os.path.join(mask_dir, '*.nii'))
        )
        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError(f"Image/Mask count mismatch: {len(self.image_paths)} vs {len(self.mask_paths)}")
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        msk = np.array(nib.load(self.mask_paths[idx]).get_fdata())
        if img is None or msk is None:
            raise IOError(f"Failed to load idx={idx}")
        img = cv2.resize(img, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
        img = torch.from_numpy(img.astype(np.float32)/255.0).unsqueeze(0)
        msk = torch.from_numpy((msk.astype(np.float32) > 0.5).astype(np.float32)).unsqueeze(0)
        return img, msk

# 2. U-Net with separate feature & output heads
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, feat_list=[64,128,256], out_ch=1):
        super().__init__()
        # Encoder
        self.downs, self.pools = nn.ModuleList(), nn.ModuleList()
        prev_ch = in_ch
        for f in feat_list:
            self.downs.append(DoubleConv(prev_ch, f))
            self.pools.append(nn.MaxPool2d(2))
            prev_ch = f
        # Bottleneck
        self.bottleneck = DoubleConv(prev_ch, feat_list[-1]*2)
        # Decoder
        reversed_feats = list(reversed(feat_list))  # [256,128,64]
        in_decoder = [feat_list[-1]*2] + reversed_feats[:-1]  # [512,256,128]
        self.ups, self.upconvs = nn.ModuleList(), nn.ModuleList()
        for in_c, out_c in zip(in_decoder, reversed_feats):
            self.ups.append(nn.ConvTranspose2d(in_c, out_c, 2, stride=2))
            self.upconvs.append(DoubleConv(out_c*2, out_c))
        # Feature extractor (raw feature map) and mask head both use feat_list[0]
        base_ch = feat_list[0]  # = 64
        self.feature_conv = nn.Conv2d(base_ch, base_ch, 1)
        self.output_conv  = nn.Conv2d(base_ch, out_ch, 1)

    def forward_features(self, x):
        skip = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skip.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for up, conv in zip(self.ups, self.upconvs):
            x = up(x)
            s = skip.pop()
            if x.shape != s.shape:
                s = TF.center_crop(s, x.shape[2:])
            x = conv(torch.cat([s, x], dim=1))
        # 'x' now has 'base_ch' channels (feat_list[0])
        return self.feature_conv(x)

    def forward(self, x):
        feat_map = self.forward_features(x)
        return torch.sigmoid(self.output_conv(feat_map))
        self.output_conv  = nn.Conv2d(reversed_feats[-1], out_ch, 1)
        self.output_conv  = nn.Conv2d(reversed_feats[0], out_ch, 1)

    def forward_features(self, x):
        skip = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skip.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for up, conv in zip(self.ups, self.upconvs):
            x = up(x)
            s = skip.pop()
            if x.shape != s.shape:
                s = TF.center_crop(s, x.shape[2:])
            x = conv(torch.cat([s, x], dim=1))
        return self.feature_conv(x)  # raw feature map

    def forward(self, x):
        feat_map = self.forward_features(x)
        return torch.sigmoid(self.output_conv(feat_map))

# 3. Multi-Head wrapper
class MultiHeadUNet(nn.Module):
    def __init__(self, heads=3, feat_dim=64, out_ch=1):
        super().__init__()
        self.unet = UNet(in_ch=1, feat_list=[feat_dim, feat_dim*2, feat_dim*4], out_ch=out_ch)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(feat_dim, out_ch, 1), nn.Sigmoid())
            for _ in range(heads)
        ])
        self.fuse  = nn.Sequential(nn.Conv2d(heads, out_ch, 1), nn.Sigmoid())

    def forward(self, x):
        hmap = self.unet.forward_features(x)   # [B, feat_dim, H, W]
        outs = [h(hmap) for h in self.heads]    # list of [B,1,H,W]
        stack = torch.cat(outs, dim=1)          # [B, heads, H, W]
        return outs, self.fuse(stack)

# 4. Contrastive dissimilarity
def local_contrastive(heads):
    losses = []
    for i in range(len(heads)):
        for j in range(i+1, len(heads)):
            hi = heads[i].flatten(1)
            hj = heads[j].flatten(1)
            losses.append(-F.cosine_similarity(hi, hj, dim=1).mean())
    return torch.stack(losses).mean()

# 5. Training & validation epochs
def train_epoch(model, loader, opt, mi_w, alpha, device, multi=False, clip_grad=None):
    model.train()
    total_loss = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        opt.zero_grad()
        if multi:
            heads, fused = model(imgs)
            bce_fuse = F.binary_cross_entropy(fused, masks)
            bce_heads= sum(F.binary_cross_entropy(h, masks) for h in heads) / len(heads)
            cont_loss= local_contrastive(heads)
            loss = bce_fuse + alpha * bce_heads + mi_w * cont_loss
        else:
            pred = model(imgs)
            loss = F.binary_cross_entropy(pred, masks)
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def val_epoch(model, loader, device, multi=False):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            if multi:
                _, fused = model(imgs)
                total_loss += F.binary_cross_entropy(fused, masks).item()
            else:
                pred = model(imgs)
                total_loss += F.binary_cross_entropy(pred, masks).item()
    return total_loss / len(loader)

# Visualize outputs

def visualize_outputs(net1, net2, dataset, device, save_path, num_samples=8):
    net1.eval(); net2.eval()

    rows = num_samples
    fig, axs = plt.subplots(rows, 4, figsize=(16, 4 * rows))
    
    for i in range(num_samples):
        img, msk = dataset[i]
        inp = img.unsqueeze(0).to(device)
        with torch.no_grad():
            o1 = net1(inp)[0, 0].cpu().numpy()
            _, o2 = net2(inp); o2 = o2[0, 0].cpu().numpy()
        im, mt = img[0].cpu().numpy(), msk[0].cpu().numpy()
        axs[i, 0].imshow(im, cmap='gray'); axs[i, 0].set_title('Input'); axs[i, 0].axis('off')
        axs[i, 1].imshow(mt, cmap='gray'); axs[i, 1].set_title('GT'); axs[i, 1].axis('off')
        axs[i, 2].imshow(o1, cmap='gray'); axs[i, 2].set_title('UNet'); axs[i, 2].axis('off')
        axs[i, 3].imshow(o2, cmap='gray'); axs[i, 3].set_title('MHU'); axs[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# 6. Full training driver

def train_and_plot(img_dir, mask_dir, epochs=200, batch_size=4,
                   mi_max=0.1, alpha=1.0, image_size=(128, 128), clip_grad=1.0):
    ds = JpegDataset(img_dir, mask_dir, image_size)
    n_val = int(0.2 * len(ds))
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net1 = UNet(in_ch=1, feat_list=[64,128,256], out_ch=1).to(device)
    net2 = MultiHeadUNet(heads=3, feat_dim=64, out_ch=1).to(device)
    opt1 = torch.optim.Adam(net1.parameters(), lr=1e-3)
    opt2 = torch.optim.Adam(net2.parameters(), lr=1e-3)
    sched1 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt1, 'min', patience=5)
    sched2 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2, 'min', patience=5)

    save_path_unet = 'outputs/best_unet.pth'
    save_path_mhu  = 'outputs/best_mhu.pth'

    best_v1 = float('inf')
    best_v2 = float('inf')

    train_losses, val_losses = [], []
    for ep in range(1, epochs+1):
        mi_w = mi_max * min(ep / 10.0, 1.0)
        t1 = train_epoch(net1, train_loader, opt1, mi_w=0, alpha=0, device=device, multi=False, clip_grad=clip_grad)
        t2 = train_epoch(net2, train_loader, opt2, mi_w=mi_w, alpha=alpha, device=device, multi=True, clip_grad=clip_grad)
        v1 = val_epoch(net1, val_loader, device=device, multi=False)
        v2 = val_epoch(net2, val_loader, device=device, multi=True)

        if v1 < best_v1:
            best_v1 = v1
            torch.save(net1.state_dict(), save_path_unet)

        if v2 < best_v2:
            best_v2 = v2
            torch.save(net2.state_dict(), save_path_mhu)

        sched1.step(v1)
        sched2.step(v2)
        train_losses.append((t1, t2))
        val_losses.append((v1, v2))
        print(f"Epoch {ep:02d} — UNet val: {v1:.4f}, Multi-Head val: {v2:.4f} (mi_w={mi_w:.3f})")


    f = open(f'C:/Users/Mittal/Desktop/us_seg/outputs/output.txt', "a")
    print("Min Loss: ", min(train_losses), file=f)  
    print('Min Val Loss: ', min(val_losses), file=f)  
    f.close()

    # Plot losses
    epochs_arr = np.arange(1, epochs+1)
    t1s, t2s = zip(*train_losses)
    v1s, v2s = zip(*val_losses)
    plt.figure(figsize=(8,5))
    plt.plot(epochs_arr, t1s, '--', label='UNet Train')
    plt.plot(epochs_arr, v1s,  '-', label='UNet Val')
    plt.plot(epochs_arr, t2s, '--', label='MHU Train')
    plt.plot(epochs_arr, v2s,  '-', label='MHU Val')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'C:/Users/Mittal/Desktop/us_seg/outputs/losses.png')
    plt.close()

    visualize_outputs(net1, net2, ds, device, save_path='C:/Users/Mittal/Desktop/us_seg/outputs/sample_outputs.png', num_samples=8)


# Entry point
if __name__ == '__main__':
    IMG_FOLDER = 'C:/Users/Mittal/Desktop/us_seg/raw_images/'
    MSK_FOLDER = 'C:/Users/Mittal/Desktop/us_seg/segmentations/'
    train_and_plot(IMG_FOLDER, MSK_FOLDER)