import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x

class SODModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = ConvBlock(3, 64)      
        self.pool1 = nn.MaxPool2d(2)       

        self.enc2 = ConvBlock(64, 128)     
        self.pool2 = nn.MaxPool2d(2)     

        self.enc3 = ConvBlock(128, 256)    
        self.pool3 = nn.MaxPool2d(2)       

        self.bottleneck = ConvBlock(256, 512) 

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
    
        self.dec1 = ConvBlock(128, 64)
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)         
        p1 = self.pool1(e1)       

        e2 = self.enc2(p1)       
        p2 = self.pool2(e2)      

        e3 = self.enc3(p2)         
        p3 = self.pool3(e3)        

        b = self.bottleneck(p3)    

        u3 = self.up3(b)                           
        d3 = self.dec3(torch.cat([u3, e3], dim=1)) 

        u2 = self.up2(d3)                          
        d2 = self.dec2(torch.cat([u2, e2], dim=1)) 

        u1 = self.up1(d2)                         
        d1 = self.dec1(torch.cat([u1, e1], dim=1)) 

        out = self.out_conv(d1)                   
        out = torch.sigmoid(out)
        return out
def iou_score(pred, target, threshold=0.5, eps=1e-6):
    pred_bin = (pred > threshold).float()
    target_bin = (target > 0.5).float()

    pred_flat = pred_bin.view(pred_bin.size(0), -1)
    target_flat = target_bin.view(target_bin.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean()


def compute_prf1(pred, target, threshold=0.5, eps=1e-6):
    pred_bin = (pred > threshold).float()
    target_bin = (target > 0.5).float()

    pred_flat = pred_bin.view(pred_bin.size(0), -1)
    target_flat = target_bin.view(target_bin.size(0), -1)

    TP = (pred_flat * target_flat).sum(dim=1)
    FP = (pred_flat * (1 - target_flat)).sum(dim=1)
    FN = ((1 - pred_flat) * target_flat).sum(dim=1)

    precision = (TP + eps) / (TP + FP + eps)
    recall    = (TP + eps) / (TP + FN + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)

    return precision.mean(), recall.mean(), f1.mean()

if __name__ == "__main__":
    x = torch.randn(2, 3, 128, 128)
    model = SODModel()
    y = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
