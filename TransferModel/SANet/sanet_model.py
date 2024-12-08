import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

class SANet(nn.Module):
    
    def __init__(self, in_planes):
        super(SANet, self).__init__()
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim = -1)
        self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))
        
    def forward(self, content, style):
        F = self.f(mean_variance_norm(content))
        G = self.g(mean_variance_norm(style))
        H = self.h(style)
        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        b, c, h, w = G.size()
        G = G.view(b, -1, w * h)
        S = torch.bmm(F, G)
        S = self.sm(S)
        b, c, h, w = H.size()
        H = H.view(b, -1, w * h)
        O = torch.bmm(H, S.permute(0, 2, 1))
        b, c, h, w = content.size()
        O = O.view(b, c, h, w)
        O = self.out_conv(O)
        O += content
        return O

class Transform(nn.Module):
    def __init__(self, in_planes):
        super(Transform, self).__init__()
        self.sanet4_1 = SANet(in_planes = in_planes)
        self.sanet5_1 = SANet(in_planes = in_planes)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))
    def forward(self, content4_1, style4_1, content5_1, style5_1):
        return self.merge_conv(self.merge_conv_pad(self.sanet4_1(content4_1, style4_1) + self.upsample5_1(self.sanet5_1(content5_1, style5_1))))

def test_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def sanet_preprocess(img):
    # Thay đổi kích thước ảnh với padding để giữ nguyên tỷ lệ
    h, w = img.size
    max_size = 512  # Kích thước cố định
    
    # Tính toán tỷ lệ scale
    scale = min(max_size/h, max_size/w)
    
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Resize ảnh
    img = img.resize((new_w, new_h), Image.BILINEAR)
    
    return img
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = Transform(in_planes = 512)

def evaluate(content_dir, style_dir, result_dir, decoder_path, transform_path, vgg_path, device, steps=1):
    # Giữ nguyên các bước khởi tạo model
    decoder.eval()
    transform.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load(decoder_path))
    transform.load_state_dict(torch.load(transform_path))
    vgg.load_state_dict(torch.load(vgg_path))

    # Giữ nguyên các bước tách layer
    norm = nn.Sequential(*list(vgg.children())[:1])
    enc_1 = nn.Sequential(*list(vgg.children())[:4])
    enc_2 = nn.Sequential(*list(vgg.children())[4:11])
    enc_3 = nn.Sequential(*list(vgg.children())[11:18])
    enc_4 = nn.Sequential(*list(vgg.children())[18:31])
    enc_5 = nn.Sequential(*list(vgg.children())[31:44])

    # Di chuyển model
    norm.to(device)
    enc_1.to(device)
    enc_2.to(device)
    enc_3.to(device)
    enc_4.to(device)
    enc_5.to(device)
    transform.to(device)
    decoder.to(device)

    # Sử dụng transform gốc
    content_tf = test_transform()
    style_tf = test_transform()

    # Tạo thư mục kết quả
    os.makedirs(result_dir, exist_ok=True)

    # Lấy danh sách ảnh
    content_images = [f for f in os.listdir(content_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    style_images = [f for f in os.listdir(style_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    for content_name in content_images:
        content_path = os.path.join(content_dir, content_name)
        
        # Mở ảnh, convert sang RGB để đảm bảo tính nhất quán
        content_image = Image.open(content_path).convert('RGB')
        
        # Chuyển đổi ảnh
        content_img = content_tf(content_image)
        content_img = content_img.to(device).unsqueeze(0)

        for style_name in style_images:
            style_path = os.path.join(style_dir, style_name)
            
            # Mở ảnh, convert sang RGB
            style_image = Image.open(style_path).convert('RGB')
            
            # Chuyển đổi ảnh
            style_img = style_tf(style_image)
            style_img = style_img.to(device).unsqueeze(0)

            with torch.no_grad():
                # Bắt đầu quá trình style transfer
                content = content_img.clone()
                for x in range(steps):
                    print(f'Iteration {x + 1}/{steps} for content {content_name} and style {style_name}')

                    Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))
                    Content5_1 = enc_5(Content4_1)

                    Style4_1 = enc_4(enc_3(enc_2(enc_1(style_img))))
                    Style5_1 = enc_5(Style4_1)

                    content = decoder(transform(Content4_1, Style4_1, Content5_1, Style5_1))
                    content.clamp_(0, 255)

                # Lưu ảnh kết quả
                content = content.cpu()
                output_name = f"{splitext(content_name)[0]}_stylized_{splitext(style_name)[0]}.png"
                output_path = os.path.join(result_dir, output_name)
                save_image(content, output_path)
                print(f"Saved: {output_path}")




content_path = "D:/CS406-ImageProcessingAndApplication/Evaluate/images/content"
style_path = "D:/CS406-ImageProcessingAndApplication/Evaluate/images/style"
decoder_path = "D:/CS406-ImageProcessingAndApplication/Evaluate/TransferModel/SANet/decoder_iter_500000.pth"
transform_path = "D:/CS406-ImageProcessingAndApplication/Evaluate/TransferModel/SANet/transformer_iter_500000.pth"
vgg_path = "D:/CS406-ImageProcessingAndApplication/Evaluate/TransferModel/SANet/vgg_normalised.pth"

evaluate(content_path, style_path,"D:/CS406-ImageProcessingAndApplication/Evaluate/images/result", decoder_path, transform_path, vgg_path, device)