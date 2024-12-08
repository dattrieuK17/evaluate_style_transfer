from skimage.metrics import structural_similarity as ssim
from PIL import Image
import os
from utils import *

content_dir = "D:/CS406-ImageProcessingAndApplication/Evaluate/images/content"
style_dir = "D:/CS406-ImageProcessingAndApplication/Evaluate/images/style"
result_dir = "D:/CS406-ImageProcessingAndApplication/Evaluate/images/result"

def style_Transfer():
    models = load_model()
    i = 1
    content_images = [f for f in os.listdir(content_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    style_images = [f for f in os.listdir(style_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    for model_name, model in models.items():
        for content_name in content_images:
            content_path = os.path.join(content_dir, content_name)
            for style_name in style_images:
                print(f"{i}) {content_name} + {style_name}")
                style_path = os.path.join(style_dir, style_name)

                content_image = Image.open(content_path).convert('RGB')
                style_image = Image.open(style_path).convert('RGB')
            
                style_tensor = model['preprocess'](style_image)
                content_tensor = model['preprocess'](content_image)
                output_tensor = model['model'](content_tensor, style_tensor)
                output_image = tensor_to_pil(output_tensor[0])

                result_path = os.path.join(result_dir, model_name, f"{content_name.split('.')[0]}_{style_name.split('.')[0]}.png")
                os.makedirs(os.path.dirname(result_path), exist_ok=True)

                output_image.save(result_path, format="PNG")
                print("Ket qua da duoc luu")
                i += 1

style_Transfer()

def calculate_ssim(image1_path, image2_path):
    # Đọc ảnh từ đường dẫn
    img1 =Image.open(image1_path).convert("RGB")
    img2 = Image.open(image2_path).convert("RGB")
    
    # Kiểm tra xem ảnh có được đọc thành công không
    if img1 is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh tại: {image1_path}")
    if img2 is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh tại: {image2_path}")
    
    # Kiểm tra kích thước ảnh
    if img1.shape != img2.shape:
        raise ValueError("Hai ảnh phải có cùng kích thước để tính SSIM.")
    
    # Tính SSIM
    score, _ = ssim(img1, img2, multichannel=True)
    return score

def get_image_name():
    folder_names = []
    all_image_names = []
    for folder_name in os.listdir(result_dir):
        folder_names.append(folder_name)
        image_names = [image_name for image_name in os.listdir(os.path.join(result_dir, folder_name))]
        all_image_names.append(image_names)
    return folder_names, all_image_names

def evaluate():
    folder_names, all_image_names = get_image_name()
    ssim_score = []
    for folder_name, image_names in zip(folder_names, all_image_names):
        folder_path = os.path.join(result_dir, folder_name)
        ssim = []
        for image_name in image_names:
            content_name = image_name.split('_')[0]
            content_path = os.path.join(content_dir, content_name)
            result_path = os.path.join(folder_path, image_name)
            ssim.append(calculate_ssim(content_path, result_path))
        ssim_score.append(ssim)
    return ssim


    

