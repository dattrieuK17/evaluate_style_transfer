from skimage.metrics import structural_similarity as ssim
from PIL import Image
import os
from utils import *

content_dir = "D:/CS406-ImageProcessingAndApplication/Evaluate/images/content"
style_dir = "D:/CS406-ImageProcessingAndApplication/Evaluate/images/style"
result_dir = "D:/CS406-ImageProcessingAndApplication/Evaluate/images/result"


def calculate_ssim(content_image, result_image):
    
    # Kiểm tra kích thước ảnh
    if content_image.shape != result_image.shape:
        raise ValueError("Hai ảnh phải có cùng kích thước để tính SSIM.")
    
    # Tính SSIM
    score, _ = ssim(content_image, result_image, multichannel=True)
    return score

def get_content_image(result_image_name):
    content_image_name = result_image_name.split('+')[0]
    for image_name in os.listdir(content_dir):
        if image_name.split(".")[0] == content_image_name: 
            content_image_name == image_name
            break
    return Image.open(os.path.join(content_dir, content_image_name))


def evaluate():
    ssim_avg = []
    for model_name in os.listdir(result_dir):
        ssim = 0
        i = 0
        model_name_path = os.path.join(result_dir, model_name)
        for category_style in os.listdir(model_name_path):
            category_style_path = os.path.join(model_name_path, category_style)
            for result_image_name in os.listdir(category_style_path):
                content_image = get_content_image(result_image_name)
                result_image = Image.open(os.path.join(category_style_path, result_image_name))
                ssim += calculate_ssim(content_image, result_image)
                i += 1
        ssim_avg.append({"model" : model_name, "ssim_avg" : float(ssim / i), "number_image" : i})

    return ssim


    

