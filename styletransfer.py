from PIL import Image
from utils import *
import argparse


# content_dir = "D:/CS406-ImageProcessingAndApplication/Evaluate/images/content"
# style_dir = "D:/CS406-ImageProcessingAndApplication/Evaluate/images/style"
# result_dir = "D:/CS406-ImageProcessingAndApplication/Evaluate/images/result"

def style_Transfer(content_dir, style_dir, result_dir):
    models = load_model()
    
    content_images = [f for f in os.listdir(content_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    style_images = [f for f in os.listdir(style_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    for model_name, model in models.items():
        for content_name in content_images:
            content_path = os.path.join(content_dir, content_name)
            for style_name in style_images:
                print(f"{content_name} + {style_name}")
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
                print("Ket qua da duoc luu o", result_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Style Transfer Script")
    parser.add_argument("--content_dir", type=str, required=True, help="Folder containing content images.")
    parser.add_argument("--style_dir", type=str, required=True, help="Folder containing style images.")
    parser.add_argument("--result_dir", type=str, required=True, help="Folder containing result images.")


    args = parser.parse_args()

    style_Transfer(args.content_dir, args.style_dir, args.method, args.size)
