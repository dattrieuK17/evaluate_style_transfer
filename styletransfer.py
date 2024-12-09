from PIL import Image
from utils import *
import argparse


# content_dir = "D:/CS406-ImageProcessingAndApplication/Evaluate/images/content"
# style_dir = "D:/CS406-ImageProcessingAndApplication/Evaluate/images/style"
# result_dir = "D:/CS406-ImageProcessingAndApplication/Evaluate/images/result"

def style_Transfer(content_dir, style_dir, result_dir):
    models = load_model()
    
    content_images = [f for f in os.listdir(content_dir)]
    
    for model_name, model in models.items():
        if model_name == "AdaAttnN": continue
        result_model_name_path = os.path.join(result_dir, model_name)
        os.makedirs(result_model_name_path, exist_ok=True) # dir: AdaAttN, AdaIN
        i = 1
        for style_category in os.listdir(style_dir): # 10 category trong style
            result_category_path = os.path.join(result_model_name_path, style_category) # AdaAttN -> result_category_path
            os.makedirs(result_category_path, exist_ok=True)

            style_category_path = os.path.join(style_dir, style_category)

            for style_name in os.listdir(style_category_path):
                style_path = os.path.join(style_category_path, style_name)

                for content_name in content_images:
                    content_path = os.path.join(content_dir, content_name)
                    print(f"{i}) {content_name} + {style_name}")
                    

                    content_image = Image.open(content_path).convert('RGB')
                    style_image = Image.open(style_path).convert('RGB')
                
                    style_tensor = model['preprocess'](style_image)
                    content_tensor = model['preprocess'](content_image)
                    output_tensor = model['model'](content_tensor, style_tensor)
                    output_image = tensor_to_pil(output_tensor[0])

                    output_path = os.path.join(result_category_path, f"{content_name.split('.')[0]}+{style_name.split('.')[0]}.png")
                    

                    output_image.save(output_path, format="PNG")
                    print("Ket qua da duoc luu")
                    i += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir", type=str, required=True, help="Folder containing content images.")
    parser.add_argument("--style_dir", type=str, required=True, help="Folder containing style images.")
    parser.add_argument("--result_dir", type=str, required=True, help="Folder containing result images.")


    args = parser.parse_args()

    style_Transfer(args.content_dir, args.style_dir, args.result_dir)
