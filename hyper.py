import os
IMG_SIZE = 512
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONTENT_IMAGE_FOLDER = os.path.join(BASE_DIR, 'static', 'images', 'content')
STYLE_IMAGE_FOLDER = os.path.join(BASE_DIR, 'static', 'images', 'style')

AdaAttN = 'AdaAttN'
AdaAttN_encoder = 'D:/CS406-ImageProcessingAndApplication/Parrots/TransferModel/AdaAttN/vgg_normalised.pth'
AdaAttN_decoder = 'D:/CS406-ImageProcessingAndApplication/Parrots/TransferModel/AdaAttN/latest_net_decoder.pth'
AdaAttN_adattn_3 = 'D:/CS406-ImageProcessingAndApplication/Parrots/TransferModel/AdaAttN/latest_net_adaattn_3.pth'
AdaAttN_adattn_4_5 = 'D:/CS406-ImageProcessingAndApplication/Parrots/TransferModel/AdaAttN/latest_net_transformer.pth'

AdaIN = 'AdaIN'
AdaIN_encoder = 'D:/CS406-ImageProcessingAndApplication/Parrots/TransferModel/AdaIN/vgg_normalised.pth'
AdaIN_decoder = 'D:/CS406-ImageProcessingAndApplication/Parrots/TransferModel/AdaIN/adain_decoder.pth'

TFStyleTransfer = 'TF-StyleTransfer'
TF_model = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
