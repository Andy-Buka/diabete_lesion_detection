from torchvision import transforms
from PIL import Image
import os
from collections import OrderedDict
import scipy.io as io
import numpy as np
import torch
import cv2

# path
result_path = "../../results_eval"
true_image_path = "../../FGADR-Seg-set_Release"

HE_path = os.path.join(true_image_path, 'HardExudate_Masks')
Hh_path = os.path.join(true_image_path, 'Hemohedge_Masks')
IRMA_path = os.path.join(true_image_path, 'IRMA_Masks')
M_path = os.path.join(true_image_path, 'Microaneurysms_Masks')
N_path = os.path.join(true_image_path, 'Neovascularization_Masks')
SE_path = os.path.join(true_image_path, 'SoftExudate_Masks')

# 将正确图像的路径打包
true_image_directory = OrderedDict([
                                    ('HardExudate_Masks', HE_path),
                                    ('Hemohedge_Masks', Hh_path),
                                    ('IRMA_Masks', IRMA_path),
                                    ('Microaneurysms_Masks', M_path),
                                    ('Neovascularization_Masks', N_path),
                                    ('SoftExudate_Masks', SE_path)
                                    ])

img_list = [os.path.splitext(f)[0] for f in os.listdir(result_path) if f.endswith('png')]

PIL_trans_tensor = transforms.ToTensor()
trans_resize = transforms.Resize((1280, 1280))

img_show_list = ["1811_1", "1833_1"]

# 对每一个测试图像
for idx, img_name in enumerate(img_list):

    # 拿到测试tensor
    img_result = cv2.imread(os.path.join(result_path, img_name + '.png'), 0)
    # 二值化
    ret, binary = cv2.threshold(img_result, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_result_tensor = PIL_trans_tensor(binary)
    if img_name in img_show_list:
        img = transforms.ToPILImage()(img_result_tensor)
        img.show(title=img_name)

    # 在每个正确图像的路径下，如果有，拿到tensor
    tot_image = torch.zeros([1, 1280, 1280])
    for name, root in true_image_directory.items():
        img_temp_path = os.path.join(root, img_name + '.png')
        if os.path.exists(img_temp_path):
            img_temp = Image.open(os.path.join(img_temp_path))
            img_temp_tensor = PIL_trans_tensor(img_temp)

            # 将四个子图像重合为a
            tot_image = tot_image + img_temp_tensor[0]
    # 图像值域在0-1范围
    threshold = 0
    tot_image = tot_image > threshold
    tot_image = tot_image.to(torch.float)
    if img_name in img_show_list:
        img_ori = transforms.ToPILImage()(tot_image)
        img_ori.show(title=img_name)


    # 计算该图像正确率（a与img_result_tensor比较）
    temp = 0
    tot = 1280 * 1280
    minus = tot_image - img_result_tensor
    minus = minus == 0
    minus = minus.to(torch.float)
    acc = minus.sum() / tot
    print(img_name)
    print(acc)




