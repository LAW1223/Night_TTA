import re

#### change dataset
lineList = []
matchPattern = re.compile(r'N')
file = open(r'E:\My Paper\CVPR\RGB-T image Semantic Segmentation\MF\val.txt', 'r', encoding='UTF-8')
while 1:
    line = file.readline()
    print(line)
    if not line:
        print("Read file End or Error")
        break
    elif matchPattern.search(line):
        pass
    else:
        lineList.append(line)
file.close()
print(len(lineList))
file = open(r'E:\My Paper\CVPR\RGB-T image Semantic Segmentation\MF\val_day.txt', 'w', encoding='UTF-8')
for i in lineList:
    file.write(i)
file.close()

import os
# import numpy as np
# from PIL import Image

#### read_image
# def read_image(self, name, folder):
#     file_path = os.path.join(self.data_dir, '%s/%s.png' % (folder, name))
#     image = np.array(Image.open(r"E:\My Paper\CVPR\RGB-T image Semantic Segmentation\MF\labels\00001D.png"))  # (w,h,c)
#     image.flags.writeable = True
#     return image
#
# image = np.array(Image.open(r"E:\My Paper\CVPR\RGBT_TTA\MS-UDA\Labels\set00_V000_I00659.png"))  # (w,h,c)
# image.flags.writeable = True
# print("不重复数字：", np.unique(image))
# print(np.shape(image))

#### replce filename
# def replcaeFileName(pic_path):
#     piclist = os.listdir(pic_path)
#     total_num = len(piclist)
#
#     i = 1
#     print(total_num)
#     for pic in piclist:
#         if pic.endswith(".png"):  # 修改成你自己想要重命名的文件格式
#             old_path = os.path.join(os.path.abspath(pic_path), pic)
#             a=pic[:-15]+pic[-10:]
#             # a=pic[:-7]+".png"
#             print(a)
#             new_path = os.path.join(os.path.abspath(pic_path), a)  # 修改成了1000+N这种格式
#
#             os.renames(old_path, new_path)
#             print("把原图片命名格式：" + old_path + u"转换为新图片命名格式：" + new_path)
#             i = i + 1
#
#
# if __name__ == '__main__':
#     rootDir = r'E:\My Paper\CVPR\RGBT_TTA\MS-UDA_PR\night_th'
#     replcaeFileName(rootDir)


#### generate train val txt for FRT
# import random
#
# def generate_train_val_txt(pic_path):
#     piclist = os.listdir(pic_path)
#     total_num = len(piclist)
#     a=int(0.9*total_num)
#     print(piclist)
#     random.shuffle(piclist)
#     train_list=piclist[0:a]
#     val_list=piclist[a:]
#     print(len(train_list))
#     print(len(val_list))
#
#     file = open(r'E:\My Paper\CVPR\RGBT_TTA\MS-UDA_PR\train_day.txt', 'w', encoding='UTF-8')
#     for i in train_list:
#         file.write(i+'\n')
#
#     file = open(r'E:\My Paper\CVPR\RGBT_TTA\MS-UDA_PR\val_day.txt', 'w', encoding='UTF-8')
#     for i in val_list:
#         file.write(i+'\n')
# #
# def generate_test_txt(pic_path):
#     piclist = os.listdir(pic_path)
#
#     file = open(r'E:\My Paper\CVPR\RGBT_TTA\MS-UDA_PR\test_night.txt', 'w', encoding='UTF-8')
#     for i in piclist:
#         file.write(i+'\n')
# #
# if __name__ == '__main__':
#     rootDir = r'E:\My Paper\CVPR\RGBT_TTA\MS-UDA_PR\day_th'
#     generate_train_val_txt(rootDir)
    # generate_test_txt(rootDir)

### 拼接图像 ###
# import  PIL
# data_dir=r"E:\My Paper\CVPR\RGBT_TTA\MF"
#
#
# def read_image(name, folder):
#     file_path = os.path.join(data_dir, '%s/%s' % (folder, name))
#     image = np.asarray(PIL.Image.open(file_path))
#     return image
#
# image = read_image('1570722156_952177040.png', 'images')
# image = read_image('1570722156_952177040.png', 'RGB')
# Thermal = read_image('1570722156_952177040.png', 'Thermal')
# Thermal=np.expand_dims(Thermal,axis=2)
# print(Thermal.shape)
# image_1 = np.concatenate((image, Thermal),axis=2)
# print(image_1.shape)
#
# print((image_1[:,:,:3]==image).all())


#### 编码图像
# def get_coding_1():
#     coding = {}
#
#     coding[0] = [70, 70, 70]
#     coding[1] = [244, 35, 232]
#     coding[2] = [128, 64, 128]
#     coding[3] = [168, 168, 168]
#     coding[4] = [0, 255, 255]
#     coding[5] = [255, 165, 0]
#     coding[6] = [107, 142, 35]
#     coding[7] = [255, 255, 0]
#     coding[8] = [70, 130, 180]
#     coding[9] = [220, 20, 60]
#     coding[10] = [0, 255, 0]
#     coding[11] = [190, 153, 153]
#
#     coding[12] = [0, 0, 0]
#
#     for k, v in coding.items():
#         v = np.flip(v)
#         coding[k] = v
#
#     return coding
#
# def find_keys(dict, val):
#   return list(key for key, value in dict.items() if value.all == val.all)
#
# def categorize(source, target):
#     coding = get_coding_1()
#     keys = list(coding.keys())
#     values = list(coding.values())
#     # print(values)
#     piclist = os.listdir(source)
#
#     for i in piclist:
#         file_path = os.path.join(source, i)
#         image = np.array(Image.open(file_path))
#         image.flags.writeable = True
#         print("不重复数字：", np.unique(image))
#         label = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
#         for cid in values:
#             print(cid)
#             print(find_keys(coding,cid))
#             print(np.where(image.all == cid.all))
#
#             os._exit()
#             # print(np.where(image == cid))
#             label[image == cid] = find_keys(coding,cid)
#         img = Image.fromarray(np.uint8(label))
#         path= os.path.join(target, i)
#         img.save(path,'png')

# if __name__ == '__main__':
#     source = r'E:\My Paper\CVPR\RGBT_TTA\FRT\test\SegmentationClassPNG'
#     target = r'E:\My Paper\CVPR\RGBT_TTA\FRT\test\Labels'
#     categorize(source, target)

# image = np.array(Image.open(r"E:\My Paper\CVPR\RGBT_TTA\FRT\test\Labels\fl_ir_aligned_1579110420_8345710670_rgb.png"))  # (w,h,c)
# image.flags.writeable = True
# print("不重复数字：", np.unique(image))
# print(image)
# print(np.shape(image))

#### change path
# import os
# import shutil
# import tqdm
#
# old_path = r'E:\My Paper\CVPR\RGBT_TTA\MS-UDA_PR\night'
# new_path = r'E:\My Paper\CVPR\RGBT_TTA\MS-UDA_PR\Labels'
#
# files = os.listdir(old_path)
# for i in tqdm.tqdm(range(len(files))):
#     if (files[i][-11:] == '_pseudo.png'):
#         a=files[i][:-11]+'.png'
#         print(a)
#         old_file_path = old_path + '\\' + files[i]
#         new_file_path = new_path + '\\' + a
#         shutil.copy(old_file_path, new_file_path)

#### copy image MF
import os
import shutil
import tqdm
# from PIL import Image
# import numpy as np
# old_path = r'E:\My Paper\CVPR\RGBT_TTA\MF\images'
# new_path = r'E:\My Paper\CVPR\RGBT_TTA\RTFNet-master\visualization\M2'
#
#
# file = open(r'E:\My Paper\CVPR\RGBT_TTA\MF\test_night.txt', 'r', encoding='UTF-8')
# while 1:
#     line = file.readline()
#     print(line)
#     if not line:
#         print("Read file End or Error")
#         break
#     c=line[:-1]+'.png'
#     a=line[:-1]+'_RGB'+'.png'
#     b=line[:-1]+'_t'+'.png'
#     old_file_path = old_path + '\\' + c
#     new_file_path_rgb = new_path + '\\' + a
#     new_file_path_t = new_path + '\\' + b
#     # print(new_file_path)
#     # shutil.copy(old_file_path, new_file_path)
#     image = np.asarray(Image.open(old_file_path),dtype=np.uint8)
#     print(image[:,:,:3].shape)
#     img = Image.fromarray(np.reshape(image[:,:,:3],(480,640,3)))
#     print(img.size)
#     img.save(new_file_path_rgb)
#     img_t = Image.fromarray(np.reshape(image[:,:,3:],(480,640)))
#     img_t.save(new_file_path_t)

#### copy image MF
import os
import shutil
import tqdm
from PIL import Image
import numpy as np
old_path_RGB = r'E:\My Paper\CVPR\RGBT_TTA\MS-UDA\RGB'
old_path_T = r'E:\My Paper\CVPR\RGBT_TTA\MS-UDA\Thermal'
old_path_V=r'E:\My Paper\CVPR\RGBT_TTA\MS-UDA\Visualize'

new_path = r'E:\My Paper\CVPR\RGBT_TTA\RTFNet-master\visualization\MS-UDA1'


file = open(r'E:\My Paper\CVPR\RGBT_TTA\MS-UDA\test_night.txt', 'r', encoding='UTF-8')
while 1:
    line = file.readline()
    print(line)
    if not line:
        print("Read file End or Error")
        break
    c=line[:-1]
    a=line[:-5]+'_RGB'+'.png'
    b=line[:-5]+'_t'+'.png'
    d=line[:-5]+'_v'+'.png'
    old_file_path_RGB = old_path_RGB + '\\' + c
    old_file_path_t = old_path_T + '\\' + c
    old_file_path_v = old_path_V + '\\' + c

    new_file_path_rgb = new_path + '\\' + a
    new_file_path_t = new_path + '\\' + b
    new_file_path_V = new_path + '\\' + d

    shutil.copy(old_file_path_RGB, new_file_path_rgb)
    shutil.copy(old_file_path_t, new_file_path_t)
    shutil.copy(old_file_path_v, new_file_path_V)


