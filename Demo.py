import cv2
import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# img = cv2.imread('./CameraIrRaw/1.bmp')
# imgInfo = img.shape
# size = (imgInfo[1],imgInfo[0])
# print(size)
videoWrite = cv2.VideoWriter(r'E:\My Paper\CVPR\RGBT_TTA\RTFNet-master\visualization\MS-UDA\RTF.mp4',cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),20,(640,512))# 写入对象
# 参数：1 file name 2 编码器 3 帧率 4 size
# 1-50张图片
file_path = r'E:\My Paper\CVPR\RGBT_TTA\RTFNet-master\visualization\MS-UDA'
piclist = os.listdir(file_path)
c=0
for pic in piclist:
    if c>=100:
        print(c)
        break
    if "RTF" in pic and 'mp4' not in pic:
        c = c + 1
        # img = Image.open(file_path+'\\'+pic).convert("RGB").save("w.png")
        # img = cv2.imread("w.png")
        print(file_path+'\\'+pic)
        img = cv2.imread(file_path+'\\'+pic)

        videoWrite.write(img)
videoWrite.release()

# for i in range(1,3978):
#     fileName = r'E:\My Paper\CVPR\RGBT_TTA\RTFNet-master\visualization\MS-UDA'+str(i)+'.png'
#     img = cv2.imread(fileName)
#     videoWrite.write(img) # 写入方法 1 jpg data
# videoWrite.release()
# print('end!')