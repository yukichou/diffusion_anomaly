# quick_fg_mask.py
import cv2, glob, os
root = r'./Defect_Spectrum/DS-MVTec/screw/image/good'
out  = r'./obj_foreground_mask/screw/train/masks'
os.makedirs(out, exist_ok=True)
for p in glob.glob(os.path.join(root, '*.png')):
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    th = 255 - th                     # 亮背景、暗物体时翻转
    th = cv2.medianBlur(th, 5)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))
    mname = os.path.splitext(os.path.basename(p))[0] + '_mask.png'
    cv2.imwrite(os.path.join(out, mname), th)