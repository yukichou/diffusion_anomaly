import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
from torchvision import transforms
import random
from PIL import Image

class MVTec_Anomaly_Detection(Dataset):
    def __init__(self, args,sample_name,length=5000,anomaly_id=None,recon=False):
        self.recon=recon
        self.good_path='%s/%s/train/good'%(args.mvtec_path,sample_name)
        self.good_files=[os.path.join(self.good_path,i) for i in os.listdir(self.good_path)]
        self.root_dir = '%s/%s'%(args.generated_data_path,sample_name)
        self.anomaly_names=os.listdir(self.root_dir)
        if anomaly_id!=None:
            self.anomaly_names=self.anomaly_names[anomaly_id:anomaly_id+1]
            print('training subsets',self.anomaly_names)
        l=len(self.anomaly_names)
        self.anomaly_num = l
        self.img_paths=[]
        self.mask_paths=[]
        for idx,anomaly in enumerate(self.anomaly_names):
            img_path=[]
            mask_path=[]
            for i in range(min(len(os.listdir(os.path.join(self.root_dir,anomaly,'mask'))),500)):
                img_path.append(os.path.join(self.root_dir,anomaly,'image','%d.jpg'%i))
                mask_path.append(os.path.join(self.root_dir,anomaly,'mask','%d.jpg'%i))
            self.img_paths.append(img_path.copy())
            self.mask_paths.append(mask_path.copy())
        for i in range(l):
            print(len(self.img_paths[i]),len(self.mask_paths[i]))
        self.loader=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256,256])
        ])
        self.length=length
        if self.length is None:
            self.length=len(self.good_files)
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if random.random()>0.5:
            image=self.loader(Image.open(self.good_files[idx%len(self.good_files)]).convert('RGB'))
            mask=torch.zeros((1,image.size(-2),image.size(-1)))
            has_anomaly = np.array([0], dtype=np.float32)
            sample = {'image': image, 'has_anomaly': has_anomaly, 'mask': mask, 'anomay_id': -1}
            if self.recon:
                sample['source']=image
        else:
            anomaly_id=random.randint(0,self.anomaly_num-1)
            img_path=self.img_paths[anomaly_id][idx% len(self.mask_paths[anomaly_id])]
            image = self.loader(Image.open(img_path).convert('RGB'))
            mask_path = self.mask_paths[anomaly_id][idx % len(self.mask_paths[anomaly_id])]
            mask = self.loader(Image.open(mask_path).convert('L'))
            mask=(mask>0.5).float()
            if mask.sum()==0:
                has_anomaly = np.array([0], dtype=np.float32)
                anomaly_id=-1
            else:
                has_anomaly = np.array([1], dtype=np.float32)
            sample = {'image': image, 'has_anomaly': has_anomaly, 'mask': mask, 'anomay_id': anomaly_id}
            if self.recon:
                img_path = self.img_paths[anomaly_id][idx % len(self.mask_paths[anomaly_id])]
                img_path=img_path.replace('image','recon')
                ori_image = self.loader(Image.open(img_path).convert('RGB'))
                sample['source']=ori_image
        return sample
        
# class MVTec_Anomaly_Detection_ours(Dataset):
#     def __init__(self, args,sample_name,length=5000,anomaly_id=None,recon=False):
#         self.recon=recon
#         self.good_path='%s/%s/train/good'%(args.mvtec_path,sample_name)
#         self.good_files=[os.path.join(self.good_path,i) for i in os.listdir(self.good_path)]
#         self.root_dir = '%s/%s'%(args.generated_data_path,sample_name)
#         self.anomaly_names=os.listdir(self.root_dir)
#         if anomaly_id!=None:
#             self.anomaly_names=self.anomaly_names[anomaly_id:anomaly_id+1]
#             print('training subsets',self.anomaly_names)
#         l=len(self.anomaly_names)
#         self.anomaly_num = l
#         self.img_paths=[]
#         self.mask_paths=[]
#         for idx,anomaly in enumerate(self.anomaly_names):
#             img_path=[]
#             mask_path=[]
#             for i in range(min(len(os.listdir(os.path.join(self.root_dir,anomaly,'masks'))),500)):
#                 img_path.append(os.path.join(self.root_dir,anomaly,'image','%d.jpg'%i))
#                 mask_path.append(os.path.join(self.root_dir,anomaly,'masks','%d.jpg'%i))
#             self.img_paths.append(img_path.copy())
#             self.mask_paths.append(mask_path.copy())
#         for i in range(l):
#             print(len(self.img_paths[i]),len(self.mask_paths[i]))
#         self.loader=transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Resize([256,256])
#         ])
#         self.length=length
#         if self.length is None:
#             self.length=len(self.good_files)
#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         if random.random()>0.5:
#             image=self.loader(Image.open(self.good_files[idx%len(self.good_files)]).convert('RGB'))
#             mask=torch.zeros((1,image.size(-2),image.size(-1)))
#             has_anomaly = np.array([0], dtype=np.float32)
#             sample = {'image': image, 'has_anomaly': has_anomaly, 'mask': mask, 'anomay_id': -1}
#             if self.recon:
#                 sample['source']=image
#         else:
#             anomaly_id=random.randint(0,self.anomaly_num-1)
#             img_path=self.img_paths[anomaly_id][idx% len(self.mask_paths[anomaly_id])]
#             image = self.loader(Image.open(img_path).convert('RGB'))
#             mask_path = self.mask_paths[anomaly_id][idx % len(self.mask_paths[anomaly_id])]
#             mask = self.loader(Image.open(mask_path).convert('L'))
#             mask=(mask>0.5).float()
#             if mask.sum()==0:
#                 has_anomaly = np.array([0], dtype=np.float32)
#                 anomaly_id=-1
#             else:
#                 has_anomaly = np.array([1], dtype=np.float32)
#             sample = {'image': image, 'has_anomaly': has_anomaly, 'mask': mask, 'anomay_id': anomaly_id}
#             if self.recon:
#                 img_path = self.img_paths[anomaly_id][idx % len(self.mask_paths[anomaly_id])]
#                 img_path=img_path.replace('image','recon')
#                 ori_image = self.loader(Image.open(img_path).convert('RGB'))
#                 sample['source']=ori_image
#         return sample
        
class MVTec_classification_train(Dataset):
    def __init__(self, args,sample_name):
        self.root_dir = '%s/%s'%(args.generated_data_path,sample_name)
        self.root_dir = '%s/%s'%(args.generated_data_path,sample_name)
        self.anomaly_names=os.listdir(self.root_dir)
        self.img_paths=[]
        self.labels=[]
        for idx, anomaly in enumerate(self.anomaly_names):
            anomaly_dir = os.path.join(self.root_dir, anomaly, 'image')
            
            image_files = [f for f in os.listdir(anomaly_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

            for i, image_file in enumerate(image_files[:min(len(image_files), 500)]):
                image_path = os.path.join(anomaly_dir, image_file)
                self.img_paths.append(image_path)
                self.labels.append(idx)
        self.loader=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256,256])
        ])
        self.length=len(self.img_paths)
    def __len__(self):
        return self.length*5
    def class_num(self):
        return len(self.anomaly_names)
    def return_anomaly_names(self):
        return self.anomaly_names
    def __getitem__(self, idx):
        image=self.loader(Image.open(self.img_paths[idx%len(self.img_paths)]).convert('RGB'))
        label=self.labels[idx%len(self.img_paths)]
        return image,label

class MVTec_classification_test(Dataset):
    def __init__(self, args,sample_name,anomaly_names,num_images):
        root_dir = '%s/%s/test'%(args.mvtec_path,sample_name)
        self.anomaly_names=anomaly_names
        self.num_images = num_images
        self.img_paths=[]
        self.labels=[]
        for idx, anomaly_name in enumerate(self.anomaly_names):
            img_path=os.path.join(root_dir,anomaly_name)
            img_files = os.listdir(img_path)
            img_files.sort(key=lambda x: int(x[:3]))
            # l = len(img_files) // 3
            l = num_images
            self.img_paths += [os.path.join(img_path, file_name) for file_name in img_files[l:]]
            self.labels+=[idx for file_name in img_files[l:]]
        self.loader=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256,256])
        ])
        self.length=len(self.img_paths)
    def __len__(self):
        return self.length
    def class_num(self):
        return len(self.anomaly_names)
    def __getitem__(self, idx):
        image=self.loader(Image.open(self.img_paths[idx%len(self.img_paths)]).convert('RGB'))
        label=self.labels[idx%len(self.img_paths)]
        return image,label

class MVTec_classification_test_one_third(Dataset):
    def __init__(self, args,sample_name,anomaly_names,num_images):
        root_dir = '%s/%s/test'%(args.mvtec_path,sample_name)
        self.anomaly_names=anomaly_names
        self.num_images = num_images
        self.img_paths=[]
        self.labels=[]
        for idx, anomaly_name in enumerate(self.anomaly_names):
            img_path=os.path.join(root_dir,anomaly_name)
            img_files = os.listdir(img_path)
            img_files.sort(key=lambda x: int(x[:3]))
            l = len(img_files) // 3
            self.img_paths += [os.path.join(img_path, file_name) for file_name in img_files[l:]]
            self.labels+=[idx for file_name in img_files[l:]]
        self.loader=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256,256])
        ])
        self.length=len(self.img_paths)
    def __len__(self):
        return self.length
    def class_num(self):
        return len(self.anomaly_names)
    def __getitem__(self, idx):
        image=self.loader(Image.open(self.img_paths[idx%len(self.img_paths)]).convert('RGB'))
        label=self.labels[idx%len(self.img_paths)]
        return image,label