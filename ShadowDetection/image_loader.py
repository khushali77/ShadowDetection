import glob
import os
from PIL import Image
import torch.utils.data as data
resnext_101_32_path = 'resnext_101_32x4d.pth'
#training_root = '/content/drive/MyDrive/ViSha/dataset/train_new'
#training_root = '/content/drive/MyDrive/Merged_Data'
training_root = '/content/drive/MyDrive/Data/SBU/SBU-shadow/SBUTrain4KRecoveredSmall/'

def make_dataset(root):

    # final = []
    # print(root)
    # print(training_root)
    # for dirpath, dirs, files in os.walk(os.path.join(root,'keyframe_images')):
    #   for filename in files:
    #     temp = dirpath.split('/')
    #     final.append((os.path.join(dirpath,filename), os.path.join(root,'keyframe_labels',temp[-1],os.path.splitext(filename)[0]+'.png')))
    # print(len(final))
    # return final
    
    img_list = [os.path.splitext(f) for f in os.listdir(os.path.join(root, 'ShadowImages')) if f.endswith('.jpg') or f.endswith('.png')]
    print(len(img_list))
    return [
        (os.path.join(root, 'ShadowImages', img_name[0]+img_name[1]), os.path.join(root, 'ShadowMasks', img_name[0] + '.png'))
        for img_name in img_list]


class ImageFolder(data.Dataset):
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        #print(img_path)
        
        img = Image.open(img_path).convert('RGB')
        #print("shape:",img.size)
        target = Image.open(gt_path).convert('L')
        #print("mask:",target.size)

        #added
        w,h = img.size
        target = target.resize((w,h) , Image.BILINEAR)

        img, target = self.joint_transform(img, target)
        img = self.transform(img)
        target = self.target_transform(target)
        #print("after transform:" ,img.shape)
        
        #print("mask:",target.shape)
        # print(target)
        return img, target

    def __len__(self):
        return len(self.imgs)