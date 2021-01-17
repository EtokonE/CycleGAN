ffrom glob import glob
import numpy as np
import cv2
from matplotlib.pyplot import imread
class DataLoader():
    def __init__(self, dataset, image_shape=(128,128)):
        self.dataset_name = dataset
        self.image_shape = image_shape

    def read_image(self, image_path):
        'Read an image from a file as an array'
        image = imread(image_path, 'RGB').astype(np.float)
        return image

    def load_image(self, image_path):
        'Read and prepare image'
        image = self.read_image(image_path)
        image = cv2.resize(image, self.image_shape)
        image = image/127.5 - 1.
        return image

    def load_dataset(self, domain, batch_size=5, test=False):
        'Load data from dataset folders, and prepare image for batch'
        # Select the folder:  'testA' / 'trainB'...
        image_shape = (128,128)
        if test:
            data_split = f'test{domain}'
        else:
            data_split = f'train{domain}'

        path = glob(f'./cyclegan_datasets/{self.dataset_name}/{data_split}/*')    

        # define batch
        batch = np.random.choice(a=path, size=batch_size)

        # prepare images
        images = []
        for image in batch:
            # read
            image = self.read_image(image)
            # resize
            if test:
                image = cv2.resize(image, image_shape)
            else:
                image = cv2.resize(image, image_shape)
                # flip
                if np.random.random() > 0.5:
                    image = np.fliplr(img)
            
            images.append(image)
        # normalization
        images = np.array(images)/127.5 - 1.
        
        return images

    


    def load_batch(self, batch_size=5, image_shape=(128,128)):
        # define path
        path_A = glob('./cyclegan_datasets/%s/trainA/*' % (self.dataset_name))
        path_B = glob('./cyclegan_datasets/%s/trainB/*' % (self.dataset_name))

        # how many batches are in the dataset
        self.n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = self.n_batches * batch_size


        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)

        for i in range(self.n_batches-1):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):
                img_A = self.read_image(img_A)
                img_B = self.read_image(img_B)

                img_A = cv2.resize(img_A, image_shape)
                img_B = cv2.resize(img_B, image_shape)

                if np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B
