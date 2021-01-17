from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import LeakyReLU, Conv2DTranspose
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import BatchNormalization
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader





class Gan():
    def __init__(self):
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)

        self.dataset_name = 'apple2orange'
        self.data_loader = DataLoader(dataset=self.dataset_name,
                                      image_shape=(self.img_rows, self.img_cols))

        # Calculate output shape of Discriminator (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # build and compile discriminator: A -> [real/fake]
        self.d_model_A = self.build_discriminator()
        self.d_model_A.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        # build and compile discriminator: B -> [real/fake]
        self.d_model_B = self.build_discriminator()
        self.d_model_B.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        # build generator: A -> B
        self.g_AtoB = self.build_generator() 
        # build generator: B -> A
        self.g_BtoA = self.build_generator()

        # define input images for both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # identity element
        fake_B = self.g_AtoB(img_A)
        # forward cycle
        fake_A = self.g_BtoA(img_B)
        # backward cycle
        back_to_A = self.g_BtoA(fake_B)
        back_to_B = self.g_AtoB(fake_A)
        # Identity mapping of images
        img_A_id = self.g_BtoA(img_A)
        img_B_id = self.g_AtoB(img_B)


        # For the combined model we will only train the generators
        self.d_model_A.trainable = False
        self.d_model_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_model_A(fake_A)
        valid_B = self.d_model_B(fake_B)

        # define a composite model for updating generators by adversarial and cycle loss
        self.composite_model = Model(inputs=[img_A, img_B], outputs=[valid_A, valid_B, back_to_A, back_to_B, img_A_id, img_B_id])

        # compile model with weighting of least squares loss and L1 loss
        self.composite_model.compile(loss=['mse', 'mse',
                                           'mae', 'mae', 
                                           'mae', 'mae'],
                                     loss_weights=[1, 1, 10, 10, 1, 1], 
                                     optimizer = optimizer)

    def build_discriminator(self):
        'Define the Discriminator model'
        # initialization weight
        init = RandomNormal(stddev=0.02)
        # input_image
        in_image = Input(shape=self.img_shape)
        # optimizer use
        optimizer = Adam(lr=0.0002, beta_1=0.5)

        def disc_layer(in_image, out_channels, strides=(2,2), instance_norm=True, initializer=init):
            'Layer for building Discriminator'
            d = Conv2D(out_channels, kernel_size=(4,4), strides=strides, padding='same', kernel_initializer=initializer)(in_image)
            if instance_norm:
                d = InstanceNormalization(axis=-1)(d)
            d = LeakyReLU(alpha=0.2)(d)
            return d

        # convolutions layers
        d = disc_layer(in_image, 64, instance_norm=False)
        d = disc_layer(d, 128)
        d = disc_layer(d, 256)
        d = disc_layer(d, 512)
        d = disc_layer(d, 512, strides=(1,1))

        # output layer
        out = Conv2D(1, 4, padding='same', kernel_initializer=init)(d)

        # define model
        model = Model(in_image, out)
        return model

    def build_generator(self, n_resnet=9):
        'Define the Generator model'
        # initialization weight
        init = RandomNormal(stddev=0.02)
        # input_image
        in_image = Input(shape=self.img_shape)

        def resnet_block(n_filters, input_layer, initializer=init):
            'Residual Connection block for building generator'

            # first layer
            rb = Conv2D(filters=n_filters, kernel_size=3, padding='same', kernel_initializer=initializer)(input_layer)
            rb = InstanceNormalization(axis=-1)(rb)
            rb = Activation('relu')(rb)

            # second layer
            rb = Conv2D(filters=n_filters, kernel_size=3, padding='same', kernel_initializer=initializer)(rb)
            rb = InstanceNormalization(axis=-1)(rb)

            # residual connection 
            rb = Concatenate()([rb, input_layer])
            return rb

        def main_block(input_layer, in_features=64, downsampling=True, initializer=init):
            'Downsampling or Upsampling block'
            if downsampling == True:
                out_features = in_features*2
                g = Conv2D(out_features, kernel_size=3, strides=(2,2), padding='same', kernel_initializer=initializer)(input_layer)
            elif downsampling == False:
                out_features = in_features//2
                g = Conv2DTranspose(out_features, kernel_size=3, strides=(2,2), padding='same', kernel_initializer=initializer)(input_layer)

            
            g = InstanceNormalization(axis=-1)(g)
            g = Activation('relu')(g)
            return g

        # c7s1-64
        g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
        g = InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)
    
        # d128     
        g = main_block(input_layer=g, in_features=64, downsampling=True)
        # d256
        g = main_block(input_layer=g, in_features=128, downsampling=True)

        # R256
        for _ in range(n_resnet):
            g = resnet_block(256, g)

        # u128
        g = main_block(input_layer=g, in_features=256, downsampling=False)
        # u64
        g = main_block(input_layer=g, in_features=128, downsampling=False)

        # c7s1-3
        g = Conv2D(3, (7,7), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization(axis=-1)(g)
        out_image = Activation('tanh')(g)
    
        model = Model(in_image, out_image)
        return model

    
    def train(self, epochs, batch_size, sample_interval=5):
        
        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)
        
        # Train loop
        for epoch in range(epochs):
            for batch, (img_A, img_B) in enumerate(self.data_loader.load_batch(batch_size)):

                # Discriminators
                #-----------------------------------------
                # translate images to the other domain
                fake_B = self.g_AtoB.predict(img_A)
                fake_A = self.g_BtoA.predict(img_B)
                
                # train Discriminator A
                d_model_A_loss_real = self.d_model_A.train_on_batch(img_A, valid)
                d_model_A_loss_fake = self.d_model_A.train_on_batch(fake_A, fake)
                d_model_A_loss = 0.5 * np.add(d_model_A_loss_real, d_model_A_loss_fake)
                # train Discriminator B
                d_model_B_loss_real = self.d_model_B.train_on_batch(img_B, valid)
                d_model_B_loss_fake = self.d_model_B.train_on_batch(fake_B, fake)
                d_model_B_loss = 0.5 * np.add(d_model_B_loss_real, d_model_B_loss_fake)

                # total loss of Discriminators
                total_d_loss = 0.5 * np.add(d_model_A_loss, d_model_B_loss)



                # Generators
                #---------------------------------------
                g_loss = self.composite_model.train_on_batch([img_A, img_B],
                                                             [valid, valid, img_A, img_B, img_A, img_B])

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                                                                        % ( epoch, epochs,
                                                                            batch, self.data_loader.n_batches,
                                                                            total_d_loss[0], 100*total_d_loss[1],
                                                                            g_loss[0],
                                                                            np.mean(g_loss[1:3]),
                                                                            np.mean(g_loss[3:5]),
                                                                            np.mean(g_loss[5:6]),
                                                                            elapsed_time))
                

                # If at save interval => save generated image samples
                if batch % sample_interval == 0:
                    self.sample_images(epoch, batch)

                if batch % 150 == 0:
                    self.g_AtoB.save('./models/generatorA2B.h5')
                    self.g_BtoA.save('./models/generatorB2A.h5')
                    self.d_model_A.save('./models/discriminatorA.h5')
                    self.d_model_B.save('./models/discriminatorB.h5')

                    #self.g_AtoB.save_weights('models/generatorA2B_weights_epoch_%d.h5' % epoch)
                    #self.g_BtoA.save_weights('models/generatorB2A_weights_epoch_%d.h5' % epoch)
                    #self.d_model_A.save_weights('models/discriminatorA_weights_epoch_%d.h5' % epoch)
                    #self.d_model_B.save_weights('models/discriminatorB_weights_epoch_%d.h5' % epoch)

    def sample_images(self, epoch, batch):
        os.makedirs('./images/%s' % self.dataset_name, exist_ok=True)
        r, c = 2, 3

        imgs_A = self.data_loader.load_dataset(domain="A", batch_size=1, test=True)
        imgs_B = self.data_loader.load_dataset(domain="B", batch_size=1, test=True)


        # Translate images to the other domain
        fake_B = self.g_AtoB.predict(imgs_A)
        fake_A = self.g_BtoA.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = self.g_BtoA.predict(fake_B)
        reconstr_B = self.g_AtoB.predict(fake_A)

        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("./images/%s/%d_%d.png" % (self.dataset_name, epoch, batch))
        plt.close()





if __name__ == '__main__':
    gan = Gan()
    #checkpoint = ModelCheckpoint('model{epoch:08d}.h5', period=1)
    gan.train(epochs=200, batch_size=5, sample_interval=80)
