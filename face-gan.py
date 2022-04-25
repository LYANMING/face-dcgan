import numpy as np
import pandas as pd
import os
import tensorflow as tf
PIC_DIR = 'image/'

from tqdm import tqdm
from PIL import Image

IMAGES_COUNT = 10000

ORIG_WIDTH = 178
ORIG_HEIGHT = 208
diff = (ORIG_HEIGHT - ORIG_WIDTH)//2

WIDTH = 128
HEIGHT = 128
crop_rect = (0, diff, ORIG_WIDTH, ORIG_HEIGHT -diff)

images = []
for pic_file in tqdm(os.listdir(PIC_DIR)[:IMAGES_COUNT]):
    pic = Image.open(PIC_DIR + pic_file).crop(crop_rect)
    pic.thumbnail((WIDTH, HEIGHT), Image.ANTIALIAS)
    images.append(np.uint8(pic))


images = np.array(images) / 255
print(images.shape)
from matplotlib import pyplot as plt

plt.figure(1, figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(images[i])
    plt.axis('off')
plt.show()

# 生成器
# 卷积神经网络，每次在图片是取一小块的数据
from keras import Input  # keras用 Python 编写的高级神经网络 API
from keras.layers import Dense, Reshape, LeakyReLU, Conv2D, Conv2DTranspose, Flatten, Dropout
from keras.models import Model
from keras.optimizers import RMSprop

LATENT_DIM = 32  # 32维的随机向量
CHANNELS = 3  # rgb通道，灰度是1


def create_generator():
    gen_input = Input(shape=(LATENT_DIM,))  # 实例化keras张量，shape（32，）表示预期的输入将是一批32维的向量

    x = Dense(128 * 16 * 16)(gen_input)  # 全连接层
    x = LeakyReLU()(x)  # 高级激活函数
    x = Reshape((16, 16, 128))(x)  # 实现不同维度任意层之间的对接

    x = Conv2D(256, 5, padding='same')(x)  # padding 填充方式
    x = LeakyReLU()(x)

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, 5, padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(256, 5, padding='same')(x)   # padding填充
    x = LeakyReLU()(x)
    x = Conv2D(CHANNELS, 7, activation='tanh', padding='same')(x)   # activation激活，生成的值在-1~1之间

    generator = Model(gen_input, x)
    return generator


# 判别器
def create_discriminator():
    disc_input = Input(shape=(HEIGHT, WIDTH, CHANNELS))

    x = Conv2D(256, 3)(disc_input)
    x = LeakyReLU()(x)

    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, 4, strides=2)(x)    # Conv2D多用于图片的识别，提取特征值
    x = LeakyReLU()(x)

    x = Flatten()(x)
    x = Dropout(0.4)(x)

    x = Dense(1, activation='sigmoid')(x)
    discriminator = Model(disc_input, x)

    optimizer = RMSprop(
        lr=.0001,
        clipvalue=1.0,
        decay=1e-8
    )

    discriminator.compile(
        optimizer=optimizer,
        loss='binary_crossentropy'  # 二分类 binary_crossentropy交叉熵损失函数
    )

    return discriminator



generator = create_generator()
discriminator = create_discriminator()

print(generator)
print(discriminator)
discriminator.trainable = False # 在gan模型中判别器固定不变
# 当discriminator被compile之后，即使设置了discriminator.trainable=False，该discriminator仍然可以通过train_on_batch的方式被训练；
gan_input = Input(shape=(LATENT_DIM, ))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)
optimizer = RMSprop(lr=.0001, clipvalue=1.0, decay=1e-8)    # lr学习率，decay每次参数更新后学习率衰减值，clipvalue用于控制梯度裁剪
gan.compile(optimizer=optimizer, loss='binary_crossentropy')


import time
iters = 100
batch_size = 1

RES_DIR = 'C:/Users/25321/Desktop/'
FILE_PATH = '%s/generated_%d.png'
if not os.path.isdir(RES_DIR):
    os.mkdir(RES_DIR)

CONTROL_SIZE_SQRT = 6
control_vectors = np.random.normal(size=(CONTROL_SIZE_SQRT**2, LATENT_DIM)) / 2     # 正态分布抽取随机样本


start = 0
d_losses = []   # discriminator损失
a_losses = []   # 整个模型的损失
images_saved = 0
for step in range(iters):
    start_time = time.time()    # 计时
    latent_vectors = np.random.normal(size=(batch_size, LATENT_DIM))    # 随机生成噪声
    generated = generator.predict(latent_vectors)   # 生成一批虚假图片

    real = images[start:start + batch_size]
    combined_images = np.concatenate([generated, real]) # 混合真实和虚假图片

    labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])  # 前一批标签为1，后一批标签为0.因为原始的生成器Loss存在梯度消失问题；训练生成器的时候，考虑反转标签，real=fake, fake=real
    labels += .05 * np.random.random(labels.shape)  # 噪声标签。0或者1的标签，我们称之为hard label，这种标签可能会使得判别器的损失值迅速跌落至0

    d_loss = discriminator.train_on_batch(combined_images, labels)  # 对判别器进行训练
    d_losses.append(d_loss)

    latent_vectors = np.random.normal(size=(batch_size, LATENT_DIM))
    misleading_targets = np.zeros((batch_size, 1))

    a_loss = gan.train_on_batch(latent_vectors, misleading_targets) # 训练生成器
    a_losses.append(a_loss)

    start += batch_size
    if start > images.shape[0] - batch_size:
        start = 0

    if step % 50 == 49: # 隔多少次保存一次模型
        gan.save_weights('checkpoint/gan.h5')

        print('%d/%d: d_loss: %.4f,  a_loss: %.4f.  (%.1f sec)' % (step + 1, iters, d_loss, a_loss, time.time() - start_time))

        control_image = np.zeros((WIDTH * CONTROL_SIZE_SQRT, HEIGHT * CONTROL_SIZE_SQRT, CHANNELS))
        control_generated = generator.predict(control_vectors)
        for i in range(CONTROL_SIZE_SQRT ** 2):
            x_off = i % CONTROL_SIZE_SQRT
            y_off = i // CONTROL_SIZE_SQRT
            control_image[x_off * WIDTH:(x_off + 1) * WIDTH, y_off * HEIGHT:(y_off + 1) * HEIGHT, :] = control_generated[i, :, :, :]
        im = Image.fromarray(np.uint8(control_image * 255))
        im.save(FILE_PATH % (RES_DIR, images_saved))
        images_saved += 1


plt.figure(1, figsize=(12, 8))
plt.subplot(121)
plt.plot(d_losses)
plt.xlabel('epochs')
plt.ylabel('discriminant losses')
plt.subplot(122)
plt.plot(a_losses)
plt.xlabel('epochs')
plt.ylabel('adversary losses')
plt.show()


import imageio
import shutil

images_to_gif = []
for filename in os.listdir(RES_DIR):
    images_to_gif.append(imageio.imread(RES_DIR + '/' + filename))
imageio.mimsave('C:/Users/25321/Desktop/visual.gif', images_to_gif)
shutil.rmtree(RES_DIR)