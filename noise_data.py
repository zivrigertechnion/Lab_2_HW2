import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import os


def noisy(noise_typ, image,num):
    if noise_typ == "gauss":
        row, col = image.shape
        mean = 0
        sigma = 0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row, col = image.shape
        s_vs_pvec = [0.5,0.25]
        s_vs_p = s_vs_pvec[num%2]
        amountvec = [0.004,0.008]
        amount = amountvec[int(num>2)]
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[tuple(coords)] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[tuple(coords)] = 0
        return out

    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals/4) / float(vals)
        return noisy

    elif noise_typ =="speckle":
        row, col = image.shape
        gauss = np.random.randn(row, col)
        gauss = gauss.reshape(row, col)
        noisy = image + image * gauss / 4
        return noisy


if __name__ == '__main__':
    if not os.path.exists('train_new'):
        os.makedirs('train_new')
    methods_list = ["s&p", "speckle", "gauss", "poisson"]
    names_list = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']
    for name in names_list:
        if not os.path.exists('train_new/'+name):
            os.makedirs('train_new/'+name)
        png_file_names = ['train/' + name + '/' + f for f in listdir('train/' + name)]
        for png in png_file_names:
            method = "s&p"
            img = plt.imread(png)
            # cv2.imshow('no', img)
            #cv2.waitKey(0)
            for i in range(4):
                noisy_img = noisy(method, img,i)
                plt.imsave('train_new/'+png[6:-4]+'_'+str(i)+'_noisy_'+method+'.png', noisy_img,cmap="gray")


