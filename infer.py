import numpy as np
#np.random.seed(777)
import chainer
from chainer import cuda
from chainer import serializers
import chainer.functions as F

import argparse
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import PIL
from PIL import ImageDraw
import get_fish_data as gfd


parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument('--original', action='store_true',
                    default=True, help='train on original MNIST')
group.add_argument('--translated', action='store_true',
                    default=False, help='train on translated MNIST')
group.add_argument('--cluttered', action='store_true',
                    default=False, help='train on translated & cluttered MNIST')
parser.add_argument('--lstm', type=bool, default=False,
                    help='use LSTM units in core layer')
parser.add_argument('-m', '--model', type=str,
                    default='models/ram_original_epoch800.chainermodel',
                    help='load model weights from given file')
parser.add_argument('-g', '--gpuid', type=int, default=-1,
                    help='GPU device ID (default is CPU)')
args = parser.parse_args()



# hyper-params for each task
filename = 'ram_original'
g_size = 64
n_steps = 6
n_scales = 1

# init RAM model
from ram import RAM
model = RAM(g_size=g_size, n_steps=n_steps, n_scales=n_scales, use_lstm=args.lstm)

print('load model from {}'.format(args.model))
serializers.load_hdf5(args.model, model)

test = gfd.make_test_tuple_dataset(gfd.read_from_csv(gfd.test_file_name))
test_data, test_targets = np.array(test).transpose()
test_data = np.array(list(test_data)).reshape(test_data.shape[0], 1, 256, 256)


def test_all_data(model, test_data, file_name):
    result_file = open(file_name, "w")
    for i in range(len(test_data)):
        x = test_data[i][np.newaxis,:,:,:]
        init_l = np.random.uniform(low=-1, high=1, size=2)
        y, ys, ls = model.infer(x, init_l)
        str_y = ['{:.2f}'.format(x) for x in ys[-1]]
        result_file.write(test_targets[i] + "," + ",".join(str_y))
        result_file.write("\n")


def display_result(model, test_data, index):
    image = PIL.Image.fromarray(test_data[index][0] * 255).convert('RGB')
    x = test_data[index][np.newaxis, :, :, :]
    init_l = np.random.uniform(low=-1, high=1, size=2)
    y, ys, ls = model.infer(x, init_l)
    locs = ((ls + 1) / 2) * (np.array(test_data.shape[2:4]) + 1)

    # plot results
    from crop import crop
    plt.subplots_adjust(wspace=0.35, hspace=0.05)

    for t in range(0, n_steps):
        # digit with glimpse
        plt.subplot(3 + n_scales, n_steps, t + 1)

        # green if correct otherwise red
        if np.argmax(ys[t]) == test_targets[index]:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)

        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        xy = np.array([locs[t, 1], locs[t, 0], locs[t, 1], locs[t, 0]])
        wh = np.array([-g_size // 2, -g_size // 2, g_size // 2, g_size // 2])
        xys = [xy + np.power(2, s) * wh for s in range(n_scales)]

        for xy in xys:
            draw.rectangle(xy=list(xy), outline=color)
        del draw
        plt.imshow(canvas)
        plt.axis('off')

        # glimpse at each scale
        gs = crop(x, center=ls[t:t + 1], size=g_size)
        plt.subplot(3 + n_scales, n_steps, n_steps + t + 1)
        plt.imshow(gs.data[0, 0], cmap='gray')
        plt.axis('off')

        for k in range(1, n_scales):
            s = np.power(2, k)
            patch = crop(x, center=ls[t:t + 1], size=g_size * s)
            patch = F.average_pooling_2d(patch, ksize=s)
            gs = F.concat((gs, patch), axis=1)
            plt.subplot(3 + n_scales, n_steps, n_steps * (k + 1) + t + 1)
            plt.imshow(gs.data[0, k], cmap='gray')
            plt.axis('off')

        # output probability
        plt.subplot2grid((3 + n_scales, n_steps), (1 + n_scales, t), rowspan=2)
        plt.barh(np.arange(8), ys[t], align='center')
        plt.xlim(0, 1)
        plt.ylim(-0.5, 9.5)

        if t == 0:
            plt.yticks(np.arange(8))
        else:
            plt.yticks(np.arange(8), ['' for _ in range(8)])
        plt.xticks([])

    plt.show()

# test_all_data(model, test_data, "result_file_stage_1")
# display_result(model, test_data, 2)
