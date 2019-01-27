import json
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import os

Image.MAX_IMAGE_PIXELS = 1000000000


def make_dataset(img_file, ano_df, L, image_id, only_ship=1):
    if not 'dataset' in os.listdir():
        os.mkdir('dataset')
        os.mkdir('dataset/images')
        os.mkdir('dataset/labels')
    img = np.array(Image.open(img_file))
    imgs = []
    print('image size: ' + str(img.shape))
    cnt = 0
    for i in range(int(img.shape[0] / L)):
        for j in range(int(img.shape[1] / L)):
            s_img = img[i * L:(i + 1) * L, j * L:(j + 1) * L, :]
            if np.sum(s_img) > 0:
                s_img = Image.fromarray(s_img)
                imgs.append(s_img)
                s_df = ano_df[ano_df['X'] == j]
                s_df = s_df[s_df['Y'] == i]
                if len(s_df) > 0 or not only_ship:
                    cnt += 1
                    s_img.save('dataset/images/{0}_{1}_{2}_sat_img.jpg'.format(image_id, i, j))
                    label = ''
                    # draw = ImageDraw.Draw(s_img)
                    for b in s_df.index:
                        # draw.rectangle((s_df.loc[b,'x1'],s_df.loc[b,'y1'],s_df.loc[b,'x2'],s_df.loc[b,'y2']))
                        label = label + '{0} {1} {2} {3} {4}\n'.format(s_df.loc[b, 'class'], \
                                float(s_df.loc[b, 'x1'] + s_df.loc[b, 'x2']) / (2 * L), float(s_df.loc[b, 'y1'] + \
                                s_df.loc[b, 'y2']) / (2 * L),float(s_df.loc[b, 'x2'] - s_df.loc[b, 'x1']) /(L), \
                                float(s_df.loc[b, 'y2'] - s_df.loc[b, 'y1']) / (L))
                    with open('dataset/labels/{0}_{1}_{2}_sat_img.txt'.format(image_id, i, j), mode='w') as f:
                        f.write(label)
                    # plt.figure(figsize=(10,10))
                    # plt.imshow(s_img)
                    # plt.show()

    print(str(cnt) + ' images gnerated!!')
    return


def fd(d):
    result = []
    cls_dic = {'ship_moving': 0, 'ship_not_moving': 1, 'barge': 2, 'unknown': 3}
    result.append(cls_dic[d['category']])
    b = d['box2d']
    if (int(b['x1'] / L) == int(b['x2'] / L) and (int(b['y1'] / L) == int(b['y2'] / L))):
        result.append(int(b['x1'] / L))
        result.append(int(b['y1'] / L))
        result.append(b['x1'] % L)
        result.append(b['y1'] % L)
        result.append(b['x2'] % L)
        result.append(b['y2'] % L)
    else:
        for i in range(6):
            result.append(-1)
    return result


def make_annotation_df(anno_file, L):
    df = pd.DataFrame(columns=['class', 'X', 'Y', 'x1', 'y1', 'x2', 'y2'], dtype='int16')
    f = open(anno_file)
    json_dict = json.load(f)
    for i in range(len(json_dict['labels'])):
        df.loc[i, :] = fd(json_dict['labels'][i])
    for c in df.columns:
        df[c] = df[c].astype('int16')
    return df


L = 416
for i in range(15,20):
    img_file = 'train_images/train_' + str(i).zfill(2) + '.jpg'
    anno_file = 'train_annotations/train_' + str(i).zfill(2) + '.json'
    print(img_file)
    df = make_annotation_df(anno_file, L)
    make_dataset(img_file, df, L, i)
