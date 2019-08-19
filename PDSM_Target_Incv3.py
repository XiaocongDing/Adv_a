import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import  math



tf.logging.set_verbosity(tf.logging.ERROR)
sess = tf.InteractiveSession()

image = tf.Variable(tf.zeros((299, 299, 3)))

def inception(image, reuse):
    preprocessed = tf.multiply(tf.subtract(tf.expand_dims(image, 0), 0.5), 2.0)
    arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        logits, _ = nets.inception.inception_v3(
            preprocessed, 1001, is_training=False, reuse=reuse)
        logits = logits[:,1:] # ignore background class
        probs = tf.nn.softmax(logits) # probabilities
    return logits, probs

logits, probs = inception(image, reuse=False)

import tempfile
from urllib.request import urlretrieve
import tarfile
import os
data_dir = tempfile.mkdtemp()
#inception_tarball, _ = urlretrieve(
#    'http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz')
tarfile.open('../inception_v3_2016_08_28.tar.gz', 'r:gz').extractall(data_dir)
restore_vars = [
    var for var in tf.global_variables()
    if var.name.startswith('InceptionV3/')
]
saver = tf.train.Saver(restore_vars)
saver.restore(sess, os.path.join(data_dir, 'inception_v3.ckpt'))

import json
import matplotlib.pyplot as plt

imagenet_json = '../imagenet.json'
with open(imagenet_json) as f:
    imagenet_labels = json.load(f)


def classify(img, correct_class=None, target_class=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    fig.sca(ax1)
    p = sess.run(probs, feed_dict={image: img})[0]  #probs概率
    ax1.imshow(img)
    fig.sca(ax1)
    #print(logits.eval(), probs.eval())
    topk = list(p.argsort()[-10:][::-1])
    topprobs = p[topk]
    barlist = ax2.bar(range(10), topprobs)
    print(topprobs)
    if target_class in topk:
        barlist[topk.index(target_class)].set_color('r')
    if correct_class in topk:
        barlist[topk.index(correct_class)].set_color('g')
    plt.sca(ax2)
    plt.ylim([0, 1.1])
    plt.xticks(range(10),
               [imagenet_labels[i][:15] for i in topk],
               rotation='vertical')
    fig.subplots_adjust(bottom=0.2)
    plt.show()
    return topprobs

import PIL
from PIL import Image
import numpy as np
import csv
birth_data=[]
with open('../images.csv') as csvfile:
    csv_reader=csv.reader(csvfile)
    birth_header=next(csv_reader)
    for row in csv_reader:
        birth_data.append(row)

image_label=np.array(birth_data)
image_label=image_label[:,[0,6,7]] #文件名。真值标签，目标标签
for x in range(len(image_label)):
    float(image_label[x,1])
for x in range(len(image_label)):
    float(image_label[x,2])
#文件名为主键，建立字典
vecdict=dict(zip(image_label[:,0],image_label[:,[1,2]]))

#构建计算图
x = tf.placeholder(tf.float32, (299, 299, 3))

x_hat = image  # our trainable adversarial input ;;image = tf.Variable(tf.zeros((299, 299, 3)))
assign_op = tf.assign(x_hat, x)

learning_rate = tf.placeholder(tf.float32, ())
y_hat = tf.placeholder(tf.int32, ())
y_hat2 = tf.placeholder(tf.int32,())

labels = tf.one_hot(y_hat, 1000)
labels2 = tf.one_hot(y_hat2, 1000)
loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=[labels])
optim_step = tf.train.GradientDescentOptimizer(
    learning_rate).minimize(loss, var_list=[x_hat])

epsilon = tf.placeholder(tf.float32, ())

below = x - epsilon
above = x + epsilon
projected = tf.clip_by_value(tf.clip_by_value(x_hat, below, above), 0, 1)
with tf.control_dependencies([projected]):
    project_step = tf.assign(x_hat, projected)
xassgign=tf.assign(x_hat,x)

#img_path, _ = urlretrieve('http://www.anishathalye.com/media/2017/07/25/cat.jpg')
success_sta=[]
prob_origin_save=[]
prob_target_save=[]
import time
import pickle
for u in range(0,22,2):
    u = u/10;
    for ratio in range(4,24,4):
        for perturbation in range(8,9):
            count=0
            success = 0;  # 作为成功样本的计数
            unsuccess = 0;
            for filepath in sorted(tf.gfile.Glob(os.path.join('../NIPS2017AAC/','*.png'))):
                time_start=time.time()
                filename=filepath[15:31]
                img_class = vecdict[filename][0]#该图片的正确标签
                img_class=int(img_class)
                img_class-=1

                img = PIL.Image.open(filepath)
                big_dim = max(img.width, img.height)
                wide = img.width > img.height
                new_w = 299 if not wide else int(img.width * 299 / img.height)
                new_h = 299 if wide else int(img.height * 299 / img.width)
                img = img.resize((new_w, new_h)).crop((0, 0, 299, 299))
                img = (np.asarray(img) / 255.0).astype(np.float32)
                imgsave=img.copy()
                img3=img.copy()
                #classify(img, correct_class=img_class)
                tf.reset_default_graph()

                #参数设置
                demo_epsilon = perturbation / 255.0  # a really small perturbation
                demo_lr = 1e-1
                demo_steps = 32
                demo_target = vecdict[filename] [1] # "guacamole"  没有目标的攻击目标就是自己，loss取反从而变大
                demo_target = int(demo_target)
                demo_target -= 1
                n = 2682 * ratio
                gt=0
                # initialization step
                sess.run(assign_op, feed_dict={x: img})
                # projected gradient descent
                img0=img.copy()
                ministep=1/255
                below2=img - perturbation/255
                above2=img + perturbation/255
                #迭代过程
                for i in range(demo_steps):
                    # gradient descent step
                    sess.run(xassgign, feed_dict={x: img})
                    _, loss_value = sess.run(
                        [optim_step, loss],
                        feed_dict={learning_rate: demo_lr, y_hat: demo_target, y_hat2:img_class})
                    # project step
                    # sess.run(project_step, feed_dict={x: img0, epsilon: demo_epsilon})
                    adv1 = x_hat.eval()  # retrieve the adversarial example
                    tmp = adv1 - imgsave
                    tmnp = np.array(tmp)
                    fanshu = sum(sum(sum(abs(tmnp))))
                    gt= gt*u+tmnp/fanshu
                    img=imgsave.copy()
                    #img=img+tmnp

                    gt_abs=-abs(gt)
                    flat_indices = np.argpartition(gt_abs.ravel(), n - 1)[:n]
                    row_indices, col_indices, deep = np.unravel_index(flat_indices, gt_abs.shape)
                    img = imgsave.copy()

                    for j in range(n):
                        mstep = adv1[row_indices[j], col_indices[j], deep[j]] - imgsave[row_indices[j], col_indices[j], deep[j]]
                        if mstep > 0:
                            img[row_indices[j], col_indices[j], deep[j]] = img[row_indices[j], col_indices[j], deep[
                                j]] + ministep
                        elif mstep < 0:
                            img[row_indices[j], col_indices[j], deep[j]] = img[row_indices[j], col_indices[j], deep[
                                j]] - ministep
                        else:
                            img[row_indices[j], col_indices[j], deep[j]] = img[row_indices[j], col_indices[j], deep[j]]

                    img=np.clip(np.clip(img,below2,above2),0,1)

                    imgsave=img.copy()

                        # if prob_target > 0.5:
                        #     print('step %d, loss=%g' % (i + 1, loss_value))
                        #     break

                #adv = x_hat.eval()  # retrieve the adversarial example
                time_end=time.time()
                p = sess.run(probs, feed_dict={image: img})[0]
                prob_target = p[demo_target]
                p2=sess.run(probs,feed_dict={image: img3})[0]
                prob_origin=p2[img_class]
                prob_origin_adv=p[img_class]
                topk = list(p.argsort()[-10:][::-1])
                topk2 = list(p2.argsort()[-10:][::-1])
                count+=1
                print(topk,img_class,'Picture_num:',count,'Perturbation:',perturbation,'timecost:',time_end-time_start)
                print('正确样本预测概率',prob_origin,'目标样本预测率',prob_target,'攻击之后样本',prob_origin_adv)
                prob_origin_save.append(prob_origin)
                prob_target_save.append(prob_target)
                tf.get_default_graph()

                # if count>100:
                #     break

                if topk2[0] != img_class:
                    unsuccess += 1
                    print('原图识别失败，没有攻击意义')
                elif topk[0] == demo_target:
                    success += 1
                    print('hack successfully')
                else:
                    print('unsuccessfully')
                print('成功次数', success)
                suc_rate=success/(count-unsuccess)
                print('当前成功率:',suc_rate)
                print('当前目标类平均置信度:',np.mean(prob_origin_save))

                # classify(img, correct_class=img_class, target_class=demo_target)
                #
                # img2 = PIL.Image.open(savepath)
                # r,g,b,a=img2.split()
                # img2=Image.merge("RGB",(r,g,b))
                # big_dim = max(img2.width, img2.height)
                # wide = img2.width > img2.height
                # new_w = 299 if not wide else int(img2.width * 299 / img2.height)
                # new_h = 299 if wide else int(img2.height * 299 / img2.width)
                # img2 = img2.resize((new_w, new_h)).crop((0, 0, 299, 299))
                # img2 = (np.asarray(img2) / 255.0).astype(np.float32)
                # classify(img2, correct_class=img_class, target_class=demo_target)
                # error=adv-img2
                # img2=img2*255
                # adv=adv*255
                # print(img2[4,:,2])
                # print(adv[4,:,2])

            success_rate=success/964
            print("扰动为" + str(perturbation) + '时，成功率为' + str(success_rate))
            success_sta.append(success_rate)

        print('成功率:',success_sta)


        u_str=str(u)
        ratio_str=str(ratio)
        f=open('./success_rate_PDSM_target_Incv3_'+u_str+'_'+ratio_str,'wb')
        pickle.dump(success_sta,f)
        f.close()
        f=open('./prob_origin_save_PDSM_target_Incv3_'+u_str+'_'+ratio_str,'wb')
        pickle.dump(prob_origin_save,f)
        f.close()
        f=open('./prob_target_save_PDSM_target_Incv3_'+u_str+'_'+ratio_str,'wb')
        pickle.dump(prob_target_save,f)
        f.close()
