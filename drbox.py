import os
import os.path
import sys
import random
from tkinter import CURRENT
import numpy as np
from glob import glob
import tensorflow as tf
from model import *
from rbox_functions import *
import scipy.misc
import pickle
import imageio
from PIL import Image
import cv2 as cv
from datetime import datetime
import tempfile

ARUCO_DICT = {
    "DICT_4X4_50": cv.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv.aruco.DICT_APRILTAG_36h11
}

TXT_DIR = './data' 
INPUT_DATA_PATH = TXT_DIR + '/train'
TEST_DATA_PATH = TXT_DIR + '/test'
PRETRAINED_NET_PATH = "./vgg16.npy"
SAVE_PATH = './result' 
TRAIN_BATCH_SIZE = 16
IM_HEIGHT = 300
IM_WIDTH = 300
IM_CDIM = 3
FEA_HEIGHT4 = 38
FEA_WIDTH4 = 38
FEA_HEIGHT3 = 75
FEA_WIDTH3 = 75
STEPSIZE4 = 8
STEPSIZE3 = 4

# PRIOR_ANGLES = [0, 30, 60, 90, 120, 150] # Percobaan 1
PRIOR_ANGLES = [0, 20, 40, 60, 80, 100, 120, 140, 160] # Percobaan 2
# PRIOR_ANGLES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170] # Percobaan 3

#PRIOR_HEIGHTS =[[4.0, 7.0, 10.0, 13.0],[3.0,8.0,12.0,17.0,23.0]] #[3.0,8.0,12.0,17.0,23.0] #
#PRIOR_WIDTHS = [[15.0, 25.0, 35.0, 45.0],[20.0,35.0,50.0,80.0,100.0]]#[20.0,35.0,50.0,80.0,100.0]  

# Percobaan 1
#PRIOR_HEIGHTS = [[10.0, 15.0, 75.0, 80.0],[10.0, 15.0, 75.0, 80.0]]
#PRIOR_WIDTHS = [[10.0, 15.0, 75.0, 80.0],[10.0, 15.0, 75.0, 80.0]]
PRIOR_HEIGHTS = [[15, 90], [15, 90]]
PRIOR_WIDTHS = [[15, 90], [15, 90]]
#######

ITERATION_NUM = 50000 #10000 #10000 #10000 #10000 #10000 #10000 #10000 #10000 #10000 #50000 
OVERLAP_THRESHOLD = 0.5
IS180 = True
NP_RATIO = 3
LOC_WEIGHTS = [0.1, 0.1, 0.2, 0.2, 0.1]

# Ini untuk loading parameter prior rbox yang sudah digenerate sebelumnya
LOAD_PREVIOUS_POS = True

WEIGHT_DECAY = 0.0005
DISPLAY_INTERVAL = 100 #100
SAVE_MODEL_INTERVAL = 2000
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # select the used GPU
TEST_BATCH_SIZE = 1
TEST_RESOLUTION_IN = 3
TEST_RESOLUTION_OUT = [3]

# Ini yang divariasikan untuk score threshold
TEST_SCORE_THRESHOLD = 0.1

TEST_NMS_THRESHOLD = 0.1
TEST_HEIGHT_STEP = 0.85
TEST_WIDTH_STEP = 0.85
flags = tf.app.flags
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
FLAGS = flags.FLAGS

USE_THIRD_LAYER = 1
FPN_NET = 1
USE_FOCAL_LOSS = 1
focal_loss_factor = 2.5

class DrBoxNet():
    def __init__(self):                
        for stage in ['train', 'test']:
            self.get_im_list(stage)
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.global_step = tf.Variable(0, trainable=False)        
        self.model_save_path = os.path.join(SAVE_PATH, 'model')
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        self.get_encoded_positive_box()
        random.shuffle(self.train_im_list)
        self.train_list_idx = 0        
        self.input_im = tf.placeholder(tf.float32, shape=[None, IM_HEIGHT, IM_WIDTH, IM_CDIM])
        #self.input_idx = tf.placeholder(tf.int32, shape=[None])
        self.prior_num = [len(PRIOR_ANGLES)*len(PRIOR_WIDTHS[0]), len(PRIOR_ANGLES)*len(PRIOR_WIDTHS[1])]
        self.total_prior_num = FEA_HEIGHT4*FEA_WIDTH4*self.prior_num[1]+FEA_HEIGHT3*FEA_WIDTH3*self.prior_num[0]*USE_THIRD_LAYER        
        print("Total Prior Num : ", self.total_prior_num)
        with open("training_parameters.txt", "a") as files:
            files.writelines("Training Parameter")
            files.writelines('\n')
            files.writelines("Total prior rboxes: " + str(self.total_prior_num))
            files.writelines('\n')
        self.para_num = 5
        self.cls_num = 1
        self.batch_pos_box = tf.placeholder(tf.float32, shape=[None, self.para_num])
        self.batch_pos_idx = tf.placeholder(tf.int32, shape=[None])
        self.batch_pos_ind = tf.placeholder(tf.float32, shape=[None])
        #self.batch_pos_num = tf.placeholder(tf.int32, shape=[None])
        self.batch_neg_mask = tf.placeholder(tf.float32, shape=[None])
        self.pos_label = tf.placeholder(tf.float32, shape=[None, self.cls_num + 1])
        self.neg_label = tf.placeholder(tf.float32, shape=[None, self.cls_num + 1])
        if FLAGS.train:
            self.detector = VGG16(self.prior_num, self.para_num, self.cls_num, FPN_NET, USE_THIRD_LAYER, TRAIN_BATCH_SIZE)
        else:
            self.detector = VGG16(self.prior_num, self.para_num, self.cls_num, FPN_NET, USE_THIRD_LAYER, TEST_BATCH_SIZE)
        self.loc, self.conf = self.detector(self.input_im)
        self.conf_softmax = tf.nn.softmax(self.conf)
        self.hard_negative_mining()
        self.compute_conf_loss()
        self.compute_loc_loss()        
        self.reg_loss = tf.add_n(self.detector.regular_loss(WEIGHT_DECAY))
        self.loss = self.loc_loss + self.conf_loss #+ self.reg_loss
        self.saver = tf.train.Saver(max_to_keep=5)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            
    def compute_conf_loss(self):
        pos_tensor = tf.gather(self.conf, self.batch_pos_idx)
        neg_tensor = tf.gather(self.conf, self.batch_neg_idx)
        self.pos_tensor = pos_tensor
        self.neg_tensor = neg_tensor
        if USE_FOCAL_LOSS:
            pos_prob = tf.slice(tf.nn.softmax(pos_tensor),[0,1],[-1,1])
            neg_prob = tf.slice(tf.nn.softmax(neg_tensor),[0,0],[-1,1])
            self.conf_pos_losses = tf.nn.softmax_cross_entropy_with_logits(logits=pos_tensor, labels=self.pos_label)
            self.conf_neg_losses = tf.nn.softmax_cross_entropy_with_logits(logits=neg_tensor, labels=self.neg_label)
            self.conf_pos_loss = tf.reduce_mean(tf.multiply((1-pos_prob)**focal_loss_factor, self.conf_pos_losses))
            self.conf_neg_loss = tf.reduce_mean(tf.multiply((1-neg_prob)**focal_loss_factor, self.conf_neg_losses))
        else:
            self.conf_pos_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pos_tensor, labels=self.pos_label))
            self.conf_neg_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=neg_tensor, labels=self.neg_label) * self.batch_neg_mask)
        self.conf_loss = self.conf_pos_loss + self.conf_neg_loss
    
    def compute_loc_loss(self):
        loc_tensor = tf.gather(self.loc, self.batch_pos_idx)
        self.loc_tensor = loc_tensor
        loc_diff = tf.add(loc_tensor, -1*self.batch_pos_box)
        loc_diff = tf.abs(loc_diff)
        loc_l1_smooth = tf.where(tf.greater(loc_diff, 1.0), loc_diff - 0.5, tf.square(loc_diff) * 0.5)
        self.loc_loss = tf.reduce_mean(loc_l1_smooth)
    
    def hard_negative_mining(self):
        conf = self.conf_softmax
        conf = tf.transpose(conf)
        conf = tf.slice(conf, [0, 0], [1, self.total_prior_num*TRAIN_BATCH_SIZE])
        conf = tf.squeeze(conf)
        conf = -1*tf.add(conf, self.batch_pos_ind)
        for batch_idx in range(TRAIN_BATCH_SIZE):
            batch_slice = tf.slice(conf, [batch_idx*self.total_prior_num], [self.total_prior_num])
            neg_top_k = tf.nn.top_k(batch_slice, self.max_neg_num)
            neg_idx = neg_top_k.indices + batch_idx*self.total_prior_num
            neg_idx = tf.squeeze(neg_idx)
            if batch_idx == 0:
                self.batch_neg_idx = neg_idx
            else:
                self.batch_neg_idx = tf.concat([self.batch_neg_idx, neg_idx], 0)
    
    def get_im_list(self, stage):        
        if stage == 'train': 
            infile = open(os.path.join(TXT_DIR, 'train.txt'))
            self.train_im_list = []
            k = 0
            for line in infile:
                line = line.strip()
                line = str(k) + ' ' + line
                self.train_im_list.append(line)
                k += 1
                if k == 5120:
                    break                        
            infile.close()
            self.train_im_num = len(self.train_im_list)
        else:
            infile = open(os.path.join(TXT_DIR, 'test.txt'))
            self.test_im_list = []
            for line in infile:
                self.test_im_list.append(line)            
            infile.close()
            self.test_im_num = len(self.test_im_list)

    def get_encoded_positive_box(self):        
        prior_box4 = PriorRBox(IM_HEIGHT, IM_WIDTH, FEA_HEIGHT4, FEA_WIDTH4, STEPSIZE4, PRIOR_ANGLES, PRIOR_HEIGHTS[1], PRIOR_WIDTHS[1])
        prior_box3 = PriorRBox(IM_HEIGHT, IM_WIDTH, FEA_HEIGHT3, FEA_WIDTH3, STEPSIZE3, PRIOR_ANGLES, PRIOR_HEIGHTS[0], PRIOR_WIDTHS[0])
        if USE_THIRD_LAYER:            
            prior_box = np.concatenate((prior_box3, prior_box4), axis=0)
        else:
            prior_box = prior_box4            
        self.prior_box = prior_box
        self.ind_one_hot = {}
        self.positive_indice = {}
        self.encodedbox = {}
        self.pos_num = {}        
        self.max_neg_num = 0
        if not FLAGS.train:
            return
        if LOAD_PREVIOUS_POS:
            with open(os.path.join(INPUT_DATA_PATH, 'ind_one_hot.pkl'),'rb') as fid:
                self.ind_one_hot = pickle.load(fid)
            with open(os.path.join(INPUT_DATA_PATH, 'positive_indice.pkl'),'rb') as fid:
                self.positive_indice = pickle.load(fid)
            with open(os.path.join(INPUT_DATA_PATH, 'encodedbox.pkl'),'rb') as fid:
                self.encodedbox = pickle.load(fid)

        time_start = datetime.now()
        with open("training_parameters.txt","a") as files:
            files.writelines("Time start :" + time_start.strftime("%H:%M:%S"))
            files.writelines('\n')

        for k in range(self.train_im_num):
            if k % 100 == 0:
                print('Preprocessing {}'.format(k))
                print('k', k)
            im_rbox_info = self.train_im_list[k]
            im_rbox_info = im_rbox_info.split(' ')
            idx = eval(im_rbox_info[0])
            rbox_fn = im_rbox_info[2]
            rbox_path = os.path.join(INPUT_DATA_PATH, rbox_fn)
            rboxes = []
            rboxes = np.array(rboxes)
            i = 0
            with open(rbox_path, 'r') as infile:
                for line in infile:
                    rbox = []
                    ii = 0
                    for rbox_param in line.split(' '):
                        if ii == 0 or ii == 2: # center x or width
                            rbox.append(eval(rbox_param)/IM_WIDTH)
                        elif ii == 1 or ii == 3: # center y or height
                            rbox.append(eval(rbox_param)/IM_HEIGHT)
                        elif ii == 5:
                            rbox.append(eval(rbox_param))
                        ii += 1
                    rbox = np.array(rbox)
                    rbox = rbox[np.newaxis, :]
                    if i == 0:
                        gt_box = rbox
                    else:
                        gt_box = np.concatenate((gt_box, rbox), axis=0)
                    i += 1                                                        
            if not LOAD_PREVIOUS_POS:
                self.ind_one_hot[idx], self.positive_indice[idx], self.encodedbox[idx] = MatchRBox(prior_box, gt_box, OVERLAP_THRESHOLD, IS180)
                self.encodedbox[idx] /= LOC_WEIGHTS
            self.pos_num[idx] = len(self.positive_indice[idx])
            if self.max_neg_num < self.pos_num[idx]:
                self.max_neg_num = self.pos_num[idx]
        self.max_neg_num *= NP_RATIO
        if not LOAD_PREVIOUS_POS: 
            with open(os.path.join(INPUT_DATA_PATH, 'ind_one_hot.pkl'),'wb') as fid:
                pickle.dump(self.ind_one_hot, fid)
            with open(os.path.join(INPUT_DATA_PATH, 'positive_indice.pkl'),'wb') as fid:
                pickle.dump(self.positive_indice, fid)
            with open(os.path.join(INPUT_DATA_PATH, 'encodedbox.pkl'),'wb') as fid:
                pickle.dump(self.encodedbox, fid)

        time_end = datetime.now()
        with open("training_parameters.txt","a") as files:
            files.writelines("Time end :" + time_end.strftime("%H:%M:%S"))
            files.writelines('\n')
            files.writelines("Elapsed time :" + str(time_end - time_start))
            files.writelines('\n')

    def get_next_batch_list(self):    
        idx = self.train_list_idx        
        if idx + TRAIN_BATCH_SIZE > self.train_im_num:
            batch_list = np.arange(idx, self.train_im_num)
            # shuffle the data in one category
            random.shuffle(self.train_im_list)
            new_list = np.arange(0, TRAIN_BATCH_SIZE-(self.train_im_num-idx))
            batch_list = np.concatenate((batch_list, new_list))
            self.train_list_idx = TRAIN_BATCH_SIZE-(self.train_im_num-idx)
        else:
            batch_list = np.arange(idx, idx+TRAIN_BATCH_SIZE)
            self.train_list_idx = idx+TRAIN_BATCH_SIZE
        return batch_list

    def train(self):
        #train_step = tf.train.MomentumOptimizer(self.learning_rate, self.momentum).minimize(self.loss, global_step=self.global_step)
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
        self.sess.run(tf.global_variables_initializer())
        
        # load the model if there is one
        could_load, checkpoint_counter = self.load()
        if could_load:
            self.sess.run(self.global_step.assign(checkpoint_counter))
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load the pretrained network FINISHED")
        
        for iter_num in range(ITERATION_NUM+1):
            # print("Iter num : ", iter_num)
            input_im = np.zeros((TRAIN_BATCH_SIZE, IM_HEIGHT, IM_WIDTH, IM_CDIM))
            input_im = input_im.astype('float32')
            batch_list = self.get_next_batch_list()
            batch_pos_box = []
            batch_pos_box = np.array(batch_pos_box)
            batch_pos_ind = []
            batch_pos_ind = np.array(batch_pos_ind)
            batch_pos_idx = []
            batch_pos_idx = np.array(batch_pos_idx)
            batch_pos_num = []
            batch_pos_num = np.array(batch_pos_num)
            batch_neg_mask = np.zeros(TRAIN_BATCH_SIZE*self.max_neg_num)
            k = 0            
            for batch_idx in batch_list:
                im_rbox_info = self.train_im_list[batch_idx]
                im_rbox_info = im_rbox_info.split(' ')
                real_idx = eval(im_rbox_info[0])
                #input_idx[k] = real_idx
                im = imageio.imread(os.path.join(INPUT_DATA_PATH, im_rbox_info[1]))
                imm = np.zeros((IM_HEIGHT, IM_WIDTH, IM_CDIM))
                if len(im.shape) == 2:
                    for ij in range(IM_CDIM):
                        imm[:,:,ij] = im
                    im = imm
                input_im[k] = im.reshape(IM_HEIGHT, IM_WIDTH, IM_CDIM).astype('float32')
                # select all or part of regression parameters in furture (to be done)
                if k==0:
                    batch_pos_box = self.encodedbox[real_idx]
                    batch_pos_ind = self.ind_one_hot[real_idx]
                    batch_pos_idx = self.positive_indice[real_idx]
                    batch_pos_num = [self.pos_num[real_idx]]                
                else:
                    batch_pos_box = np.concatenate((batch_pos_box, self.encodedbox[real_idx]), axis=0)
                    batch_pos_ind = np.concatenate((batch_pos_ind, self.ind_one_hot[real_idx]), axis=0)
                    batch_pos_idx = np.concatenate((batch_pos_idx, self.positive_indice[real_idx]+k*self.total_prior_num), axis=0)                    
                    batch_pos_num = np.concatenate((batch_pos_num, [self.pos_num[real_idx]]), axis=0)
                batch_neg_mask[k*self.max_neg_num:k*self.max_neg_num+self.pos_num[real_idx]*NP_RATIO] = 1.0
                #self.batch_pos_num[k] = self.pos_num[real_idx]
                #self.batch_neg_num[k] = self.batch_pos_num[k] * NP_RATIO
                k += 1
            batch_pos_ind = batch_pos_ind.astype('float32')
            total_batch_pos_num = np.sum(batch_pos_num)
            #total_batch_neg_num = total_batch_pos_num * NP_RATIO
            total_batch_neg_num = TRAIN_BATCH_SIZE * self.max_neg_num
            total_batch_pos_num = total_batch_pos_num.astype('int32')
            #total_batch_neg_num = total_batch_neg_num.astype('int32')
            batch_neg_mask *= (1.0 * total_batch_neg_num / total_batch_pos_num)
            #print('total_batch_neg_num {}, total_batch_pos_num {}'.format(total_batch_neg_num, total_batch_pos_num))
            #batch_neg_mask *= 1
            pos_label = np.zeros((total_batch_pos_num, 2))
            pos_label[:,1] = 1
            neg_label = np.zeros((total_batch_neg_num, 2))
            neg_label[:,0] = 1
            
            counter = self.sess.run(self.global_step)
            if counter > 80000:
                self.learning_rate = 0.0001
            if counter > 100000:
                self.learning_rate = 0.00001
            if counter > 120000:
                self.learning_rate = 0.000001
            self.sess.run(train_step, feed_dict={self.input_im:input_im, self.batch_pos_box:batch_pos_box, self.batch_pos_ind:batch_pos_ind,
                        self.batch_pos_idx:batch_pos_idx, self.batch_neg_mask:batch_neg_mask, self.pos_label:pos_label, self.neg_label:neg_label})
            if counter % DISPLAY_INTERVAL == 0:
                loss, loc_loss, conf_loss, conf_pos_loss, conf_neg_loss, reg_loss = self.sess.run([
                            self.loss, self.loc_loss, self.conf_loss, self.conf_pos_loss, self.conf_neg_loss, self.reg_loss],
                            feed_dict={self.input_im:input_im,
                            self.batch_pos_box:batch_pos_box, self.batch_pos_ind:batch_pos_ind, self.batch_pos_idx:batch_pos_idx, self.batch_neg_mask:batch_neg_mask,
                            self.pos_label:pos_label, self.neg_label:neg_label})
                print("Loss", loss)
                print("counter:[" + str(counter) + "], loss:" + str(loss) + ", loc_loss:" + str(loc_loss) + ", conf_loss:" + str(conf_loss) + ", conf_pos_loss:" + str(conf_pos_loss) + ", conf_neg_loss:" + str(conf_neg_loss) + ", reg_loss:" + str(reg_loss))
                with open('training_loss.txt', 'a') as files:
                    loss_str = str(counter) + " " + str(loss) + " " + str(loc_loss) + " " + str(conf_loss) + " " + str(conf_pos_loss) + " " + str(conf_neg_loss) + " " + str(reg_loss)
                    files.write(loss_str)
                    files.write('\n')

            if counter % SAVE_MODEL_INTERVAL == 0:
                self.save(counter)

    def test(self):
        # load the trained model
        could_load, checkpoint_counter = self.load()
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        label = 1
        # Start time measurement
        time_start = datetime.now()
        for test_info in self.test_im_list:
            test_im_rbox_info = test_info.split(' ')
            test_im_path = os.path.join(TEST_DATA_PATH, test_im_rbox_info[0])
            test_rbox_gt_path = os.path.join(TEST_DATA_PATH, test_im_rbox_info[0]+'.rbox')
            test_result_path = TXT_DIR + '/' + os.path.basename(SAVE_PATH)
            if not os.path.exists(test_result_path):
                os.makedirs(test_result_path)
            test_rbox_output_path = os.path.join(test_result_path, os.path.basename(test_rbox_gt_path).replace('.jpg','') + '.score')
            test_im = imageio.imread(test_im_path)
            if 'L2' in test_im_path:
                not_zero   = np.where(test_im != 0)
                is_zero    = np.where(test_im == 0)
                mean_value = np.sum(test_im[not_zero])/len(not_zero[0])
                for temp_idx in range(len(is_zero[0])):
                    test_im[is_zero[0][temp_idx], is_zero[1][temp_idx]] = mean_value   
            temp = np.zeros((test_im.shape[0], test_im.shape[1], IM_CDIM))
            for chid in range(IM_CDIM):
                temp[:,:,chid] = test_im[:,:,chid]
            test_im = temp
            [height, width, _] = test_im.shape
            print('Start detection' + test_im_path)
            count = 0
            islast = 0
            inputdata = np.zeros((TEST_BATCH_SIZE, IM_HEIGHT, IM_WIDTH, IM_CDIM))
            inputdata = inputdata.astype('float32')
            inputloc = np.zeros((TEST_BATCH_SIZE, IM_CDIM))
            rboxlist = []
            scorelist = []
            #start = time.time()
            for i in range(len(TEST_RESOLUTION_OUT)):                                
                xBegin, yBegin = 0, 0
                width_i = int(round(width * TEST_RESOLUTION_IN / TEST_RESOLUTION_OUT[i]))
                height_i = int(round(height * TEST_RESOLUTION_IN / TEST_RESOLUTION_OUT[i]))
                image_i = cv.resize(test_im, (width_i, height_i), cv.INTER_AREA)
                while 1:
                    if islast == 0:                        
                        width_S = IM_WIDTH * TEST_RESOLUTION_OUT[i] / TEST_RESOLUTION_IN #int(round(IM_WIDTH * TEST_RESOLUTION_OUT[i] / TEST_RESOLUTION_IN))
                        height_S = IM_HEIGHT * TEST_RESOLUTION_OUT[i] / TEST_RESOLUTION_IN #int(round(IM_HEIGHT * TEST_RESOLUTION_OUT[i] / TEST_RESOLUTION_IN))
                        xEnd = xBegin + width_S
                        yEnd = yBegin + height_S
                        xEnd = min(xEnd, width)
                        yEnd = min(yEnd, height)
                        xBeginHat = int(round(xBegin * TEST_RESOLUTION_IN / TEST_RESOLUTION_OUT[i]))
                        yBeginHat = int(round(yBegin * TEST_RESOLUTION_IN / TEST_RESOLUTION_OUT[i]))
                        xEndHat = int(round(xEnd * TEST_RESOLUTION_IN / TEST_RESOLUTION_OUT[i]))
                        yEndHat = int(round(yEnd * TEST_RESOLUTION_IN / TEST_RESOLUTION_OUT[i]))
                        
                        subimage = np.zeros((IM_HEIGHT, IM_WIDTH, IM_CDIM))
                        subimage[0:yEndHat-yBeginHat, 0:xEndHat-xBeginHat, 0:3] = image_i[yBeginHat:yEndHat, xBeginHat:xEndHat, 0:3]
                        inputdata[count] = subimage.astype('float32')
                        inputloc[count] = [xBegin,yBegin,TEST_RESOLUTION_OUT[i]/TEST_RESOLUTION_IN]
                        count = count + 1
                    if count == TEST_BATCH_SIZE or islast == 1:
                        loc_preds, conf_preds = self.sess.run([self.loc, self.conf_softmax], feed_dict={self.input_im:inputdata})
                        for j in range(TEST_BATCH_SIZE):
                            conf_preds_j = conf_preds[j*self.total_prior_num:(j+1)*self.total_prior_num, 1]
                            loc_preds_j  = loc_preds[j*self.total_prior_num:(j+1)*self.total_prior_num, :]
                            index = np.where(conf_preds_j > TEST_SCORE_THRESHOLD)[0]
                            conf_preds_j  = conf_preds_j[index]
                            loc_preds_j   = loc_preds_j[index]
                            loc_preds_j   = loc_preds_j.reshape(loc_preds_j.shape[0]*self.para_num)
                            prior_boxes_j = self.prior_box[index].reshape(len(index) * self.para_num)
                            inputloc_j = inputloc[j]
                            if len(loc_preds_j) > 0:
                                rbox, score = DecodeNMS(loc_preds_j, prior_boxes_j, conf_preds_j, inputloc_j, index, TEST_NMS_THRESHOLD, IM_HEIGHT, IM_WIDTH)
                                rboxlist.extend(rbox)
                                scorelist.extend(score)
                        count = 0
                    if islast == 1:
                        break
                    xBegin = xBegin + int(round(TEST_WIDTH_STEP * width_S))
                    if  xEnd >= width: #xBegin
                        if yEnd >= height:
                            islast = 0
                            break
                        xBegin = 0
                        yBegin = yBegin + int(round(TEST_HEIGHT_STEP * height_S))
                        if yBegin >= height:
                            if i == len(TEST_RESOLUTION_OUT) - 1:
                                islast = 1
                            else:
                                break
            nms_out = NMSOutput(rboxlist, scorelist, TEST_NMS_THRESHOLD, label, test_rbox_output_path)
            self.visualize_output(test_im_path, nms_out)
        time_end = datetime.now()
        elapsed_time = (time_end - time_start)
        print("Computation time for 100 images : ", str(elapsed_time))
        print("Computation time for 1 images : ", str(elapsed_time/100.0))
    
    def visualize_output(self, filename, nms_out, score_threshold=0.5):
        image = cv.imread(filename)
        for i in range(len(nms_out)):
            # x, y, w, h, label, angle, score = nms_out[i]
            x = nms_out[i][0]
            y = nms_out[i][1]
            w = nms_out[i][2]
            h = nms_out[i][3]
            label = nms_out[i][4]
            angle = nms_out[i][5]
            score = nms_out[i][6]
            color = (0,0,255)
            # cv.circle(image, (int(x),int(y)), 2, color, -1)
            if score >= score_threshold:
                tl_x = int(x - (w//2))
                tl_y = int(y - (h//2))
                bl_x = int(x - (w//2))
                bl_y = int(y + (h//2))

                tr_x = int(x + (w//2))
                tr_y = int(y - (h//2))
                br_x = int(x + (w//2))
                br_y = int(y + (h//2))

                R = cv.getRotationMatrix2D(center=(x,y), angle=-angle, scale=1)
                tl = np.array([[tl_x], [tl_y], [1]])
                bl = np.array([[bl_x], [bl_y], [1]])
                tr = np.array([[tr_x], [tr_y], [1]])
                br = np.array([[br_x], [br_y], [1]])

                r_tl = R.dot(tl)
                r_bl = R.dot(bl)
                r_tr = R.dot(tr)
                r_br = R.dot(br)

                points = np.array([[r_tl[0], r_tl[1]],
                                    [r_tr[0], r_tr[1]],
                                    [r_br[0], r_br[1]],
                                    [r_bl[0], r_bl[1]],], dtype=np.int32)
                
                points = points.reshape((- 1 , 1 , 2 ))
                cv.circle(image, (int(x), int(y)), 2, color, -1)
                cv.polylines(image, [points], True, color, 2)
        # Save result to folder ./data/result/result_xxxxxx.jpg
        result_filename = filename.replace("test", "result")
        print("Result filename :", result_filename)
        cv.imwrite(result_filename, image)
    
    def test_from_depth(self):
        # Start time measurement
        time_start = datetime.now()

        # Load from pickle
        CURRENT_PATH = os.getcwd()
        with open(CURRENT_PATH + '/d455_data/list_color_frame.pickle.5a', 'rb') as f:
            list_color_frame = pickle.load(f)
        with open(CURRENT_PATH + '/d455_data/list_depth_frame.pickle.5a', 'rb') as f:
            list_depth_frame = pickle.load(f)
        
        # Load the trained model
        could_load, checkpoint_counter = self.load()
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        label = 1
        
        arucoDict = cv.aruco.Dictionary_get(ARUCO_DICT['DICT_ARUCO_ORIGINAL'])
        arucoParams = cv.aruco.DetectorParameters_create()

        for image_number in range(len(list_color_frame)):
            print("Test image number : ", image_number)
            image_from_list = list_color_frame[image_number]
            cv_color_img = image_from_list.copy()
            cv_depth_img = list_depth_frame[image_number]

            # detect ArUco markers in the input frame
            (corners, ids, rejected) = cv.aruco.detectMarkers(cv_color_img, arucoDict, parameters=arucoParams)
            if len(corners) == 4:
                # flatten the ArUco IDs list
                ids = ids.flatten()
                # loop over the detected ArUCo corners
                list_marker_points = [[0,0],[0,0],[0,0],[0,0],[0,0]] # order dalam list 3, 2, 0, 1
                for (markerCorner, markerID) in zip(corners, ids):
                    # extract the marker corners (which are always returned
                    # in top-left, top-right, bottom-right, and bottom-left order
                    corners = markerCorner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corners
                    # convert each of the (x, y)-coordinate pairs to integers
                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))	
                    # draw the bounding box of the ArUCo detection
                    cv.line(cv_color_img, topLeft, topRight, (0, 255, 0), 2)
                    cv.line(cv_color_img, topRight, bottomRight, (0, 255, 0), 2)
                    cv.line(cv_color_img, bottomRight, bottomLeft, (0, 255, 0), 2)
                    cv.line(cv_color_img, bottomLeft, topLeft, (0, 255, 0), 2)
                    # compute and draw the center (x, y)-coordinates of the
                    # ArUco marker
                    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                    cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                    
                    if markerID < len(list_marker_points):
                        list_marker_points[markerID]=[cX,cY]
                                       
                    cv.circle(cv_color_img, (cX, cY), 4, (0, 0, 255), -1)
                    # draw the ArUco marker ID on the frame
                    cv.putText(cv_color_img, str(markerID), (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                initial_points = np.float32([list_marker_points[0],list_marker_points[1],list_marker_points[2],list_marker_points[3]])
                target_points = np.float32([[0,0], [300,0], [0,300], [300,300]])

                M = cv.getPerspectiveTransform(initial_points, target_points)
                cv_test_img = cv.warpPerspective(image_from_list.copy(), M, (300,300))
                cv.imwrite("./test_image.jpg", cv_test_img)
            
            # io_test_img = imageio.imread("./test_image.jpg")
            io_test_img = cv.cvtColor(cv.imread("./test_image.jpg"), cv.COLOR_BGR2RGB)
            temp = np.zeros((io_test_img.shape[0], io_test_img.shape[1], IM_CDIM))
            for chid in range(IM_CDIM):
                temp[:,:,chid] = io_test_img[:,:,chid]
            io_test_img = temp

            [height, width, _] = io_test_img.shape
            count = 0
            islast = 0
            inputdata = np.zeros((TEST_BATCH_SIZE, IM_HEIGHT, IM_WIDTH, IM_CDIM))
            inputdata = inputdata.astype('float32')
            inputloc = np.zeros((TEST_BATCH_SIZE, IM_CDIM))
            rboxlist = []
            scorelist = []
            for i in range(len(TEST_RESOLUTION_OUT)):                                
                xBegin, yBegin = 0, 0
                width_i = int(round(width * TEST_RESOLUTION_IN / TEST_RESOLUTION_OUT[i]))
                height_i = int(round(height * TEST_RESOLUTION_IN / TEST_RESOLUTION_OUT[i]))
                image_i = cv.resize(io_test_img, (width_i, height_i), cv.INTER_AREA)
                while 1:
                    if islast == 0:                        
                        width_S = IM_WIDTH * TEST_RESOLUTION_OUT[i] / TEST_RESOLUTION_IN #int(round(IM_WIDTH * TEST_RESOLUTION_OUT[i] / TEST_RESOLUTION_IN))
                        height_S = IM_HEIGHT * TEST_RESOLUTION_OUT[i] / TEST_RESOLUTION_IN #int(round(IM_HEIGHT * TEST_RESOLUTION_OUT[i] / TEST_RESOLUTION_IN))
                        xEnd = xBegin + width_S
                        yEnd = yBegin + height_S
                        xEnd = min(xEnd, width)
                        yEnd = min(yEnd, height)
                        xBeginHat = int(round(xBegin * TEST_RESOLUTION_IN / TEST_RESOLUTION_OUT[i]))
                        yBeginHat = int(round(yBegin * TEST_RESOLUTION_IN / TEST_RESOLUTION_OUT[i]))
                        xEndHat = int(round(xEnd * TEST_RESOLUTION_IN / TEST_RESOLUTION_OUT[i]))
                        yEndHat = int(round(yEnd * TEST_RESOLUTION_IN / TEST_RESOLUTION_OUT[i]))
                        
                        subimage = np.zeros((IM_HEIGHT, IM_WIDTH, IM_CDIM))
                        subimage[0:yEndHat-yBeginHat, 0:xEndHat-xBeginHat, 0:3] = image_i[yBeginHat:yEndHat, xBeginHat:xEndHat, 0:3]
                        inputdata[count] = subimage.astype('float32')
                        inputloc[count] = [xBegin,yBegin,TEST_RESOLUTION_OUT[i]/TEST_RESOLUTION_IN]
                        count = count + 1
                    if count == TEST_BATCH_SIZE or islast == 1:
                        loc_preds, conf_preds = self.sess.run([self.loc, self.conf_softmax], feed_dict={self.input_im:inputdata})
                        for j in range(TEST_BATCH_SIZE):
                            conf_preds_j = conf_preds[j*self.total_prior_num:(j+1)*self.total_prior_num, 1]
                            loc_preds_j  = loc_preds[j*self.total_prior_num:(j+1)*self.total_prior_num, :]
                            index = np.where(conf_preds_j > TEST_SCORE_THRESHOLD)[0]
                            conf_preds_j  = conf_preds_j[index]
                            loc_preds_j   = loc_preds_j[index]
                            loc_preds_j   = loc_preds_j.reshape(loc_preds_j.shape[0]*self.para_num)
                            prior_boxes_j = self.prior_box[index].reshape(len(index) * self.para_num)
                            inputloc_j = inputloc[j]
                            if len(loc_preds_j) > 0:
                                rbox, score = DecodeNMS(loc_preds_j, prior_boxes_j, conf_preds_j, inputloc_j, index, TEST_NMS_THRESHOLD, IM_HEIGHT, IM_WIDTH)
                                rboxlist.extend(rbox)
                                scorelist.extend(score)
                        count = 0
                    if islast == 1:
                        break
                    xBegin = xBegin + int(round(TEST_WIDTH_STEP * width_S))
                    if  xEnd >= width: #xBegin
                        if yEnd >= height:
                            islast = 0
                            break
                        xBegin = 0
                        yBegin = yBegin + int(round(TEST_HEIGHT_STEP * height_S))
                        if yBegin >= height:
                            if i == len(TEST_RESOLUTION_OUT) - 1:
                                islast = 1
                            else:
                                break
            
            nms_out = NMSOutput(rboxlist, scorelist, TEST_NMS_THRESHOLD, label)
            self.show_output(image_number+1, image_from_list.copy(), cv_color_img, cv_test_img, cv_depth_img, nms_out, M=M)
        time_end = datetime.now()
        elapsed_time = (time_end - time_start)
        print("Computation time for 10 images : ", str(elapsed_time))
        print("Computation time for 1 images : ", str(elapsed_time/10.0))
    
    def show_output(self, image_number, raw_img, aruco_img, drbox_img, depth_img, nms_out, score_threshold=0.1, M=np.eye(3)):
        raw_roi = drbox_img.copy()

        # Parameter
        scale_x = 2 # 60 cm / 300 pixel
        scale_y = 2 # 60 cm / 300 pixel
        cam_to_base = 800 # 700 mm jarak kamera ke base
        
        marker_height = 12 # ketinggian marker
        bin_height = 6 # ketinggian permukaan bin dari meja

        depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_img.copy(), alpha=0.03), cv.COLORMAP_JET)
        M_invers = np.linalg.pinv(M)
        R = cv.getRotationMatrix2D((0,0), 90, 1)
        RR = np.eye(3)
        RR[:2,:3] = R[:,:]
        for i in range(len(nms_out)):
            # x, y, w, h, label, angle, score = nms_out[i]
            x = nms_out[i][0]
            y = nms_out[i][1]
            # if w > h
            if(nms_out[i][2] > nms_out[i][3]):
                w = nms_out[i][2]
                h = nms_out[i][3]
                angle = nms_out[i][5]
            else:
                h = nms_out[i][2]
                w = nms_out[i][3]
                angle = 90 + nms_out[i][5] 
            label = nms_out[i][4]
            
            score = nms_out[i][6]
            color_red = (0,0,255)
            color_green = (0,255,0)

            if score >= score_threshold:
                tl_x = int(x - (w//2))
                tl_y = int(y - (h//2))
                bl_x = int(x - (w//2))
                bl_y = int(y + (h//2))

                tr_x = int(x + (w//2))
                tr_y = int(y - (h//2))
                br_x = int(x + (w//2))
                br_y = int(y + (h//2))

                R = cv.getRotationMatrix2D(center=(x,y), angle=-angle, scale=1)
                l = 25
                line_head = np.array([[x + l], [y], [1]])
                tl = np.array([[tl_x], [tl_y], [1]])
                bl = np.array([[bl_x], [bl_y], [1]])
                tr = np.array([[tr_x], [tr_y], [1]])
                br = np.array([[br_x], [br_y], [1]])

                r_tl = R.dot(tl)
                r_bl = R.dot(bl)
                r_tr = R.dot(tr)
                r_br = R.dot(br)
                r_line_head = R.dot(line_head)
                points = np.array([[r_tl[0], r_tl[1]],
                                    [r_tr[0], r_tr[1]],
                                    [r_br[0], r_br[1]],
                                    [r_bl[0], r_bl[1]],], dtype=np.int32)
                points = points.reshape((- 1 , 1 , 2 ))
                
                # Draw on drbox output
                cv.line(drbox_img, (int(x), int(y)), (int(r_line_head[0]), int(r_line_head[1])), color_red, 2)
                cv.circle(drbox_img, (int(x), int(y)), 5, color_green, -1)
                cv.polylines(drbox_img, [points], True, color_red, 2)

                # Invers tranform
                object_loc = np.array([[[x, y]]], dtype=np.float32)
                object_loc_in_raw = cv.perspectiveTransform(object_loc, M_invers)
                rr_tl = cv.perspectiveTransform(np.array([[[r_tl[0,0], r_tl[1,0]]]], dtype=np.float32), M_invers)
                rr_tr = cv.perspectiveTransform(np.array([[[r_tr[0,0], r_tr[1,0]]]], dtype=np.float32), M_invers)
                rr_br = cv.perspectiveTransform(np.array([[[r_br[0,0], r_br[1,0]]]], dtype=np.float32), M_invers)
                rr_bl = cv.perspectiveTransform(np.array([[[r_bl[0,0], r_bl[1,0]]]], dtype=np.float32), M_invers)
                rr_line_head = cv.perspectiveTransform(np.array([[[r_line_head[0,0], r_line_head[1,0]]]], dtype=np.float32), M_invers)
                rr_points = np.hstack((rr_tl, rr_tr, rr_br, rr_bl))
                rr_points = rr_points.reshape((- 1 , 1 , 2 ))
                raw_x = object_loc_in_raw[0,0,0]
                raw_y = object_loc_in_raw[0,0,1]

                # Draw on raw image
                cv.line(aruco_img, (int(raw_x), int(raw_y)), (int(rr_line_head[0,0,0]), int(rr_line_head[0,0,1])), color_red, 2)
                cv.circle(aruco_img, (int(raw_x), int(raw_y)), 5, color_green, -1)
                cv.polylines(aruco_img, [rr_points.astype(np.int32)], True, color_red, 2)

                # Draw on depth image
                cv.line(depth_colormap, (int(raw_x), int(raw_y)), (int(rr_line_head[0,0,0]), int(rr_line_head[0,0,1])), color_red, 2)
                cv.circle(depth_colormap, (int(raw_x), int(raw_y)), 5, color_green, -1)
                cv.polylines(depth_colormap, [rr_points.astype(np.int32)], True, color_red, 2)

                # Calculation
                depth = depth_img[int(raw_y), int(raw_x)] # kalkulasi jarak dari depth
                pose_x = x * scale_x
                pose_y = y * scale_y
                pose_z = (cam_to_base - depth) # dikonversi dalam satuan mm
                pose_yaw = angle

                print("Angle :", angle, "degrees")
                print("Depth :", depth, "mm")
                print("Object Detection :", (int(x), int(y), int(w), int(h), pose_yaw))
                print("Pose Estimation :", (pose_x, pose_y, pose_z, 0, 0, pose_yaw))

                # drbox output, actual pose, predicted pose, 
                print("%.2f, %.2f, %1d, %1d, %1d, %1d, %.2f, %.2f, %.2f, %.2f" % (x, y, 0, 0, 0, 0, pose_x, pose_y, pose_z, pose_yaw))

        save_path = os.getcwd() + "/d455_result/5a/"

        raw_img_path = save_path + "/raw_rgb_" + str(image_number) + ".jpg"
        raw_roi_path = save_path + "/raw_roi_" + str(image_number) + ".jpg"
        aruco_img_path = save_path + "/result_rgb_" + str(image_number) + ".jpg"
        drbox_img_path = save_path + "/result_roi_" + str(image_number) + ".jpg"
        result_depth_path = save_path + "/result_depth_" + str(image_number) + ".jpg"

        print("raw_img_path :", raw_img_path)
        print("raw_roi_path :", raw_roi_path)
        print("aruco_img_path :", aruco_img_path)
        print("drbox_img_rotate_path :", drbox_img_path)
        print("result_depth_path :", result_depth_path)

        cv.imwrite(raw_img_path, raw_img)
        cv.imwrite(raw_roi_path, raw_roi)
        cv.imwrite(aruco_img_path, aruco_img)
        cv.imwrite(drbox_img_path, drbox_img)
        cv.imwrite(result_depth_path, depth_colormap)

        cv.imshow("Aruco Detector", aruco_img)
        cv.imshow("Hasil", drbox_img)
        cv.imshow("Depth Image", depth_colormap)

        cv.waitKey(0)

    def save(self, step):
        model_name = "DrBoxNet.model"
        self.saver.save(self.sess, os.path.join(self.model_save_path, model_name), global_step=step)

    def load(self):
        import re
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.model_save_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.model_save_path, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(os.path.join(self.model_save_path, ckpt_name)))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            print(" [*] Load the pretrained network")
            self.load_prenet()
            return False, 0                                  
    
    def load_prenet(self):
        data_list = np.load(PRETRAINED_NET_PATH, allow_pickle=True, encoding='latin1').item()
        data_keys = data_list.keys()
        var_list = self.detector.vars
        for var in var_list:
            for key in data_keys:
                if key in var.name:
                    if 'weights' in var.name:                        
                        self.sess.run(tf.assign(var, data_list[key][0]))
                        print("pretrained net {} weights -> scene net {}".format(key, var.name))
                        break
                    else: # for biases
                        self.sess.run(tf.assign(var, data_list[key][1]))
                        print("pretrained net {} biases  -> scene net {}".format(key, var.name))
                        break            
                
if __name__ == '__main__':
    net = DrBoxNet()
    if FLAGS.train:
        net.train()
    else:
        net.test_from_depth()
                        
