import tensorflow as tf
import cv2
import os
from Model_interface.Model import Model
import numpy as np
import pickle
import json
from collections import OrderedDict
from imgaug import augmenters as iaa
import sys
_slim_path = '/media/nvme/tfslim/models/research/slim'
sys.path.append(_slim_path)
import random
from preprocessing import inception_preprocessing
mean_key = [47, 74, 84, 86, 93]
_TRAIN_IMAGES_ = '/media/nvme/Datasets/facetag/Test/'
#_TRAIN_IMAGES_ = '/media/nvme/Datasets/Celeba/img_align_celeba/'
_VAL_IMAGES_ = '/media/nvme/Datasets/facetag/val/'
#_VAL_IMAGES_ = '/media/Disk3/Datasetdir/custom/Test/'
#_VAL_IMAGES_ = '/media/nvme/Datasets/Celeba/Val/'
_TAGS_ = '/media/nvme/Datasets/facetag/status/'
_LABEL_PATH_ = '/media/Disk3/Datasetdir/custom/clabs/'
_MEAN_PATH_ = '/media/Disk3/Datasetdir/custom/clabs/mean.txt'



pic_in = open("/media/Disk3/Projects/PersonalGit/Mypyscripts/Mypyscripts/dict.pickle","rb")
pic_mean_in = open("/media/Disk3/Projects/PersonalGit/Mypyscripts/Mypyscripts/mean.pickle","rb")
pic_lut_celeba = open('/media/nvme/Datasets/Celeba/lut_celeba.pickle', 'rb')
_TRAIN_CAT_IMAGES= '/media/nvme/Datasets/catdog/train/'
_VAL_CAT_IMAGES_='/media/nvme/Datasets/catdog/tes/'
tags_dict_map = pickle.load(pic_in)
tags_dict_map = OrderedDict(sorted(tags_dict_map.items(), key=lambda t: t[0]))
mean_image = pickle.load(pic_mean_in)
lut_celeba = pickle.load(pic_lut_celeba)

celeba_key = ['5_o_Clock_Shadow',
              'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
              'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup',
              'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
              'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
              'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

_SUMMARY_           = True
_BATCH_SIZE_        = 20
_IMAGE_WIDTH_       = 299
_IMAGE_HEIGHT_      = 299
_PAD_WIDTH_ = 299
_PAD_HEIGHT_ = 299
_IMAGE_CSPACE_      = 3
_CLASSES_           = 12
_MODEL_NAME_        ='MDM'
_ITERATIONS_        = 500000
_LEARNING_RATE_     =  0.01
_SAVE_DIR_          = '/media/nvme//TFmodels/Pnas/'
_SAVE_INTERVAL_     = 5000
_RESTORE_           =  False
_TEST_              =   False
_DROPOUT_ = 0.6
_STATE_         = 'Test'
_SAVE_ITER_     = 10000
_GRAD_NORM_     = 0.5
_RENORM_        = True
_PATCHES_ = 106
_TRAINING_ = True

aug = iaa.SomeOf((0, None), [
        #iaa.AdditiveGaussianNoise(scale=(0, 0.002)),
        iaa.Noop(),
        iaa.GaussianBlur(sigma=(0.0, 1.5)),
        iaa.Dropout(p=(0, 0.02)),
        iaa.AddElementwise((-40, 40), per_channel=0.5),
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*1)),
        #iaa.ContrastNormalization((0.5, 1.5)),
        #iaa.Affine(scale=(1.0, 1.2)),
        #iaa.Affine(rotate=(-10, 10)),
        #iaa.Affine(shear=(-16, 16))

        #iaa.CoarseDropout(0.2, size_percent=(0.001, 0.2))
        ], random_order=True)


def apply_tansform_mat(landmarks, M):
    rot_pts = []
    for point in landmarks:
        x = point[0]
        y = point[1]

        qx = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        qy = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        rot_pts.append([qx, qy])
    return rot_pts


def augment(image, landmark):
    # Affine transformation
    key = random.randint(0, 3)
    w, h, _ = image.shape
    scale = random.uniform(-0.04, 0.04) + 1

    if key == 0:
        dis = image
        rot_pts = landmark
    elif key == 1:
        # rotation transformation

        angle = random.randint(-20, 20)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        dis = cv2.warpAffine(image, M, (w, h))
        rot_pts = apply_tansform_mat(landmark, M)

    elif key == 2:
        # translation transformation

        X_offset = random.randint(-20, 20)
        Y_offset = random.randint(-20, 20)
        M = np.float32([[scale, 0, X_offset], [0, scale, Y_offset]])
        dis = cv2.warpAffine(image, M, (w, h))
        rot_pts = apply_tansform_mat(landmark, M)

    elif key == 3:
        # rotation and translation transformation
        angle = random.randint(-40, 40)
        M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        X_offset = random.randint(-20, 20)
        Y_offset = random.randint(-20, 20)
        M_tr = np.float32([[scale, 0, X_offset], [0, scale, Y_offset]])
        dis = cv2.warpAffine(image, M_rot, (w, h))
        dis = cv2.warpAffine(dis, M_tr, (w, h))
        rot_pts = apply_tansform_mat(landmark, M_rot)
        rot_pts = apply_tansform_mat(rot_pts, M_tr)
    '''
    aug2 = iaa.SomeOf((0, None), [
        # iaa.AdditiveGaussianNoise(scale=(0, 0.002)),
        iaa.Noop(),
        iaa.GaussianBlur(sigma=(0.0, 1.5)),
        iaa.Dropout(p=(0, 0.2)),
        # iaa.CoarseDropout(0.2, size_percent=(0.001, 0.2))
    ], random_order=True)
    '''

    dis = aug.augment_image(dis)
    rot_pts = np.asarray(rot_pts, np.float32)
    dis = dis.astype(np.float32)
    return dis, rot_pts


def get_pts(filename):
    with open(filename) as f:
        data= f.read().split()
        data = np.asarray(data, np.float32)
        data = np.reshape(data, (106,2))
    return data


def pad_image(image, init_pts, mean_pts):
    shape = [_PAD_WIDTH_, _PAD_HEIGHT_]
    canvas = np.zeros(shape + [3], np.uint8)
    im_shape = image.shape
    index = 0
    if im_shape[0] <= im_shape[1]:
        index = 1

    resize_factor = _PAD_HEIGHT_ / float(im_shape[index])
    new_shape = [0, 0]
    new_shape[index] = _PAD_HEIGHT_
    new_shape[abs(index-1)] = int(resize_factor * im_shape[abs(index - 1)])
    image = cv2.resize(image, (new_shape[1], new_shape[0]), 0)
    canvas[0: new_shape[0], 0:new_shape[1]] = image

    init_pts[:, abs(index-1)] = init_pts[:, abs(index-1)] * new_shape[index]
    init_pts[:, index] =  init_pts[:, index] * new_shape[abs(index-1)]


    mean_pts[:, abs(index-1)] = mean_pts[:, abs(index-1)] * new_shape[index]
    mean_pts[:, index] =  mean_pts[:, index] * new_shape[abs(index-1)]
    #canvas = canvas.astype(np.float32) / 255.0
    #canvas -= 0.5
    #canvas *= 2.0
    return canvas, init_pts, mean_pts

def make_gt(j_inp):
    gt_val = np.zeros(2, np.float32)
    val_offset = 0
    for key, vals in tags_dict_map.items():
        j_value = j_inp['is_smiling']
        vals = tags_dict_map['is_smiling']
        index = vals.index(j_value)
        if j_value in 'no' or j_value in 'No':
            gt_val[0] = 0.
        else:
            gt_val[0] = 1.
        j_value = j_inp['is_accesory_visible']
        vals = tags_dict_map['is_accesory_visible']
        index = vals.index(j_value)
        if j_value in 'no' or j_value in 'No':
            gt_val[1] = 0.
        else:
            gt_val[1] = 1.
        #gt_val[val_offset + index] = 1
        val_offset += len(vals)
        return gt_val
'''
def make_catdoggt(inp):
    gt_val = np.zeros(_CLASSES_ , np.float32)

    inp = os.path.splitext(inp)[0]

    inp = os.path.splitext(inp)[0]
    #print(inp)

    inp = inp[-3:]
    if inp in 'cat':
        gt_val[0] = 1
    else:
        gt_val[1] = 1
    return gt_val

def make_celebagt(inp):
    gt_val = np.zeros(40, np.float32)
    #return gt_val
    gt_val = lut_celeba[inp+'.jpg']
    gt_val = gt_val.astype(np.float32)
    return gt_val

def get_imageandtagscat(path):
    name = os.path.split(path.decode())[1]
    name = os.path.splitext(name)[0]

    GT_data = make_catdoggt(name)

    image= cv2.imread(path.decode())
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (_IMAGE_WIDTH_,_IMAGE_HEIGHT_), 0)
    return image, GT_data

def get_imageandtags(path):
    name = os.path.split(path.decode())[1]
    name = os.path.splitext(name)[0]

    pts_name =_TAGS_+ name + '.JSON'
    with open(pts_name) as json_file:
        data = json.load(json_file)

    GT_data = make_gt(data)
    #GT_data = make_celebagt(name)
    image = cv2.imread(path.decode())
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #image = cv2.resize(image, (_IMAGE_WIDTH_,_IMAGE_HEIGHT_), 0)
    #image = aug.augment_image(image)

     
    return image, GT_data
'''

def get_classification_image(path):
    image = cv2.imread(path.decode())
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def get_tags(name):
    pts_name =_TAGS_+ name + '.JSON'
    with open(pts_name) as json_file:
        data = json.load(json_file)

    GT_data = make_gt(data)
    return GT_data

def get_landmarks_and_mean(name):
    lmpts_name = _LABEL_PATH_ + name + '.pts'
    lmpts = get_pts(lmpts_name)
    mean_pts = get_pts(_MEAN_PATH_)

    return lmpts, mean_pts

def align_landmark_image(image, lmpts, mean_pts):
    shape2 = image.shape
    lm_image, lmpts, mean_pts = pad_image(image, lmpts, mean_pts)
    lmpts[:,0] = lmpts[:, 0] /shape2[1]
    lmpts[:,1] = lmpts[:, 1] /shape2[0]

    incep_mean_pts = []
    incep_gt = []
    for key in mean_key:
        incep_mean_pts.append(mean_pts[key])
        incep_gt.append(lmpts[key])
    incep_mean_pts = np.asarray(incep_mean_pts)
    #incep_mean_pts[..., 0] *= _IMAGE_WIDTH_
    #incep_mean_pts[..., 1] *= _IMAGE_HEIGHT_
    incep_mean_pts = np.asarray(incep_mean_pts).reshape([-1])
    incep_gt = np.asarray(incep_gt).reshape([-1])
    #if _TRAINING_:
        #lm_image, lmpts = augment(lm_image, lmpts)

    lm_image = lm_image.astype(np.float32)
    lm_image = lm_image / 255.0
    lm_image -= 0.5
    lm_image *= 2.0

    return lm_image, lmpts, mean_pts, incep_mean_pts, incep_gt

def get_image_tags_points(path):
    name = os.path.split(path.decode())[1]
    name = os.path.splitext(name)[0]

    image = get_classification_image(path)
    GT_data = get_tags(name)
    lmpts, mean_pts = get_landmarks_and_mean(name)
    lm_image, lmpts, mean_pts , incep_mean_pts, incep_lm_gt = align_landmark_image(image, lmpts, mean_pts)
    #augment_image_and_keypoints()
    #augment_image_and_keypoints
    GT_data = np.concatenate((GT_data, incep_lm_gt), axis=0)
    return GT_data, lm_image, lmpts, mean_pts, incep_mean_pts

def decode_img_lmpts(image1):
    gt_data, image, lmpts, mean_pts, incep_mean_pts= tf.py_func(get_image_tags_points, [image1], (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
    #image = inception_preprocessing.preprocess_image(image, _IMAGE_HEIGHT_, _IMAGE_WIDTH_, is_training=False)
    image.set_shape([_PAD_WIDTH_, _PAD_WIDTH_, 3])
    gt_data.set_shape([_CLASSES_])
    lmpts.set_shape([106, 2])
    mean_pts.set_shape([106, 2])
    incep_mean_pts.set_shape([10])
    return gt_data, image, lmpts, mean_pts, incep_mean_pts

'''
def decode_ims(image1):
    image, gt_data= tf.py_func(get_imageandtags, [image1], (tf.uint8, tf.float32) )
    image = inception_preprocessing.preprocess_image(image, _IMAGE_HEIGHT_, _IMAGE_WIDTH_, is_training=_TRAINING_ )
    image.set_shape([_IMAGE_WIDTH_, _IMAGE_HEIGHT_, 3])
    #image = tf.image.per_image_standardization(image)
    #image -= mean_image
    #image = (tf.cast(image, tf.float32) / 255.0) - mean_image
    #image = tf.image.random_brightness(image,0.2)
    #image = tf.image.random_contrast(image, 0.85, 1.15)
    #image = tf.image.random_flip_left_right(image)

    gt_data.set_shape([_CLASSES_ ])

    return image, gt_data
'''

def main():
    Train_images = os.listdir(_TRAIN_IMAGES_)
    Train_images_full = []
    Val_images_full = []

    for image in Train_images:
        Train_images_full.append(_TRAIN_IMAGES_ + image)

    Val_images = os.listdir(_VAL_IMAGES_)


    for image in Val_images:
        Val_images_full.append(_VAL_IMAGES_ + image)

    #Train_images_full = Train_images_full[0:1]
    if _TRAINING_:
        print('TRAINING SET')
        dataset = tf.data.Dataset.from_tensor_slices((Train_images_full))
        dataset = dataset.shuffle(buffer_size=25000).repeat()
    else:
        print('VALIDATION_SET')
        dataset = tf.data.Dataset.from_tensor_slices((Val_images_full))
        dataset = dataset.shuffle(buffer_size=1)


    dataset = dataset.map(decode_img_lmpts)
    dataset = dataset.batch(_BATCH_SIZE_)


    val_dataset = tf.data.Dataset.from_tensor_slices((Val_images_full))
    val_dataset = val_dataset.shuffle(buffer_size=1).repeat()
    val_dataset = val_dataset.map(decode_img_lmpts)
    val_dataset = val_dataset.batch(_BATCH_SIZE_)

    train_iterator = dataset.make_one_shot_iterator()
    test_iterator = val_dataset.make_one_shot_iterator()
    handle = tf.placeholder(tf.string, shape=[])

    iter = tf.data.Iterator.from_string_handle(handle, dataset.output_types, dataset.output_shapes)


    with tf.name_scope('MDM'):
        Simple_DNN = Model(Model_name=_MODEL_NAME_, Summary=_SUMMARY_, \
                Batch_size=_BATCH_SIZE_, Image_width=_IMAGE_WIDTH_, Image_height=_IMAGE_HEIGHT_,\
               Image_cspace=_IMAGE_CSPACE_, Classes=_CLASSES_, Save_dir=_SAVE_DIR_, \
               State=_STATE_, Dropout=_DROPOUT_, Grad_norm=_GRAD_NORM_, Renorm = _RENORM_, 
                           iter=train_iterator, handle=handle, train_iter= train_iterator,
                          test_iter= test_iterator, Patches=_PATCHES_, Training= _TRAINING_ )

    Optimizer_params_adam = {'beta1': 0.9, 'beta2':0.999, 'epsilon':0.1}




    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        #Simple_DNN.Set_optimizer(starter_learning_rate= _LEARNING_RATE_, Optimizer='ADAM', Optimizer_params=Optimizer_params_adam, decay_steps=50000, decay_rate=0.8)
        #Simple_DNN.Construct_Accuracy_op()
        print('Constructing Writers')
        #Simple_DNN.Construct_Writers()
        print('Writers Constructed')
        if _TRAINING_:
            Simple_DNN.Train_Iter(iterations=_ITERATIONS_, save_iterations=_SAVE_INTERVAL_, restore=_RESTORE_, log_iteration=5)
        #Simple_DNN.Try_restore()
        true = 0
        false = 0

        for i in range(len(Val_images)):
            preds , gt , image = Simple_DNN.Predict(session=session)
            image = image[0]
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            thre, rtr = cv2.threshold(preds, 0.5, 1, cv2.THRESH_BINARY)

            rtr = np.squeeze(rtr)
            gt = np.squeeze(gt)
            rtr2 = rtr.astype(np.bool)
            preds = np.squeeze(preds)
            gt = gt.astype(np.bool)
            print(gt)
            print(rtr2)
            print(preds)
            '''
            print('preds')
            for index, val in enumerate(rtr2):
                if val:
                    print(celeba_key[index], preds[index])
            print('GT')
            
            for index, val in enumerate(gt):
                if val:
                    print(celeba_key[index])
            '''
            #print(celeba_key[np.squeeze(rtr).astype(np.bool)])
            #print(preds)
            #print(celeba_key[gt.astype(np.bool)])
            #print(image)
            if gt[0] == True:
                prefix = 'Accesory'
            else:
                prefix = 'Not_Accesory'
            image = ((image + 1)/2)*255
            if rtr2[0] == gt[0]:
                cv2.imwrite('out/true/' + str(i) +'_' + prefix+ '_.jpg', image)
                true +=1
                print('TRUE')
            else:
                cv2.imwrite('out/false/' + str(i) + '_' + prefix + '.jpg', image)
                false +=1
                print('FALSE')

        true = float(true) * 100/ float(len(Val_images))
        false = float(false) * 100/ float(len(Val_images)) 
        print('True:', true)
        print('False:', false)
        input('test')
if __name__ == "__main__":
    main()