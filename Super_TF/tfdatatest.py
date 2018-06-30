import tensorflow as tf
import cv2
import os
from Model_interface.Model import Model
import numpy as np

_LABEL_PATH_ = '/media/Disk3/Datasetdir/custom/clabs/'
_Train_IMAGE_PATH_ = '/media/Disk3/Datasetdir/custom/Test/'
#_Train_IMAGE_PATH_ = 'E:/Datasetdir/custom/Test/'
_Train_MASK_PATH_ = '/media/Disk3/Datasetdir/custom/Test/'
_Val_IMAGE_PATH_ ='/media/Disk3/Datasetdir/custom/Test/'
_Val_MASK_PATH_ = '/media/Disk3/Datasetdir/custom/Test/'
_MEAN_PATH_ = '/media/Disk3/Datasetdir/custom/clabs/mean.txt'
_SUMMARY_           = True
_BATCH_SIZE_        = 1
_IMAGE_WIDTH_       = 512
_IMAGE_HEIGHT_      = 512
_IMAGE_CSPACE_      = 3
_CLASSES_           = 1
_MODEL_NAME_        ='MDM'
_ITERATIONS_        = 500000
_LEARNING_RATE_     =  0.01
_SAVE_DIR_          = '/media/nvme/TFmodels/newmdm/'
_SAVE_INTERVAL_     = 5000
_RESTORE_           =  True
_TEST_              =   False
_OUTPUT_PATH_ = 'G:/idridout/'
_DROPOUT_ = 0.6
_STATE_         = 'Train'
_SAVE_ITER_     = 10000
_GRAD_NORM_     = 0.5
_RENORM_        = False
_PATCHES_ = 106

def get_pts(filename):
    with open(filename) as f:
        data= f.read().split()
        data = np.asarray(data, np.float32)
        data = np.reshape(data, (106,2))
    return data

def get_image_andsize(path):
    name = os.path.split(path.decode())[1]
    name = os.path.splitext(name)[0]
    pts_name =_LABEL_PATH_+ name + '.pts'
    pts = get_pts(pts_name)


    
    mean_pts = get_pts(_MEAN_PATH_)
    image= cv2.imread(path.decode())
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    shape = image.shape
     
    pts[:,0] = pts[:, 0] *512.0
    pts[:,1] = pts[:, 1] *512.0

    pts[:,0] = pts[:, 0] /shape[1]
    pts[:,1] = pts[:, 1] /shape[0]

    mean_pts[:,0] = mean_pts[:, 0] * 512.0 * 1.3 -45
    mean_pts[:,1] = mean_pts[:, 1] * 512.0 * 1.3 - 35
    image = cv2.resize(image, (512, 512),  0)
    #print(pts)
    #print(shape)
    #print(mean_pts)
    #input('shape')
    '''
    for ind, pt in enumerate(mean_pts):
        m_coords = (int(pt[0]), int(pt[1]))
        apt = pts[ind]
        a_coords = (int(apt[0]), int(apt[1]))
        cv2.circle(image, m_coords , 4, (255,0,255))
        cv2.circle(image, a_coords , 3, (0,0,255))
    '''
    return image, pts, mean_pts, path


def get_image_mask_pair(images=[], suffix='', ext='.jpg', Image_path='', _Mask_path=''):
    output_images = []
    output_masks = []
    for image in images:
        name = os.path.splitext(image)[0]
        mask_name = name + suffix + ext
        output_images.append(Image_path + image)
        output_masks.append(_Mask_path + mask_name)
    #print(output_images)
    #print(output_masks)

    return output_images, output_masks

def decode_ims(image1):
    image , pts, mean_pts, path= tf.py_func(get_image_andsize, [image1], (tf.uint8, tf.float32, tf.float32, tf.string) )
    image.set_shape([None, None, 3])
    #image1 = cv2.imread(image1)
    #image1 = tf.read_file(image1)
    #image1 = tf.image.decode_image(image1, channels=3)
    image = tf.cast(image, tf.float32) / 255
    '''
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image += tf.random_normal(
                tf.shape(image),
                stddev=0.02,
                dtype=tf.float32,
                seed=42,
                name='add_gaussian_noise')
    image = tf.clip_by_value(image, 0.0, 1.0)
    '''
    #image1 = tf.image.per_image_standardization(image1)
    pts.set_shape([106,2])
    mean_pts.set_shape([106,2])
    '''
    image2 = tf.read_file(image2)
    image2 = tf.image.decode_image(image2, channels=1)
    image2 = tf.cast(image2, tf.float32) / 255
    '''
    return image, pts, mean_pts

def main():

    crap = []
    with tf.device('/cpu:0'):
        images = os.listdir(_Train_IMAGE_PATH_)
        val_images = os.listdir(_Val_IMAGE_PATH_)
        images, masks = get_image_mask_pair(images, Image_path=_Train_IMAGE_PATH_, _Mask_path=_Train_MASK_PATH_)
        
        #crap.append(images[3])
        crap = []
        crap.append(images[1])
        #images = crap

        dataset = tf.data.Dataset.from_tensor_slices((images))
        dataset = dataset.shuffle(buffer_size=5000).repeat()
        dataset = dataset.map(decode_ims)
        dataset = dataset.batch(_BATCH_SIZE_)

        #iterator = dataset.make_one_shot_iterator()
        

        
        val_images, val_masks = get_image_mask_pair(val_images, Image_path=_Val_IMAGE_PATH_, _Mask_path=_Val_MASK_PATH_)
        val_images = images
        val_dataset = tf.data.Dataset.from_tensor_slices((val_images))
        val_dataset = val_dataset.shuffle(buffer_size=1000)
        val_dataset = val_dataset.map(decode_ims)
        val_dataset = val_dataset.batch(_BATCH_SIZE_)
        
        

        init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        i,pt, me = iterator.get_next()
        input_dict= {'input': i, 'target_pts':pt, 'mean_pts':me}
        train_init_op = iterator.make_initializer(dataset)
        val_init_op = iterator.make_initializer(val_dataset)
    
    with tf.name_scope('MDM'):
        Simple_DNN = Model(Model_name=_MODEL_NAME_, Summary=_SUMMARY_, \
                Batch_size=_BATCH_SIZE_, Image_width=_IMAGE_WIDTH_, Image_height=_IMAGE_HEIGHT_,\
               Image_cspace=_IMAGE_CSPACE_, Classes=_CLASSES_, Save_dir=_SAVE_DIR_, \
               State=_STATE_, Dropout=_DROPOUT_, Grad_norm=_GRAD_NORM_, Renorm = False, 
                           Iter=input_dict, train_op= train_init_op, val_op = val_init_op, Patches=_PATCHES_)
    
    Optimizer_params_adam = {'beta1': 0.9, 'beta2':0.999, 'epsilon':0.1}
    Simple_DNN.Set_optimizer(starter_learning_rate= _LEARNING_RATE_, Optimizer='ADAM', Optimizer_params=Optimizer_params_adam, decay_steps=5000, decay_rate=0.8)
    Simple_DNN.Construct_Accuracy_op()
    


    #Simple_DNN.attach_dataset(iter,train_init_op , val_init_op)


    #Simple_DNN. Place_arch()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        print('Constructing Writers')
        Simple_DNN.Construct_Writers()
        print('Writers Constructed')
        #Simple_DNN.Train_Iter(iterations=_ITERATIONS_, save_iterations=_SAVE_INTERVAL_, restore=_RESTORE_, log_iteration=5)
        Simple_DNN.Try_restore(session)
        Simple_DNN.Predict(session=session)
    '''
    sess = tf.InteractiveSession()
    index=0
    t  = True
    while(1):
        sess.run(train_init_op)
        
        index += 1
        if index %10 is not 0:
            sess.run(train_init_op)
            print('TRAIN')
        
        else:
            sess.run(train_init_op)
            print('VAL')
        


        i1, p, m = sess.run([i, pt, me])
        print(p.shape)
        print(m.shape)


        #i1, p, m = sess.run([i, pt, me])
        #i1 = np.asarray(i1)
        print(i1.shape)
        cv2.imshow('image', i1[0])
        #cv2.imshow('mask', m1[0])
        cv2.waitKey(0)
    '''
if __name__ == "__main__":
    main()


