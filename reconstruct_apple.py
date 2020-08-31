import numpy as np
import SimpleITK as sitk
import tensorflow as tf
import argparse
import os
import imageio
from ctreco.recoNetworkBasis import architecture
import config as cfg
from scipy import ndimage, misc


# Chose GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def apply(input_sino, model_path, save_path, max_apply, offset ,verbose = False):

    if verbose:
        print('   -loading network')

        # This loads the keras network and the first checkpoint file
        model = tf.keras.models.load_model(os.path.join(model_path, 'keras_model.h5'),
                                                custom_objects={'recofunc': architecture.recofunc,
                                                                'shrinkageact': architecture.shrinkageact,
                                                                'shrinkageact_dense': architecture.shrinkageact_dense,
                                                                'shrinkageact64': architecture.shrinkageact64,
                                                                'shrinkageact_slicing': architecture.slicing,
                                                                'shrinkageact_padding': architecture.padding,
                                                                 'tf':tf, 'cfg':cfg},
                                                compile=False)

        checkpoint = tf.train.Checkpoint(model=model)
        latest_model = tf.train.latest_checkpoint(model_path)

        restore_status = [checkpoint.restore(latest_model)]



        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.7

        config.allow_soft_placement = True  # automatically choose a supported device when the specified one doesn't support an op
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

    with tf.Session(config=config, graph=g) as sess:
        sess.run(init_op)

        for element in restore_status:
            element.run_restore_ops()

        if verbose:
            print('   -succesfully loaded from ' + model_path)
            model.summary()
            print('   -running predictions')

        # Predict the reconstructions and save as .tif
        for i, sino in enumerate(input_sino[offset:]):
            if i == max_apply:
                break
            try:
                prediction = sess.run(model(tf.expand_dims(tf.expand_dims(sino*1000, 0), -1), training=False))
                prediction = np.array(prediction)*0.012
                prediction = ndimage.zoom(prediction, [1,972/486,972/486, 1])
                imageio.imwrite(save_path + '_' + str(i+1+offset) + '.tif', np.squeeze(prediction))
            except IndexError:
                pass

        sess.close()


    if verbose:
        print('   -predictions finished')



if __name__ == "__main__":
    file_path = r"E:\Apples-CT\projections_noisefree_nrrd"
    out_path = os.path.join(file_path, '..', 'NetworkOutput50')

    model_path = r"E:\Apples-CT\networks\50"
    sparse = False

    if sparse == True:
        out_path = os.path.join(file_path, '..', 'NetworkOutput25')
        cfg.train_dim2 = 25
        cfg.rot_array = cfg.rot_array[::2]
        cfg.dense_views = 25

    # Create folder_out if it does not exist yet
    try:
        os.mkdir(out_path)
    except:
        pass

    for root, dirs, files in os.walk(file_path):
        for filename in files:

            # read .nnrd sinograms from paths
            input_sino = sitk.ReadImage(os.path.join(root, filename))
            input_sino = sitk.GetArrayFromImage(input_sino)
            if sparse == True:
                input_sino = input_sino[:,:,::2]

            save_path = os.path.join(out_path, filename.replace('.nrrd', ''))


            # Rebuild network every max_apply slices. The application gets extremely slow after several slices.
            max_apply = 100
            j = 0
            while max_apply * j < len(input_sino):
                offset = max_apply * j
                tf.reset_default_graph()
                g = tf.Graph()
                tf.keras.backend.clear_session()
                with g.as_default():
                    apply(input_sino, model_path = model_path, save_path = save_path, max_apply = max_apply, offset = offset, verbose = True)
                j += 1
