
def slicing(x):
    return tf.slice(x, begin=[0, 5, 5, 0], size=[1, 486, 486, 1])



def padding(x):
    return tf.pad(x, tf.constant([[0, 0], [5, 5], [5, 5], [0, 0]]), "CONSTANT")




def shrinkageact_dense(x, channels=1, threshold=0.0001):
    return tf.where(tf.math.abs(x) < threshold,
                    tf.zeros([1, 1376, 50, channels]), x)



def shrinkageact(x, channels=1, threshold=0.0001):
    return tf.where(tf.math.abs(x) < threshold,
                    tf.zeros([1, 688, 50, channels]), x)



def shrinkageact64(x, channels=64, threshold=0.0001):
    return tf.where(tf.math.abs(x) < threshold,
                    tf.zeros([1, 688, 50, channels]), x)


def recofunc(x):
    x = tf.reshape(x, (cfg.dense_views, 486, 486, 1))
    x = tf.contrib.image.rotate(x, cfg.rot_array, interpolation='BILINEAR')
    # x = tf.reshape(x, (1, cfg.dense_views, 262144, 1))
    x = tf.transpose(x, [3, 1, 2, 0])
    return x