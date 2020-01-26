def fix_random(seed):
    # Seed value
    # Apparently you may use different seed values at each stage
    seed_value = seed

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    import os

    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CUDNN_USE_AUTOTUNE'] = str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    import random

    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    import numpy as np

    np.random.seed(seed_value)

    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    import tensorflow as tf
    from tensorflow import set_random_seed

    set_random_seed(seed_value)
    # tf.set_random_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    from keras import backend as K

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, allow_soft_placement=True,
                                  device_count={'CPU': 1})
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

    # удивительно, но это надо для того, чтобы зафиксировать рандом в Керасе
    # Пруф: https://github.com/keras-team/keras/issues/2743
    from keras.models import Sequential