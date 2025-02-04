import tensorflow as tf


if __name__ == "__main__":
    gpus_list = tf.config.experimental.list_physical_devices("GPU")
    print(gpus_list)
    for gpu in gpus_list:
        tf.config.experimental.set_memory_growth(gpu, True)
