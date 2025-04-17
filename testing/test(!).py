import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("Built with GPU support:", tf.test.is_built_with_gpu_support())
print("Available GPU devices:", tf.config.list_physical_devices('GPU'))
import tensorflow.lite as tflite
print("TFLite is available!")
