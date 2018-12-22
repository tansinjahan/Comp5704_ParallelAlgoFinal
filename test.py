import tensorflow as tf
import numpy as np
import timeit

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

X = np.random.rand(10000,5)
Y = np.random.rand(5,10000)

with tf.device('/cpu:0'):
    def multiply_two_array(X,Y):
      return np.dot(X,Y)

    result = multiply_two_array(X,Y)
    cpu_result = tf.convert_to_tensor(result, np.float32)
    print("this is the CPU result", cpu_result)

with tf.device('/gpu:0'):
    def multiply_two_array(X,Y):
      return np.dot(X,Y)

    result = multiply_two_array(X,Y)
    gpu_result = tf.convert_to_tensor(result, np.float32)
    print("this is the GPU result", gpu_result)

sess = tf.Session(config=config)
# Test execution once to detect errors early.
try:
  sess.run(tf.global_variables_initializer())
except tf.errors.InvalidArgumentError:
  print(
      '\n\nThis error most likely means that this notebook is not '
      'configured to use a GPU.  Change this in Notebook Settings via the '
      'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
  raise

def cpu():
  sess.run(cpu_result)
  
def gpu():
  sess.run(gpu_result)

print('CPU (s):')
cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")
print(cpu_time)
print('GPU (s):')
gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
print(gpu_time)
print('GPU speedup over CPU: {}x'.format(int(cpu_time/gpu_time)))



