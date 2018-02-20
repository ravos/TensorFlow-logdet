import tensorflow as tf
myLogdet_mod = tf.load_op_library('./myLogdet_op.so')


from tensorflow.python.framework import ops


@ops.RegisterGradient("MyLogdet")
def _my_logdet_grad(op, grad1, grad2):
    grad_batch = tf.expand_dims(tf.expand_dims(grad1, axis=-1), axis=-1)
    return [tf.multiply(grad_batch, op.outputs[1])]


import numpy as np
#import time
np.random.seed(0)
numMat = 777
dim = 100
b = np.zeros(shape=(numMat, dim, dim))
for idx in range(numMat):
    a = np.random.randn(dim,dim)
    b[idx] = 0.01 * a.dot(a.T) + np.eye(dim)
b = np.array(b, dtype='float32')
A = tf.constant(b)


L_gpu = myLogdet_mod.my_logdet(A)
    
sess = tf.Session()

Ag_gpu = sess.run(L_gpu)

print(Ag_gpu[0])
print(np.log(np.linalg.det(b)))

print(np.max(np.abs(Ag_gpu[0] - np.log(np.linalg.det(b)))))
print(np.max(np.abs(Ag_gpu[1] - np.linalg.inv(b))))
