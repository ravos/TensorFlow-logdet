# TensorFlow-logdet
batched log determinant (logdet) GPU op for tensorflow

The following bash commands have been used to compile the op on a Linux machine (tensorflow 1.0 with python 2.7):

    TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

    nvcc -std=c++11 -c -o myLogdet_op.cu.o myLogdet_op_gpu_cublas.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

    gcc -std=c++11 -shared -o myLogdet_op.so myLogdet_op.cc myLogdet_op.cu.o -I $TF_INC -fPIC -L /PATH/TO/CUDA/LIB/ -lcublas -lcudart

    rm *.o

The following bash commands have been used to compile the op on a Linux machine (tensorflow 1.4 with python 3.5+):

    TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
    TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

    nvcc -std=c++11 -c -o myLogdet_op.cu.o myLogdet_op_gpu_cublas.cu.cc \
    -I $TF_INC -I$TF_INC/external/nsync/public \
    -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC \
    -L /PATH/TO/CUDA/LIB/ -L$TF_LIB -ltensorflow_framework \
    -arch=sm_30 --expt-relaxed-constexpr

    gcc -std=c++11 -shared -o myLogdet_op.so myLogdet_op.cc myLogdet_op.cu.o -I $TF_INC -fPIC -L /PATH/TO/CUDA/LIB/ -lcublas -lcudart -L$TF_LIB -ltensorflow_framework

    rm *.o
