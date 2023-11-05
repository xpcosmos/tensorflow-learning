import tensorflow as tf

# rank-1
x_rank1 = tf.constant(4, shape=(1,1), dtype = tf.float32)
print(x_rank1)

# Manually created matrix (rank-2)
x_rank2 = tf.constant([[1,2,3],[4,5,6]], dtype = tf.float32)
print(x_rank2)

# Matrix only with ones
x_ones = tf.ones((3,3), dtype=tf.int16)
print(x_ones)

# Identity Matrix
x_eye = tf.eye(3)
print(x_eye)

# Identity Matrix
x_normal_dist = tf.random.normal((3,3), mean = 0, stddev=1)
print(x_normal_dist)

# Changing types
x_cast = tf.cast(x_ones, dtype=tf.float32)
print(x_cast)

# Adding two tensors
add1 = tf.add(x_normal_dist, x_cast)
add2 = x_normal_dist + x_cast
print(add1)
print(add2)

# Dot product
dot_product = tf.tensordot(x_cast, x_normal_dist, axes=1)
print(dot_product)

# Matrix multiplication
# -----
mat1 = tf.random.normal((2, 4), dtype=tf.float32)
mat2 = tf.random.normal((4, 3), dtype=tf.float32)

# Method 1
matmul = tf.matmul(mat1, mat2)
print(matmul)

# Method 2
matmul = mat1 @ mat2
print(matmul)
# ---- 
