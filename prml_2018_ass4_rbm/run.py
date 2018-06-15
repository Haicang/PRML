import sys
from rbm import *
# load mnist dataset, no label
mnist = np.load('mnist_bin.npy')  # 60000x28x28
n_imgs, n_rows, n_cols = mnist.shape
img_size = n_rows * n_cols
print(mnist.shape)

# lr = [1e-3, 5e-4, 1e-4]
# nh = int(sys.argv[1])

# for l in lr:
#     rbm = RBMtorch(nh, img_size)

#     rbm.train(mnist, T=50, learning_rate=l, batch_size=4, log=True, gpu=True)

#     save_model(rbm, 'rbm{}_{}.model'.format(nh, lr))

nh = 196
lr = 5e-3
T=2

rbm = RBMtorch(nh, img_size)
rbm.train(mnist, T=2, learning_rate=lr, batch_size=4, log=True, gpu=True)
save_model(rbm, 'rbm{}_{}_{}.model'.format(nh, lr, T))