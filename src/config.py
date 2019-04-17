from easydict import EasyDict

configer = EasyDict()

configer.ckptdir = './ckpt'

configer.inputsize = (3, 12, 12)
configer.batchsize = 2**14
configer.n_epoch = 60

configer.lrbase = 0.001
configer.adjstep = [30, 40, 50]
configer.gamma = 0.1

configer.cuda = True

