from easydict import EasyDict

configer = EasyDict()

configer.ckptdir = './ckpt'
configer.logdir = './log'

configer.inputsize = (3, 60, 80)    # (C, H, W)
configer.batchsize = 2**14
configer.n_epoch = 60
configer.valid_freq = 1

configer.lrbase = 0.001
configer.adjstep = [30, 40, 50]
configer.gamma = 0.1

configer.cuda = True

