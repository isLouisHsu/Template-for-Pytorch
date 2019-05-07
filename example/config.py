from easydict import EasyDict

configer = EasyDict()

configer.ckptdir = './ckpt'
configer.logdir = './log'

configer.inputsize = [1]
configer.batchsize = 10
configer.n_epoch = 150

configer.lrbase = 0.001
configer.adjstep = [100]
configer.gamma = 0.1

configer.cuda = False

