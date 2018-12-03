# coding: UTF-8
import argparse
import os

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.dataset import iterator as iterator_module
from chainer.training import extensions
from chainer.dataset import convert
from chainer import initializers
from chainer.cuda import to_gpu, to_cpu
from chainer.datasets import tuple_dataset

initializer = initializers.HeNormal()

class Generator(chainer.Chain):
    def __init__(self, N_class=10):
        self.N_class = N_class
        super(Generator, self).__init__(
            #fc1=L.Linear(None, 800),
            #fc2=L.Linear(None, 28 * 28),
            
            l0z = L.Linear(None, 7*7*128, initialW = initializer),
            dc1 = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, initialW = initializer),
            dc2 = L.Deconvolution2D(64, 1, 4, stride=2, pad=1, initialW = initializer),
            bn0 = L.BatchNormalization(7*7*128),
            bn1 = L.BatchNormalization(64),
            
            )

    def __call__(self, z, label):
        #h = F.relu(self.fc1(z))
        #y = F.reshape(F.sigmoid(self.fc2(h)), (-1, 1, 28, 28))
        
        #xp = chainer.cuda.get_array_module(z.data)
        ol = self.to_onehot(label, self.N_class).astype(self.xp.float32).reshape(-1, 10)
        #print(z.shape, ol.shape)
        h = F.concat((z, ol), axis=1)
        h = F.relu(self.bn0(self.l0z(h)))
        h = F.reshape(h, (z.shape[0], 128, 7, 7))
        h = F.relu(self.bn1(self.dc1(h)))
        y = F.sigmoid((self.dc2(h)))
        
        return y

    def make_z(self, n):
        z = self.xp.random.normal(size=(n, self.N_class)).astype(self.xp.float32)
        return z

    def to_onehot(self, label, class_num):
        return self.xp.eye(class_num)[label]

class Critic(chainer.Chain):
    def __init__(self, N_class=10):
        self.N_class = N_class
        super(Critic, self).__init__(
            #fc1=L.Linear(None, 800),
            #fc2=L.Linear(None, 1),
            
            c0 = L.Convolution2D(1+self.N_class, 64, 4, stride=2, pad=1, initialW = initializer),
            c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, initialW = initializer),
            l2 = L.Linear(7*7*128, 1, initialW = initializer), # 1 output (num of output of mattyaDCGAN is 2)
            bn1 = L.BatchNormalization(128),
            )

    def __call__(self, x, label):
        #h = F.relu(self.fc1(x))
        #y = self.fc2(h)
        
        #xp = chainer.cuda.get_array_module(x.data)
        ol = self.xp.asarray(self.to_onehot(label, self.N_class), dtype=self.xp.float32)
        #ol = ol.reshape(x.shape[0], 10, 1, 1)
        #k = self.xp.ones((x.shape[0], 10, 28, 28), dtype=self.xp.float32)
        #k = k * ol
        k = F.broadcast_to(ol.reshape(-1, self.N_class, 1, 1), (ol.shape[0], ol.shape[1], 28, 28))
        h = F.concat((x, k), axis=1)
        h = F.leaky_relu(self.c0(h))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        y = self.l2(h)
        
        return y
    
    def to_onehot(self, label, class_num):
        return self.xp.eye(class_num)[label]
    
    
class Classifier(chainer.Chain):
    def __init__(self, N_class=10):
        self.N_class = N_class
        super(Classifier, self).__init__(
            #fc1=L.Linear(None, 800),
            #fc2=L.Linear(None, 1),
            
            c0 = L.Convolution2D(1, 64, 4, stride=2, pad=1, initialW = initializer),
            c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, initialW = initializer),
            l2 = L.Linear(None, self.N_class, initialW = initializer), # 1 output (num of output of mattyaDCGAN is 2)
            bn1 = L.BatchNormalization(128),
            )

    def __call__(self, x):
        #h = F.relu(self.fc1(x))
        #y = self.fc2(h)
        
        h = F.dropout(F.elu(self.c0(x)), ratio=0.4)
        h = F.dropout(F.elu(self.bn1(self.c1(h))), ratio=0.4)
        y = self.l2(h)
        
        return y


# Conditional WasserStein GAN (CWGAN)
class CWGANUpdater(training.StandardUpdater):
    def __init__(self, train_iter, test_iter, l, n_c, opt_gen, opt_cri, opt_clf, device):
        if isinstance(train_iter, iterator_module.Iterator) and isinstance(test_iter, iterator_module.Iterator):
            iterators = {'main': train_iter, 'val': test_iter}
        self._iterators = iterators
        self.generator = opt_gen.target
        self.critic = opt_cri.target
        self.classifier = opt_clf.target
        self.l = l
        self.n_c = n_c
        self._optimizers = {'generator': opt_gen, 'critic': opt_cri, 'classifier': opt_clf}
        self.device = device
        self.converter = convert.concat_examples
        self.iteration = 0

    def update_core(self):
        # train critic
        for t in range(self.n_c):
            # read data
            batch = self._iterators['main'].next()
            x = self.converter(batch, self.device)
            x, label = x[0], x[1]
            m = x.shape[0]
            H, W = x.shape[2], x.shape[3]
            xp = chainer.cuda.get_array_module(x)

            # generate
            z = self.generator.make_z(m)
            x_tilde = self.generator(z, label)

            # sampling along straight lines
            e = xp.random.uniform(0., 1., (m, 1, 1, 1))
            x_hat = e * x + (1 - e) * x_tilde

            # compute loss
            loss_gan = F.average(self.critic(x_tilde, label) - self.critic(x, label))
            grad, = chainer.grad([self.critic(x_hat, label)], [x_hat],
                                 enable_double_backprop=True)
            grad = F.sqrt(F.batch_l2_norm_squared(grad))

            loss_grad = self.l * F.mean_squared_error(grad,
                                                      xp.ones_like(grad.data))
            loss_critic = loss_gan + loss_grad

            z_normal = xp.random.uniform(-5, 5, z.shape).astype(xp.float32)
            x_fake = self.generator(z_normal, label)
            y_fake = self.classifier(x_fake)
            loss_classifier_generated = F.softmax_cross_entropy(y_fake, label)

            y = self.classifier(x)
            loss_classifier_original = F.softmax_cross_entropy(y, label)

            loss_classifier = loss_classifier_generated + loss_classifier_original

            #with chainer.using_config('enable_backprop', False):
            #    with chainer.using_config('train', False):
            #        acc_classifier_validation = F.accuracy(self.classifier(x_val), label_val)
            acc_classifier_generated = F.accuracy(y_fake, label)
            acc_classifier_original = F.accuracy(y, label)


            # update critic
            self.critic.cleargrads()
            loss_critic.backward()
            self._optimizers['critic'].update()

            # update classifier
            self.classifier.cleargrads()
            loss_classifier.backward()
            self._optimizers['classifier'].update()

            # report
            chainer.reporter.report({
                'wasserstein distance': -loss_gan, 'loss/grad': loss_grad})
            chainer.reporter.report({'loss/classifier': loss_classifier,
                                     'acc/generated': acc_classifier_generated,
                                     'acc/original': acc_classifier_original})
                                     #'acc/validation': acc_classifier_validation})

        # train generator
        # read data
        batch = self._iterators['main'].next()
        x = self.converter(batch, self.device)

        # generate and compute loss
        z = self.generator.make_z(m)
        loss_generator = F.average(-self.critic(self.generator(z, label), label))

        # update generator
        self.generator.cleargrads()
        loss_generator.backward()
        self._optimizers['generator'].update()

        # report
        chainer.reporter.report({'loss/generator': loss_generator})

import copy
class CWGAN_Evaluator(extensions.Evaluator):

    def __init__(self, iterator, classifier, converter=convert.concat_examples,
                 device=None, eval_hook=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator
        self._targets = {'clf': classifier}

        self.converter = converter
        self.device = device
        self.eval_hook = eval_hook
        
    def evaluate(self):
        iterator = self._iterators['main']
        clf = self._targets['clf']
        
        iterator_copy = copy.copy(iterator)
        summary = chainer.reporter.DictSummary()
        for batch in iterator_copy:
            observation = {}
            with chainer.reporter.report_scope(observation):
                x = self.converter(batch, self.device)
                x, label = x[0], x[1]
                batch_size = x.shape[0]
        
                #xp = chainer.cuda.get_array_module(x)
                with chainer.using_config('enable_backprop', False):
                    with chainer.using_config('train', False):
                        y = clf(x)
                
                loss_clf = F.softmax_cross_entropy(y, label)
                acc_clf = F.accuracy(y, label)
        
                observation['loss/validation'] = loss_clf
                observation['acc/validation'] = acc_clf
                
            summary.add(observation)
        
        return summary.compute_mean()

class WeightClipping(object):
    name = 'WeightClipping'

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, opt):
        for param in opt.target.params():
            xp = chainer.cuda.get_array_module(param.data)
            param.data = xp.clip(param.data, -self.threshold, self.threshold)


def main():
    parser = argparse.ArgumentParser(description='WGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=4000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='./result/snapshot_iter_300000.npz',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot')
    args = parser.parse_args()

    generator = Generator(N_class=10)
    critic = Critic(N_class=10)
    classifier = Classifier(N_class=10)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        generator.to_gpu()
        critic.to_gpu()
        classifier.to_gpu()

    opt_gen = chainer.optimizers.Adam(1e-4, beta1=0.5, beta2=0.9)
    opt_gen.setup(generator)

    opt_cri = chainer.optimizers.Adam(1e-4, beta1=0.5, beta2=0.9)
    opt_cri.setup(critic)
    opt_cri.add_hook(WeightClipping(0.1))
    
    opt_clf = chainer.optimizers.Adam(1e-4, beta1=0.5, beta2=0.9)
    opt_clf.setup(classifier)

    train, test = chainer.datasets.get_mnist(withlabel=True, ndim=3)
    """
    if args.gpu >= 0:
        import cupy as xp
        train_data, train_label, test_data, test_label = [], [], [], []
        for bit in train:
            train_data.append(bit[0])
            train_label.append(bit[1])
        for bit in test:
            test_data.append(bit[0])
            test_label.append(bit[1])       
        train_data = to_gpu(xp.array(train_data))
        train_label = to_gpu(xp.array(train_label))
        test_data = to_gpu(xp.array(test_data))
        test_label = to_gpu(xp.array(test_label))
        train = tuple_dataset.TupleDataset(train_data, train_label)
        test = tuple_dataset.TupleDataset(test_data, test_label)    
    """
    
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize,
                                                  shuffle=True)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False)

    updater = CWGANUpdater(train_iter, test_iter, 10, 5,
                          opt_gen, opt_cri, opt_clf, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    def out_generated_image(generator, H, W, rows, cols, dst):
        @chainer.training.make_extension()
        def make_image(trainer):
            n_images = rows * cols
            xp = generator.xp
            xp.random.seed(0)
            z = generator.make_z(rows * cols)
            xp.random.seed(None)
            label = xp.arange(25) % 10
            with chainer.using_config('enable_backprop', False):
                with chainer.using_config('train', False):
                    x = generator(z, label)
            x = chainer.cuda.to_cpu(x.data)

            x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
            channels = x.shape[1]
            x = x.reshape((rows, cols, channels, H, W))
            x = x.transpose(0, 3, 1, 4, 2)
            x = x.reshape((rows * H, cols * W, channels))
            x = np.squeeze(x)

            preview_dir = '{}/preview'.format(dst)
            preview_path = preview_dir +\
                '/image{:0>5}.png'.format(trainer.updater.epoch)
            if not os.path.exists(preview_dir):
                os.makedirs(preview_dir)
            Image.fromarray(x).save(preview_path)
        return make_image

    def out_zeronoise_generated_image(generator, H, W, rows, cols, dst):
        @chainer.training.make_extension()
        def make_image(trainer):
            n_images = rows * cols
            xp = generator.xp
            z = 0.0 * generator.make_z(rows * cols) # generator.xp.zeros([rows*cols, 10]).astype(generator.xp.float32) # generator.make_z(rows * cols)
            label = xp.arange(25) % 10
            with chainer.using_config('enable_backprop', False):
                with chainer.using_config('train', False):
                    x = generator(z, label)
            x = chainer.cuda.to_cpu(x.data)

            x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
            channels = x.shape[1]
            x = x.reshape((rows, cols, channels, H, W))
            x = x.transpose(0, 3, 1, 4, 2)
            x = x.reshape((rows * H, cols * W, channels))
            x = np.squeeze(x)

            preview_dir = '{}/preview'.format(dst)
            preview_path = preview_dir +\
                '/image_zeronoise{:0>5}.png'.format(trainer.updater.epoch)
            if not os.path.exists(preview_dir):
                os.makedirs(preview_dir)
            Image.fromarray(x).save(preview_path)
        return make_image


    # Evaluate the model with the test dataset for each epoch
    trainer.extend(CWGAN_Evaluator(test_iter, classifier, device=args.gpu))
    
    trainer.extend(extensions.dump_graph('wasserstein distance'))
    #trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    snapshot_interval = (args.snapshot_interval, 'iteration')
    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        generator, 'gen_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        critic, 'cri_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        classifier, 'clf_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport())
    trainer.extend(
        extensions.PlotReport(['wasserstein distance', 'loss/grad'],
                              'epoch', file_name='critic.png'))
    trainer.extend(
        extensions.PlotReport(['loss/classifier', 'acc/generated',
                               'acc/original', 'acc/validation'],
                              'epoch', file_name='classifier.png'))
    trainer.extend(
        extensions.PlotReport(
            ['loss/generator'], 'epoch', file_name='generator.png'))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'wasserstein distance', 'loss/grad',
         'loss/generator', 'loss/classifier', 'loss/validation',
         'acc/generated', 'acc/original', 'acc/validation',
         'elapsed_time']))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(out_generated_image(generator, 28, 28, 5, 5, args.out),
                   trigger=(1, 'epoch'))
    trainer.extend(out_zeronoise_generated_image(generator, 28, 28, 5, 5, args.out),
                   trigger=(1, 'epoch'))
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
    trainer.run()

if __name__ == '__main__':
    main()
