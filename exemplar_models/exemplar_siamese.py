import numpy as np
import tensorflow as tf
import logging

from exemplar_models.discretizer import HardDiscretizer, GaussKernelDiscretizer, Discretizer
from exemplar_models.exemplar_model import ExemplarModel
from exemplar_models.utils import BatchSampler, linear, assert_shape, print_shape

LOGGER = logging.getLogger(__name__)


def exemplar_relu_net(input, dout=8, layers=2, dim=16, reuse=False, output_var=False):
    with tf.variable_scope('relu_net', reuse=reuse):
        output = input
        for l in range(layers):
            with tf.variable_scope('layer_%d'%l):
                output = tf.nn.elu(linear(output, dout=dim, init_scale=1.0))
        output = linear(output, dout=dout, name='out')
        if output_var:
            logstd = linear(output, dout=dout, name='out_logstd')
    if output_var:
        return output, logstd
    return output

def exemplar_tanh_net(input, dout=8, layers=2, dim=16, reuse=False, output_var=False):
    with tf.variable_scope('tanh_net', reuse=reuse):
        output = input
        for l in range(layers):
            with tf.variable_scope('layer_%d'%l):
                output = tf.nn.tanh(linear(output, dout=dim, init_scale=1.0))
        output = linear(output, dout=dout)
        if output_var:
            logstd = linear(output, dout=dout, name='out_logstd')
    if output_var:
        return output, logstd
    return output


class ExemplarSiamese(ExemplarModel):
    def __init__(self, dX, wt_net_arch=lambda x, dout: x,
                 cat_net_arch=exemplar_tanh_net,
                 dfeat=8, name='exemplar',
                 data_transformer=None):
        super(ExemplarSiamese, self).__init__(dX, data_transformer=data_transformer)
        self.name = name
        dX = self.data_transformer.dim_transformed
        with tf.variable_scope(name) as vs:
            self.exemplars = tf.placeholder(tf.float32, [None, dX], 'exemplars')
            self.lr = tf.placeholder(tf.float32, [], 'lr')
            self.negatives = tf.placeholder(tf.float32, [None, dX], 'negatives')

            exemplars = self.exemplars
            #exemplars = tf.Print(self.exemplars, [self.exemplars], message='exemplars', summarize=10)

            with tf.variable_scope('wt_net'):
                wts_pos = wt_net_arch(exemplars, dout=dfeat)
                assert_shape(wts_pos, [None, dfeat])
            with tf.variable_scope('wt_net', reuse=True):
                features_neg = wt_net_arch(self.negatives, dout=dfeat)
                assert_shape(features_neg, [None, dfeat])
            with tf.variable_scope('wt_net', reuse=True):
                features_pos = wt_net_arch(self.exemplars, dout=dfeat)
                assert_shape(features_pos, [None, dfeat])

            self.wt_net_wts = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                scope='%s/wt_net'%vs.name)

            #wts_pos = tf.Print(wts_pos, [wts_pos], message='wts_pos')

            pos_concat = tf.concat([wts_pos, features_pos], axis=1)
            assert_shape(pos_concat, [None, 2*dfeat])
            with tf.variable_scope('cat_net'):
                pos_logits = cat_net_arch(pos_concat, dout=1)
                assert_shape(pos_logits, [None, 1])

            neg_concat = tf.concat([wts_pos, features_neg], axis=1)
            assert_shape(neg_concat, [None, 2*dfeat])
            with tf.variable_scope('cat_net', reuse=True):
                neg_logits = cat_net_arch(neg_concat, dout=1)
                assert_shape(neg_logits, [None, 1])

            self.cat_net_wts = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                 scope='%s/cat_net'%vs.name)


            self.probs_exemplar = tf.nn.sigmoid(pos_logits)

            labels_pos = tf.stop_gradient(pos_logits * 0) + 1.0
            labels_neg = tf.stop_gradient(neg_logits * 0)

            pos_cent_loss = tf.nn.sigmoid_cross_entropy_with_logits(pos_logits, labels_pos)
            neg_cent_loss = tf.nn.sigmoid_cross_entropy_with_logits(neg_logits, labels_neg)
            self.loss = tf.reduce_mean(pos_cent_loss)+tf.reduce_mean(neg_cent_loss)

            self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def generate_heatmap(self, env, negatives, target_exemplar, itrs=2000):
        gps = env.get_dense_gridpoints()
        #self.fit(exemplars=gps, negatives=negatives, itrs=itrs, nn_lr=1e-4)
        exemplar_preds = self.predict_exemplars(gps)
        #exemplar_probs = 1.0/exemplar_preds - 1.0
        exemplar_probs = exemplar_preds
        heatmap = env.predictions_to_heatmap(exemplar_probs)
        heatmap /= heatmap.max()
        return heatmap

    def _fit(self, exemplars, neg_batch_func, itrs=2000, heartbeat=500, batch_size=32,
              lr=1e-3, **kwargs):
        LOGGER.info("Fitting exemplar: %s", self.name)

        pos_sampler = BatchSampler(exemplars)
        pos_batch_func = lambda batch_size: pos_sampler.random_batch(batch_size=batch_size)

        sess = tf.get_default_session()
        loss = 0

        for i in range(itrs):
            feed_dict = {
                self.exemplars: pos_batch_func(batch_size),
                self.negatives: neg_batch_func(batch_size),
                self.lr : lr
            }
            loss, _ = sess.run([self.loss, self.train_op], feed_dict)
            if i%heartbeat == 0:
                LOGGER.info('Itr %d/%d loss: %f', i, itrs, loss)

        return loss

    def _predict_exemplars(self, points):
        sess = tf.get_default_session()
        probs = sess.run(self.probs_exemplar, {self.exemplars: points})[:,0]
        return probs

    def get_wt_net(self):
        return tf.get_default_session().run(self.wt_net_wts)


class ExemplarSiameseNoisy(object):
    def __init__(self, dX, dZ, net_arch, net_arch_args=None, cat_net_arch=None, name='exemplar_noisy',
                 loss_func=tf.nn.sigmoid_cross_entropy_with_logits):
        self.name = name
        self.dX = dX
        self.dZ = dZ
        if net_arch_args is None:
            net_arch_args = {'dout': dZ}

        with tf.variable_scope(name) as vs:
            self.exemplars = tf.placeholder(tf.float32, [None, dX], 'exemplars')
            self.negatives = tf.placeholder(tf.float32, [None, dX], 'negatives')
            self.noise_ex = tf.placeholder(tf.float32, [None, self.dZ], 'noise_ex')
            self.noise_neg = tf.placeholder(tf.float32, [None, self.dZ], 'noise_neg')
            self.lr = tf.placeholder(tf.float32, [], 'lr')
            self.num_negatives = tf.placeholder(tf.float32, [], 'num_negatives')

            with tf.variable_scope('wt_net'):
                wts_pos_mu, log_var_pos = net_arch(self.exemplars, **net_arch_args)
            with tf.variable_scope('feat_net', reuse=False):
                features_neg_mu, log_var_negfeat = net_arch(self.negatives, **net_arch_args)
            with tf.variable_scope('feat_net', reuse=True):
                features_pos_mu, log_var_posfeat = net_arch(self.exemplars, **net_arch_args)
            features_neg = tf.sqrt(tf.exp(log_var_negfeat))*self.noise_neg + features_neg_mu
            features_pos = tf.sqrt(tf.exp(log_var_posfeat))*self.noise_neg + features_pos_mu
            wts_pos = tf.sqrt(tf.exp(log_var_pos))*self.noise_ex + wts_pos_mu

            def latent_reg(log_var_enc, mu):
                latent_reg = 0.5 * tf.reduce_sum(1 + log_var_enc - tf.square(mu) - tf.exp(log_var_enc), reduction_indices=1)
                return -tf.reduce_mean(latent_reg)

            # KL between encoder and N(0,1)
            self.vae_reg = 0.5*latent_reg(log_var_pos, wts_pos_mu) + 0.25*(latent_reg(log_var_negfeat, features_neg_mu)+
                                                                           latent_reg(log_var_posfeat, features_pos_mu))
            self.vae_reg = 1.0*self.vae_reg

            pos_concat = tf.concat([wts_pos, features_pos], axis=1)
            assert_shape(pos_concat, [None, 2*dZ])
            with tf.variable_scope('cat_net'):
                pos_logits = cat_net_arch(pos_concat, dout=1)
                assert_shape(pos_logits, [None, 1])

            neg_concat = tf.concat([wts_pos, features_neg], axis=1)
            assert_shape(neg_concat, [None, 2*dZ])
            with tf.variable_scope('cat_net', reuse=True):
                neg_logits = cat_net_arch(neg_concat, dout=1)
                assert_shape(neg_logits, [None, 1])

            self.cat_net_wts = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                 scope='%s/cat_net'%vs.name)

            self.probs_exemplar = tf.nn.sigmoid(pos_logits)

            labels_pos = tf.stop_gradient(pos_logits * 0) + 1.0
            labels_neg = tf.stop_gradient(neg_logits * 0)

            pos_cent_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pos_logits, labels=labels_pos)
            neg_cent_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=neg_logits, labels=labels_neg)
            self.cent_loss = tf.reduce_mean(pos_cent_loss)+tf.reduce_mean(neg_cent_loss)
            self.loss = self.cent_loss + self.vae_reg

            self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def init_tf(self):
        tf.get_default_session().run(tf.global_variables_initializer())

    def fit(self, exemplars, negatives, **kwargs):
        sampler = BatchSampler(negatives)
        neg_batch_func = lambda batch_size: sampler.random_batch(batch_size=batch_size)
        return self.__fit(exemplars, neg_batch_func, **kwargs)

    def fit_buf(self, exemplars, negative_buf, **kwargs):
        batch_func = \
            lambda batch_size: negative_buf.random_batch(batch_size=batch_size)['observations']
        return self.__fit(exemplars, batch_func, **kwargs)

    def __fit(self, exemplars, neg_batch_func, itrs=2000, heartbeat=500, batch_size=32,
              lr=1e-3, **kwargs):
        LOGGER.info("Fitting exemplar: %s", self.name)

        pos_sampler = BatchSampler(exemplars)
        pos_batch_func = lambda batch_size: pos_sampler.random_batch(batch_size=batch_size)

        sess = tf.get_default_session()
        loss = 0

        for i in range(itrs):
            feed_dict = {
                self.exemplars: pos_batch_func(batch_size),
                self.negatives: neg_batch_func(batch_size),
                self.num_negatives: batch_size,
                self.noise_ex: 1.0*np.random.randn(batch_size, self.dZ),
                self.noise_neg: 1.0*np.random.randn(batch_size, self.dZ),
                self.lr : lr
            }
            loss, vae_reg, cent_loss, _ = sess.run([self.loss, self.vae_reg, self.cent_loss, self.train_op], feed_dict)
            if i%heartbeat == 0:
                #LOGGER.info('Itr %d/%d loss: %.2f, reg: %.2f cen: %.2f', i, itrs, loss, vae_reg, cent_loss)
                print('Itr %d/%d loss: %.2f, reg: %.2f cen: %.2f' %( i, itrs, loss, vae_reg, cent_loss))

        return loss

    def generate_heatmap(self, env, negatives, _, itrs=2000):
        gps = env.get_dense_gridpoints()
        exemplar_preds = self.predict_exemplars(gps)
        exemplar_probs = 1.0/exemplar_preds - 1.0

        heatmap = env.predictions_to_heatmap(exemplar_probs)
        return heatmap

    def predict_neg(self, points, negatives):
        sess = tf.get_default_session()
        probs = sess.run(self.probs_neg, {self.exemplars: points, self.negatives:negatives})[:,0]
        return probs

    def predict_exemplars(self, points, **kwargs):
        sess = tf.get_default_session()
        N = points.shape[0]
        probs = sess.run(self.probs_exemplar, {self.exemplars: points,
                                               self.noise_ex: np.zeros((N, self.dZ)),
                                               self.noise_neg: np.zeros((N, self.dZ))})[:,0]
        return probs


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    #test_gauss()
    test_twod()
