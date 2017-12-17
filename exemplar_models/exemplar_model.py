import tensorflow as tf
import numpy as np

from exemplar_models.discretizer import Discretizer
from exemplar_models.utils import BatchSampler


class ExemplarModel():
    def __init__(self, dX, data_transformer=None):
        if data_transformer is None:
            self.data_transformer = Discretizer(dX)
        else:
            self.data_transformer = data_transformer
        self.dX = dX

    def init_tf(self):
        tf.get_default_session().run(tf.global_variables_initializer())

    def fit(self, exemplars, negatives, **kwargs):
        exemplars = self.data_transformer.transform(exemplars)
        sampler = BatchSampler(negatives)
        batch_func = lambda batch_size: self.data_transformer.transform(
            sampler.random_batch(batch_size=batch_size))
        return self._fit(exemplars, batch_func, **kwargs)

    def fit_buf(self, exemplars, negative_buf, **kwargs):
        exemplars = self.data_transformer.transform(exemplars)
        def batch_func(batch_size, actions=False):
            batch = negative_buf.random_batch(batch_size=batch_size)
            obs = self.data_transformer.transform(batch['observations'])
            if actions:
                return obs, batch['actions']
            else:
                return obs
        return self._fit(exemplars, batch_func, **kwargs)

    def _fit(self, exemplars, neg_batch_func, **kwargs):
        raise NotImplementedError()

    def generate_heatmap(self, env, negatives, target_exemplar, itrs=2000):
        gps = env.get_dense_gridpoints()
        self.fit(exemplars=gps, negatives=negatives, itrs=itrs, nn_lr=1e-4)
        exemplar_preds = self.predict_exemplars(gps)
        exemplar_probs = 1.0/exemplar_preds - 1.0
        heatmap = env.predictions_to_heatmap(exemplar_probs)
        heatmap /= heatmap.max()
        return heatmap

    def predict_exemplars(self, points, **kwargs):
        return self._predict_exemplars(self.data_transformer.transform(points), **kwargs)

    def _predict_exemplars(self, points, **kwargs):
        raise NotImplementedError()


class ExemplarEnsemble(ExemplarModel):
    def __init__(self, exemplar_models):
        dX = exemplar_models[0].dX
        super(ExemplarEnsemble, self).__init__(dX, data_transformer=None)
        self.models = exemplar_models

    def _fit(self, exemplars, batch_func, **kwargs):
        for model in self.models:
            model._fit(exemplars, batch_func, **kwargs)

    def ensemble_prediction(self, points):
        preds = []
        for model in self.models:
            pred = model.predict_exemplars(points)
            preds.append(pred)
        return np.array(preds)

    def _predict_exemplars(self, points):
        return np.mean(self.ensemble_prediction(points), axis=0)

    def generate_heatmap(self, env, negatives, target_exemplar, itrs=2000):
        hms = []
        for i, model in enumerate(self.models):
            hm = model.generate_heatmap(env, negatives, target_exemplar.models[i], itrs=itrs)
            hms.append(hm)
        return np.mean(hms, axis=0)

    def get_nn_wts(self):
        wts = []
        for model in self.models:
            wts.append(model.get_nn_wts())
        return wts

    def set_nn_wts(self, wts):
        for i, wt in enumerate(wts):
            self.models[i].set_nn_wts(wt)
