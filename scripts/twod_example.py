import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from exemplar_models.exemplar_siamese import ExemplarSiameseNoisy, exemplar_relu_net, exemplar_tanh_net
from exemplar_models.twod_plotting import map_config, get_dense_gridpoints, predictions_to_heatmap, make_density_map


def pinwheel_data(n=10000, radial_std=.3, tangential_std=.05, rate=0.25, norm_data=None):
    num_classes = 3
    num_per_class = int(n / num_classes)
    assert n % num_classes == 0
    rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)
    features = np.random.randn(num_classes * num_per_class, 2) * np.array([radial_std, tangential_std])
    features[:, 0] += 1
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:, 0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    data = np.random.permutation(np.einsum('ti,tij->tj', features, rotations))

    # normalize data to -1, 1
    if norm_data is None:
        minx, maxx, miny, maxy = np.min(data[:, 0]), np.max(data[:, 0]), np.min(data[:, 1]), np.max(data[:, 1])
        norm_data = {'minx': minx, 'maxx': maxx, 'miny': miny, 'maxy': maxy}
    else:
        minx, maxx, miny, maxy = norm_data['minx'], norm_data['maxx'], norm_data['miny'], norm_data['maxy']
    data[:, 0] = (data[:, 0] - minx) / (maxx - minx)
    data[:, 1] = (data[:, 1] - miny) / (maxy - miny)
    data = 2 * data - 1.0
    return data 


def test_twod():
    grid_config = map_config(xs=(-1,1), ys=(-1,1))
    np.set_printoptions(suppress=True)

    dX = 2
    plot_exemplars = get_dense_gridpoints(grid_config)
    exemplars = plot_exemplars

    negatives = pinwheel_data(9999)#+3.0
    negative_density = make_density_map(negatives, grid_config)

    #feature_net_arch = lambda x: exemplar_relu_net(x, dout=10, layers=2, dim=32)
    #cat_net_arch = lambda x, dout: exemplar_relu_net(x, dout=dout, layers=2, dim=32, output_var=True)
    #ex = ExemplarSiameseNoisy(dX, dZ=64, net_arch=feature_net_arch, cat_net_arch=cat_net_arch)

    noisy_net_arch = lambda x, dout: exemplar_relu_net(x, dout=dout, layers=2, dim=32, output_var=True)
    ex = ExemplarSiameseNoisy(dX, dZ=64, net_arch=noisy_net_arch, cat_net_arch=exemplar_tanh_net)

    os.makedirs('data', exist_ok=True)
    with tf.Session():
        ex.init_tf()

        total_itrs = 50000
        plot_itrs = 50
        itr_per = int(total_itrs/plot_itrs)
        for i in range(plot_itrs):
            ex.fit(exemplars, negatives, itrs=itr_per, batch_size=256, lr=1e-3)
            exemplar_vals = ex.predict_exemplars(plot_exemplars)
            print('%d: max:%f min:%f mean:%f' % (i, np.max(exemplar_vals), np.min(exemplar_vals),
                                                 np.mean(exemplar_vals)))
            probs = (1.0/exemplar_vals) - 1
            print('\tProbs: max:%f min:%f mean:%f' % (np.max(probs), np.min(probs),
                                                 np.mean(probs)))

            # Plot density
            plt.imshow(np.c_[predictions_to_heatmap(probs, grid_config), negative_density],
                       cmap='afmhot')
            plt.savefig(filename='data/itr_%s_%d.png' % (ex.name, i))

if __name__ == "__main__":
    test_twod()
