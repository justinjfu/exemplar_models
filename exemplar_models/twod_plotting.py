import numpy as np

class MapConfig():
    def __init__(self, xs, ys, xres, yres):
        self.xs = xs
        self.ys = ys
        self.xres = xres
        self.yres=yres

def map_config(xs=(-0.3,0.3), ys=(-0.3,0.3), xres=50, yres=50):
    return MapConfig(xs,ys,xres,yres)

def make_heat_map(eval_func, map_config):
    gps = get_dense_gridpoints(map_config)
    vals = np.zeros(map_config.xres*map_config.yres)
    for i, pnt in enumerate(gps):
        vals[i] = eval_func(pnt)
    return predictions_to_heatmap(vals, map_config)


def get_dense_gridpoints(map_config):
    xl = np.linspace(map_config.xs[0], map_config.xs[1], num=map_config.xres)
    yl = np.linspace(map_config.ys[0], map_config.ys[1], num=map_config.yres)
    gridpoints = np.zeros((map_config.xres*map_config.yres, 2))
    for i in range(map_config.xres):
        for j in range(map_config.yres):
            gridpoints[i+map_config.xres*j] = np.array((xl[i], yl[j]))
    return gridpoints


def predictions_to_heatmap(predictions, map_config):
    map = np.zeros((map_config.xres, map_config.yres))
    for i in range(map_config.xres):
        for j in range(map_config.yres):
            map[i,j] = predictions[i+map_config.xres*j]
    map = map/np.max(map)
    return map.T

def make_density_map(paths, map_config):
    xs = np.linspace(map_config.xs[0], map_config.xs[1], num=map_config.xres+1)
    ys = np.linspace(map_config.ys[0], map_config.ys[1], num=map_config.yres+1)
    y = paths[:,0]
    x = paths[:,1]
    H, xedges, yedges = np.histogram2d(y, x, bins=(xs, ys))
    H = H.astype(np.float)
    H = H/np.max(H)
    return H.T

def plot_maps(combined_list=None, *heatmaps):
    import matplotlib.pyplot as plt
    combined = np.c_[heatmaps]
    if combined_list is not None:
        combined_list.append(combined)
        combined = np.concatenate(combined_list)
    else:
        combined_list = []
    plt.figure()
    plt.imshow(combined, cmap='afmhot', interpolation='none')
    plt.show()
    return combined_list

