'''
存放通用的函数, 不仅仅限于数值计算.
'''

import numpy as np
import matplotlib.pyplot as plt
import functools
import time
import os
import json
import numpy
from types import SimpleNamespace
from matplotlib.figure import Figure
from matplotlib import rc
from scipy.integrate import quad


def timer(func):
    """Print the runtime of the decorated function.

    参考自: https://realpython.com/primer-on-python-decorators/

    Parameters
    ----------
    func : function
           the function to use.
    Returns
    -------
    """
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer


def quad_recorded(func, *args, **kwargs):
    """
    use scipy.integrate.quad, but return the results with additional
    information "nc" and "vc"
    Returns:
        inte_res: the return of scipy.integrate.quad
              nc: the points calculated
              vc: the calculated functiona values
    """
    def func_recorded(x, node_container, value_container):
        res = func(x)
        node_container.append(x)
        value_container.append(res)
        return res
    nc = []
    vc = []
    inte_res = quad(lambda x: func_recorded(x, node_container=nc,
                                            value_container=vc),
                    *args, **kwargs)
    idx = np.argsort(np.array(nc))
    nc = np.array(nc)[idx].tolist()
    vc = np.array(vc)[idx].tolist()
    return inte_res, nc, vc


def counter(func):
    """Print how many times the func run

    Parameters
    ----------
    func : function
           the function to use.
    Returns
    -------
    """
    counter_decorator = []

    @functools.wraps(func)
    def wrapper_counter(*args, **kwargs):
        counter_decorator.append(*args)
        print(f'calculating {func.__name__!r} {len(counter_decorator):n}'
              + 'th time ...')
        value = func(*args, **kwargs)
        return value
    return wrapper_counter


def plot_func(funcs, a, b, n=100, debug=False, save_data=True):
    """Just give a function, get the shape of it.

    It will also save the datePrint the runtime of the decorated function.

    参考自: https://realpython.com/primer-on-python-decorators/

    Parameters
    ----------
    func : function, list of function
           the function to plot.
    a : float
        Lower limit
    b : float
        Upper limit
    n : int
        number of points.
    debug : bool
            If it is `True` , it will print the point data.
    Returns
    -------
    None
    """
    print('====== Start plot_func ============')
    if type(funcs) != list:
        funcs = [funcs]
    x = np.linspace(a, b, n)
    y = np.zeros((len(funcs), n))
    for j in range(len(funcs)):
        if len(funcs) > 1:
            print(f'===CALCULATING THE {j+1:n}/{len(funcs):n} FUNCTION===')
        func = funcs[j]

        @timer
        def calculate_x_and_y():
            for i in range(n):
                if debug:
                    print('Calculating ', i+1, 'th of ', n, 'y data ',
                          f'x={x[i]}')
                y[j, i] = func(x[i])

        calculate_x_and_y()
        plt.plot(x, y[j, :], label=f'function {j+1:n}')

    if save_data:
        dirs = './rfp_data/'
        time_tags = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        np.savetxt(dirs+'x_'+time_tags+'.csv', x, delimiter=',')
        np.savetxt(dirs+'y_'+time_tags+'.csv', y, delimiter=',')
        print('------ x and y has been saved in x.csv and y.csv ------')
        plt.savefig(dirs+'plot_func'+time_tags+'.jpg')
        print('------ Figure has been saved as plot_func.jpg ------')

    plt.legend()
    plt.show()
    print('====== Fished plot_func ============')


def plot_func2(func, a, b, a_parameters, b_parameters, N_x=100,
               N_parameters=10, gap=0, debug=False, save_data=True):
    """Plot a function with a parameters which has different values.

    Plot a function `func(x, parameters)` with a parameters which has different
    values.

    Parameters
    ----------
    func : function
           function to be plot
    a : float
        Lower limit of value x
    b : float
        Upper limit of value x
    a_parameters : float
                  Lower limit of value parameters
    b_parameters : float
                  Upper limit of value parameters

    Returns
    -------
    None
    """

    x = np.linspace(a, b, N_x)
    para = np.linspace(a_parameters, b_parameters, N_parameters)
    y = np.zeros((N_x, N_parameters))

    @timer
    def _cal():

        for j in range(N_parameters):
            for i in range(N_x):
                if debug:
                    print(f'第 {j+1:n} 条线, 第 {i+1:n} 个点, 共',
                          f'{N_parameters:n}, 条线, 每条线 {N_x:n} 个点.')
                    print(f'x={x[i]:.2f}, y={para[j]:.2f}')
                y[i, j] = func(x[i], para[j])
            plt.plot(x, y[:, j]+j*gap, label=f'Parameter={para[j]:.2f}')
            np.savetxt('./x.csv', x, delimiter=',')
            np.savetxt('./y.csv', y[:, j], delimiter=',')
    _cal()

    if save_data:
        dirs = './rfp_data/'
        time_tags = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        np.savetxt(dirs+'x_'+time_tags+'.csv', x, delimiter=',')
        np.savetxt(dirs+'y_plot_func2_'+time_tags+'.csv', y, delimiter=',')
        print('------ x and y has been saved in x.csv and y_plot_func2.csv'
              + '------')
        plt.savefig(dirs+'plot_func2'+time_tags+'.jpg')
        print('------ Figure has been saved as plot_func2.jpg ------')

    plt.legend()
    plt.show()


def plot_func3(func, a_x, b_x, a_y, b_y, N_x=100, N_y=100, debug=False,
               save_data=True):
    x = np.linspace(a_x, b_x, N_x)
    y = np.linspace(a_y, b_y, N_y)

    z = np.zeros((N_y, N_x))
    for i in range(N_y):
        for j in range(N_x):
            if debug:
                print(f'x_{j+1:n} = {x[j]:.3f}, y_{i+1:n} = {y[i]:.3f}')
            z[i, j] = func(x[j], y[i])
    x, y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z,
                    rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))

    if save_data:
        fig.savefig('./fig3d.pdf')
    plt.show()


def prettify_plot():
    """
    change the plot matplotlibrc file

    To use it, please run it before plotting.

    https://matplotlib.org/stable/tutorials/introductory/customizing.html#customizing-with-matplotlibrc-files
    """
    rc('text', usetex=True)
    rc('font', family='serif', serif='Computer Modern Roman', size=8)
    # rc('legend', fontsize=10)
    # rc('mathtext', fontset='cm')
    rc('xtick', direction='in')
    rc('ytick', direction='in')


def get_file_name(path):
    fn = os.path.basename(path)
    fn, _ = os.path.splitext(fn)
    return fn


class DensityFigure():
    """density figure"""
    def __init__(self, extent: list, z: list,
                 x_label: str, y_label: str, title: str, params: str,
                 file_name: str, lines=[], blockade_line=[]):
        self.extent = extent
        self.z = z
        self.blockade_line = blockade_line
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        self.params = params
        self.file_name = file_name
        self.data = locals()
        del self.data['self']
        self.lines = lines

    def plot_fig(self):
        """plot the density fig"""
        prettify_plot()
        fig = Figure(figsize=[6, 4])
        gs = fig.add_gridspec(nrows=1, ncols=30,
                              left=.1, bottom=.1, right=.9, top=.9,
                              wspace=.3, hspace=.3)

        ax_main = fig.add_subplot(gs[0, :20])
        ax_cbar = fig.add_subplot(gs[0, 20:21])
        ax_para = fig.add_subplot(gs[0, 25:])

        im_S = ax_main.imshow(self.z, cmap='rainbow', origin='lower',
                              extent=self.extent)
        ax_main.plot(numpy.linspace(self.extent[0], self.extent[1],
                                    len(self.blockade_line)),
                     self.blockade_line, 'y-')
        # for line in self.lines:
        #     ax_main.plot(line[0], line[1], 'y-')
        fig.colorbar(im_S, cax=ax_cbar)

        ax_main.set_xlabel(self.x_label)
        ax_main.set_ylabel(self.y_label)
        ax_main.set_title(self.title)
        ax_cbar.tick_params(size=0)
        ax_para.axis('off')
        ax_para.text(0, .4, self.params)
        self.fig = fig
        self.ax_main = ax_main
        self.ax_cbar = ax_cbar
        self.ax_para = ax_para
        return self.fig, self.ax_main, self.ax_cbar, self.ax_para

    def save_fig(self, fig_format='pdf'):
        """save the fig"""
        if hasattr(self, 'fig'):
            pass
        else:
            self.plot_fig()
        self.fig.savefig(self.file_name+'.'+fig_format)

    def save_data(self):
        """
        save the data

        how to load:

        with open('filename', 'r') as data_file:
        data = json.load(data_file)
        df = qc.DensityFigure(**data)
        """
        with open(self.file_name+'.json', 'w') as data_file:
            json.dump(self.data, data_file)


class FigureData:
    """
    A dict which figure data save in.
    For example:
        save:
            x1 = [1, 2, 3]
            y1 = [1, 2, 3]
            x2 = [2, 3, 4]
            y2 = [2, 3, 4]
            fd = FigureData()
            fd.add_data('x1', ['x1 label', x1])
            fd.add_data('y1', ['y1 label', y1])
            fd.add_data('x2', ['x2 label', x2])
            fd.add_data('y2', ['x2 label', y2])
            fd.save_data('mydata')
        load:
            fd = FigureData()
            fd.load_data('mydata')
            x1 = fd.d.x1[1]
            y1 = fd.d.y1[1]
            x2 = fd.d.x2[1]
            y2 = fd.d.y2[1]
            x1_label = fd.d.x1[0]
            y1_label = fd.d.y1[0]
            x2_label = fd.d.x2[0]
            y2_label = fd.d.y2[0]
        plot:
            plt.plot(x1, y1, label=y1_label)
            plt.plot(x2, y2, label=y2_label)

    """
    def __init__(self):
        self.data = {}

    def add_data(self, name, data):
        self.data[name] = data

    def save_data(self, file_name):
        with open(file_name + '.json', 'w') as f:
            json.dump(self.data, f)

    def load_data(self, file_name):
        with open(file_name + '.json', 'r') as f:
            self.data = json.load(f)
        self.d = SimpleNamespace(**self.data)
