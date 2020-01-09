import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
import numpy as np 
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
_window = 0
_fig = plt.figure()
def sigmoid(ip, derivate=False):
    if derivate:
        return ip*(1-ip)
    return 1.0/(1+np.exp(-1*ip))

class Plot:

    def __init__(self):
        global _window, _fig
        self.W1_points = []
        self.W2_points = []
        self.Z_points = []
        self.fig = _fig
        _window+=1
    
    def draw_3d_lineplot(self, label='LinePlot', x_label='x', y_label='y', z_label='z'):
        X = self.W1_points
        Y = self.W2_points
        Z = self.Z_points
        ax = self.fig.add_subplot(221, projection='3d')
        ax.plot(X, Y, Z, label=label)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        ax.view_init(30, 250)

        
    def draw_3d_scatter_plot(self, label='ScatterPlot', x_label='x', y_label='y', z_label='z'):
        X = self.W1_points
        Y = self.W2_points
        Z = self.Z_points
        ax = self.fig.add_subplot(222, projection='3d')
        ax.scatter(X, Y, Z, label=label)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        ax.view_init(30, 250)
    
    def draw_3d_regression_analysis_surface(self,  bias):
        global sigmoid
        title = "Z = sigmoid({}x+ {}y - {})".format(round(self.W1_points[-1],5), round(self.W2_points[-1],5), round(bias,5))
        ax = self.fig.add_subplot(223, projection='3d')
        x = np.linspace(0,1,50)
        y = np.linspace(0,1,50)
        X, Y = np.meshgrid(x,y)
        Z = sigmoid(self.W1_points[-1]*X + self.W2_points[-1]*Y - bias)
        ax.view_init(30, 250)
        ax.set_title(title)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        ax.contour3D(X, Y, Z, 50, cmap='binary')

    def draw_surface(self,ax, f_z=None):
        global _window
        x = np.linspace(-10,10,500)
        y = np.linspace(-10,10,500)
        X, Y = np.meshgrid(x,y)
        if f_z:
            Z = f_z(X,Y)
        else:
            Z = (Y-X)**2
        _window+=1
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        ax.contour3D(X, Y, Z, 50, cmap='binary')

# anim = FuncAnimation(fig, animate, init_func=init,
#                                frames=200, interval=20, blit=True)