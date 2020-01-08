import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
import numpy as np 
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
_window = 0
_fig = plt.figure()

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
        ax = self.fig.add_subplot(211, projection='3d')
        ax.plot(X, Y, Z, label=label)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        ax.view_init(30, 250)


    def draw_3d_scatter_plot(self, label='ScatterPlot', x_label='x', y_label='y', z_label='z'):
        X = self.W1_points
        Y = self.W2_points
        Z = self.Z_points
        ax = self.fig.add_subplot(212, projection='3d')
        ax.scatter(X, Y, Z, label=label)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        ax.view_init(30, 250)

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