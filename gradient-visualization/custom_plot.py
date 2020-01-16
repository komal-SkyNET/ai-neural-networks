import os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
import numpy as np 
import threading
dir_path = os.path.dirname(os.path.realpath(__file__))
_window = 0
_fig = plt.figure()

def sigmoid(ip, derivate=False):
    if derivate:
        return ip*(1-ip)
    return 1.0/(1+np.exp(-1*ip))

def plt_show():
    '''Text-blocking version of plt.show()
    Use this instead of plt.show()'''
    plt.draw()
    plt.pause(0.01)
    # plt.close()

class Plot(threading.Thread):

    def __init__(self):
        global _window, _fig
        self.W1_points = []
        self.W2_points = []
        self.bias_points = []
        self.Z_points = []
        self.fig = _fig
        self.show_animation = True
        self.lock = threading.Lock()
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
    
    def draw_3d_regression_analysis_surface(self):
        global sigmoid
        title = "Z = sigmoid({}x+ {}y - {})".format(round(self.W1_points[-1],5), round(self.W2_points[-1],5), round(self.bias_points[-1],5))
        ax = self.fig.add_subplot(111, projection='3d')
        ax.view_init(30, 250)
        ax.set_title(title)
        x = np.linspace(0,1,50)
        y = np.linspace(0,1,50)
        X, Y = np.meshgrid(x,y)
        Z = sigmoid(self.W1_points[-1]*X + self.W2_points[-1]*Y - self.bias_points[-1])
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        ax.contour3D(X, Y, Z, 50, cmap='binary')

    def init_animate_regression(self):
        global sigmoid
        x = np.linspace(0,1,20)
        y = np.linspace(0,1,20)
        X, Y = np.meshgrid(x,y)
        with self.lock:
            title = "Z = sigmoid({}x+ {}y - {})".format(round(self.W1_points[-1],5), round(self.W2_points[-1],5), round(self.bias_points[-1],5))
            Z = sigmoid(self.W1_points[0]*X + self.W2_points[0]*Y - self.bias_points[0])
        ax = self.fig.add_subplot(111, projection='3d')
        ax.view_init(30, 250)
        ax.set_title(title)
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none', label=title)
        ax.contour3D(X, Y, Z, 50, cmap='binary')
        self.anim_ax = ax
        surf._facecolors2d=surf._facecolors3d
        surf._edgecolors2d=surf._edgecolors3d
        return surf,

    def update_anim(self, frames):
        global sigmoid
        x = np.linspace(0,1,20)
        y = np.linspace(0,1,20)
        X, Y = np.meshgrid(x,y)
        with self.lock:
            title = "Z = sigmoid({}x+ {}y - {})".format(round(self.W1_points[-1],5), round(self.W2_points[-1],5), round(self.bias_points[-1],5))
            Z = sigmoid(self.W1_points[-1]*X + self.W2_points[-1]*Y - self.bias_points[-1])
        print(title)
        ax = self.anim_ax
        ax.clear()
        ax.set_title(title)
        ax.view_init(30, 250)
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none', label=title)
        ax.contour3D(X, Y, Z, 50, cmap='binary')
        surf._facecolors2d=surf._facecolors3d
        surf._edgecolors2d=surf._edgecolors3d
        if self.show_animation:
            plt_show()
        return surf,

    def animate(self):
        self.anim = FuncAnimation(self.fig, self.update_anim, init_func=self.init_animate_regression, frames=350, interval=1, blit=False)

    def save_anim(self):
        writer = FFMpegFileWriter()
        # self.anim.save('a.mp4', writer=writer)
        self.anim.save('regression_analysis_surface.gif', writer='imagemagick', fps=60)

    def draw_surface(self, ax, f_z=None):
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

