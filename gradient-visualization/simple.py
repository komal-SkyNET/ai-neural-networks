from custom_plot import Plot, plt
import numpy as np
import random as r
import sys
import copy, time, threading

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

class NeuralNet:
    global sigmoid 

    def __init__(self):
        self.inputLayers = 2
        self.outputLayer = 1
        self.plot1 = Plot()
    
    def setup(self, disable_bias=False, bias=10):
        self.i = np.array([r.random(), r.random()], dtype=float).reshape(2,)
        self.w = np.array([r.random(), r.random()], dtype=float).reshape(2,)
        if not disable_bias:
            self.bias = r.random()
        else:
            self.bias = bias
        self.disable_bias = disable_bias

    def forward_propogate(self):
        self.z = self.w*self.i
        self.o = sigmoid(sum(self.z)-self.bias)

    
    def optimize_cost(self, desired):
        i=0
        current_cost = 0.5*pow(desired - self.o, 2)
        for weight in self.w:
            dpdw =  -1*(desired-self.o) * (sigmoid(self.o, derivate=True)) * self.i[i]
            self.w[i] = self.w[i] - 2*dpdw
            i+=1
        #calculate dp/dB
        if not self.disable_bias:
            dpdB = -1*(desired-self.o) * (sigmoid(self.o, derivate=True)) * -1
            self.bias = self.bias - 2*dpdB
        self.forward_propogate()
        with self.plot1.lock:
            self.plot1.Z_points.append(current_cost)
            self.plot1.W1_points.append(self.w[0])
            self.plot1.W2_points.append(self.w[1])
            self.plot1.bias_points.append(self.bias)
    
    def train(self, ip, op):
        self.i = np.array(ip).reshape(2,)
        self.forward_propogate()
        self.optimize_cost(op[0])

class TrainThread(threading.Thread):

    def __init__(self, neural_net):
        super(TrainThread, self).__init__()
        self.neural_net = neural_net
        self.neural_net.setup()

    def run(self):
        success_rate = 0
        trial=0
        done = False
        n = self.neural_net
        while not done:
            a = [0.1,1,0.1,1]
            b = [0.1,0.1,1,1]
            c = [0,0,0,1]
            time.sleep(0.01)
            for i in range(len(a)):
                trial +=1
                n.train([a[i],b[i]],[c[i]])
                if c[i] - n.o < 0.1:
                    success_rate +=1
                    print(100*success_rate/trial, "%")
                if 100*success_rate/trial > 99 and trial > 4:
                    print(100*success_rate/trial, "%")
                    print("Network trained, took: {} trials".format(trial))
                    print("Network weights:{}, bias:{}".format(n.w, n.bias))
                    done = True
                    break

class PlotAnimationThread(threading.Thread):

    def __init__(self, neural_net):
        super(PlotAnimationThread, self).__init__()
        self.neural_net = neural_net 
    
    def animate(self):
        self.neural_net.plot1.animate()
    
    def save_anim(self):
        self.neural_net.plot1.save_anim()
    
    def run(self):
        self.animate()
        self.save_anim()


n = NeuralNet()
# Because matplotlib doesn't like background threads updating GUI WINDOWS
# if we want to view the updating plot, do not run PlotAnimationThread functions as thread 
n.plot1.show_animation = False
trainThread = TrainThread(n)
trainThread.start()
time.sleep(0.1)   
ani_thread = PlotAnimationThread(n)
ani_thread.daemon = True
ani_thread.start()
time.sleep(0.1)
trainThread.join()
print("Training thread finished")
ani_thread.join()
print('Animation thread finished')

### TODO: Make these plots work alongwith animation plot
# n.plot1.anim_ax.clear()
# plt.clf()
# n.plot1.draw_3d_lineplot()
# n.plot1.draw_3d_scatter_plot()
# plt.show()
