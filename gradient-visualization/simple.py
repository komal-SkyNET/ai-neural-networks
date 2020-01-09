from custom_plot import Plot, plt
import numpy as np
import random as r
import sys
import copy

def sigmoid(ip, derivate=False):
    if derivate:
        return ip*(1-ip)
    return 1.0/(1+np.exp(-1*ip))

class NeuralNet:
    global sigmoid 

    def __init__(self, disable_bias=False, bias=10):
        self.inputLayers = 2
        self.outputLayer = 1
        if not disable_bias:
            self.bias = r.random()
        else:
            self.bias = bias
        self.plot1 = Plot()
        self.disable_bias = disable_bias
    
    def setup(self):
        self.i = np.array([r.random(), r.random()], dtype=float).reshape(2,)
        self.w = np.array([r.random(), r.random()], dtype=float).reshape(2,)

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
        self.plot1.Z_points.append(current_cost)
        self.plot1.W1_points.append(self.w[0])
        self.plot1.W2_points.append(self.w[1])
    
    def train(self, ip, op):
        self.i = np.array(ip).reshape(2,)
        self.forward_propogate()
        self.optimize_cost(op[0])

n = NeuralNet()
n.setup()
# while sys.stdin.read(1):
success_rate = 0
trial=0
done = False
while not done:
    a = [0.1,1,0.1,1]
    b = [0.1,0.1,1,1]
    c = [0,0,0,1]
    for i in range(len(a)):
        trial +=1
        n.train([a[i],b[i]],[c[i]])
        if c[i] - n.o < 0.01:
            success_rate +=1
            print(100*success_rate/trial, "%")
        if 100*success_rate/trial > 99 and trial > 4:
            print(100*success_rate/trial, "%")
            print("Network trained, took: {} trials".format(trial))
            print("Network weights:{}, bias:{}".format(n.w, n.bias))
            done = True
            break
n.plot1.draw_3d_lineplot(x_label='weight1', y_label='weight2', z_label='error')
plot2 = Plot()
plot2.W1_points = n.plot1.W1_points
plot2.W2_points = n.plot1.W2_points
plot2.Z_points = n.plot1.Z_points
plot2.draw_3d_scatter_plot(x_label='weight1', y_label='weight2', z_label='error')

##draw regression line
plot2.draw_3d_regression_analysis_surface(n.bias)

plt.tight_layout()
plt.show()

# while True:
#     print("-----")
#     print("Enter ip1 & 2")
#     ip1 = float(sys.stdin.readline())
#     ip2 = float(sys.stdin.readline())
#     print(ip1, ip2)
#     n.i = np.array([ip1, ip2]).reshape(2,)
#     n.forward_propogate()
#     print("network output:{}".format(n.o))

#For 0.01 error rate | Network weights:[14.04340991 14.04341878], bias:21.860932213216703

