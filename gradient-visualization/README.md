## Visualizing error/cost function change for partial derivative applied to weight1 & weight2
```
Z1 = w1s1; Z2 = w2s2
Zp = Z1 + Z2
E = | d - (sigmoid(Zp))| ^ 2
w= w - 2* dE/dw (partial derivative)
```

```python
Excerpt from optimize cost function in simple.py:

  current_cost = 0.5*pow(desired - self.o, 2)
  for weight in self.w:
      dpdw =  -1*(desired-self.o) * (sigmoid(self.o, derivate=True)) * self.i[i]
      self.w[i] = self.w[i] - 2*dpdw
      i+=1
  #calculate dp/dB
  dpdB = -1*(desired-self.o) * (sigmoid(self.o, derivate=True)) * -1
        
```
![image](https://github.com/komal-SkyNET/ai-neural-networks/blob/master/gradient-visualization/weight_vs_cost.png)

## Regression & gradient descent anaylsis 

![image](https://github.com/komal-SkyNET/ai-neural-networks/blob/master/gradient-visualization/gradient_sigmoid_regression_analysis_surface_2.png)


## Regression & gradient descent anaylsis (without biasing)
![image](https://github.com/komal-SkyNET/ai-neural-networks/blob/master/gradient-visualization/without_biasing.png)
