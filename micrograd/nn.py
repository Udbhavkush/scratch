import random
from micrograd.engine import Value


class Module:
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
    
    def parameters(self):
        return []


class Neuron(Module):
    
    def __init__(self, nin, nonlin=True): # nin: number of inputs
        # initialize weights and bias
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.nonlin = nonlin # to check if we want non linear activation function or not
    
    def __call__(self, x):  # magic function to make the class object callable. Ex n = Neuron(), then we can call n(2)
        # forward pass: w *  x + b
        act = sum(wi*xi for wi, xi in zip(self.w, x)) + self.b   # raw activation
        out = act.relu() if self.nonlin else act  # since relu is a nonlinear activation function 
        return out
    
    def parameters(self):
        return self.w + [self.b]
    

class Layer(Module):
    
    def __init__(self, nin, nout, **kwargs): # nin: number of inputs for each neuron, nout: number of neurons in the layer. **kwargs used to handle nonlin called from MLP class
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
        
    def __call__(self, x):
        # we have to call neuron(x) for each neuron object
       outs = [n(x) for n in self.neurons]
       return outs[0] if len(outs) == 1 else outs
   
    def parameters(self):
       return [p for neuron in self.neurons for p in neuron.parameters()] # list comprehension for the commented code
   

class MLP(Module):
    
    def __init__(self, nin, nouts): # nin: number of inputs; nouts: size of each layer or no. of neurons in each layer. nouts is a list
        sizes = [nin] + nouts # since, for the first layer, the input will go directly for the next layer, output of the previous layer would be the input. Hence, adding nin to sizes with nouts
        self.layers = [Layer(sizes[i], sizes[i+1], nonlin = i != len(nouts)-1) for i in range(len(nouts))]  # nonlin=True for all the layers except the last layer
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x) # output of the previous layer is input to the next layer
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
