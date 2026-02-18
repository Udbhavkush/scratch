import math
import numpy as np
import matplotlib.pyplot as plt


class Value:  # we have created this class sort of an analogy for tensors. Just to manipulate the whole training process with some data structure
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0 # initialized to 0 as we assume that by default no node or weight affects the loss function. This is the derivative of loss function w.r.t. that node
        self._prev = set(_children) # this is used to keep track of the previous nodes in the graph
        self._backward = lambda: None  # this would be a function that would differ for all the different operators
        self._op = _op
        self.label = label
    
    def __repr__(self): # magic function to print the object. Can modify the string representation of the object when printed.
        return f"Value(data={self.data})"
    
    def __add__(self, other): # magic function to add two objects. Can modify the behavior of the + operator when applied to objects of this class. In case of a.__add__(b), a is self and b is other.
        other = other if isinstance(other, Value) else Value(other)
        out =  Value(self.data + other.data, (self, other), '+')
        def _backward(): # we want to take out grad and find out the self grad and other grad, so we will set self.grad to something and other.grad to something
            self.grad += 1.0 * out.grad # local derivative * global derivative
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
            
    def __mul__(self, other): # magic function to multiply two objects. Can modify the behavior of the * operator when applied to objects of this class. In case of a.__mul__(b), a is self and b is other.
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad # local derivative * global derivative
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other): # other * self
        return self * other
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self, ), f'**{other}')
        
        def _backward():
             self.grad += other * (self.data ** (other-1)) * out.grad
        out._backward = _backward
        
        return out
    
    def __neg__(self): # -self
        return self * (-1)
    
    def __sub__(self, other): # self - other
        return self + (-other)
        
    def __truediv__(self, other): # self/other
        return self * (other ** -1)
    
    def __radd__(self, other): # other + self
        return self + other
        
    def relu(self):
        x = self.data
        out = Value(max(0, x), (self, ), 'ReLU')
        
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        
        return out
        
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad # local derivative * global derivative
        out._backward = _backward
        return out
    
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = 1.0 # base condition
        for node in reversed(topo):
            node._backward()