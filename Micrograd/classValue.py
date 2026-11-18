import math
import random 
import matplotlib as plt
import numpy as np


class Value:
    def __init__(self, data, _children=(), _operation="", label=""):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._operation = _operation
        self.label = label

    def __repr__(self):
        return f'Value(data={self.data}, label="{self.label}")'

    def __add__(self, other):
        if not isinstance(other, Value): other = Value(other) # incase user uses int or float instrad of Value Class
        out = Value(self.data + other.data, _children=(self, other), _operation="+")
        return out;

    def __mul__(self, other):
        if not isinstance(other, Value): other = Value(other) # incase user uses int or floatinstrad of Value Class
        out = Value(self.data * other.data, _children=(self, other), _operation="*")
        return out;

    def __pow__(self, other):
        if not isinstance(other, (int, float)): other = Value(other) # restricting power to be int or float 
        out = Value(self.data ** other.data, _children=(self, other), _operation=f"**{other.data}")
        return out;

    def __sub__(self, other):
        return self + (-other);

    def __truediv__(self, other):
        return self * (other**-1);

    # In case user uses other * self instead of self * other
    def __rmul__(self, other):
        return self * other;

    def __tanh__(self):
        x = self.data
        tanh = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(tanh, _children=(self,), _operation="tanh")
        return out;

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
    
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


from graphviz import Digraph

def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  
  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
    if n._operation:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n._operation, label = n._operation)
      # and connect this node to it
      dot.edge(uid + n._operation, uid)

  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2._operation)

  return dot


a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = a+b
c.label = "c"

from IPython.display import display

display(draw_dot(c))
