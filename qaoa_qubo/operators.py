import numpy as np
from itertools import combinations
import math


def RMZ(circ,param,wires):
    for i in range(len(wires)-1):
        circ.cnot(wires[i],wires[i+1])
    circ.rz(param,wires[-1])
    for i in range(len(wires)-2,-1,-1):
        circ.cnot(wires[i],wires[i+1])

def read_z_terms(circ,gamma,terms):
    for term in terms[1:]:
        term_list = str(term).split('*')
        coef = float(term_list[0])
        RMZ(circ,2*gamma*coef,[int(t) for t in term_list[1:]])
