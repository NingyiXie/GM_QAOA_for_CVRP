import numpy as np
from itertools import combinations
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import MCXGate


def w(x): #D^n_k => D^x_1
    qc = QuantumCircuit(x)
    qc.x(0)
    for i in range(x-1):
        qc.cry(2*np.arccos(1/np.sqrt(x-i)),i,i+1)
        qc.cnot(i+1,i)
    return qc.to_gate(label=f' W{x}')

def sn(x): #swap network
    qc = QuantumCircuit(x)
    for i in range(1,x):
        qc.swap(0,i)
    return qc.to_gate(label=f'swap\nnet')

def ux(n): #
    qubits_num = n**2
    register = []
    for t in range(n):
        register.append([t*n+i for i in range(n)])
        
    qc = QuantumCircuit(qubits_num)
    
    #step1
    qc.append(w(n),register[0])
    for i in range(n):
        qc.x(register[-1][i])
    # qc.barrier()
    for i in range(n):
        qc.cnot(register[0][i],register[-1][i])
    # qc.barrier()
    
    #step2
    for x in range(n-1,2,-1):
        qc.append(w(x),register[-2][n-x:])
        for i in range(n):
            qc.cswap(register[-1][i],register[-2][-1],register[n-x][i])
            if i!=n-1:
                qc.append(sn(x).control(1),[register[-1][i]]+register[-2][n-x:])
        # qc.barrier()
        for i in range(n):
            qc.cnot(register[n-x][i],register[-1][i])
        # qc.barrier()
    
    #step3
    if n>2:
        for i,j in combinations(np.arange(n).tolist(),2):
            qc.append(w(2).control(2),[register[-1][i],register[-1][j],register[-2][i],register[-2][j]])
        # qc.barrier()
        for i in range(n):
            qc.cnot(register[-2][i],register[-1][i])
        # qc.barrier()
    
    return qc.to_gate(label=r'$U_x$')

def ux_dagger(n):
    qc = QuantumCircuit(n**2)
    qc.append(ux(n).inverse(),[i for i in range(n**2)])
    return qc.to_gate(label=r'$U_x^{\dagger}$')

def increment(num):
    wires = [i for i in range(num)][::-1]
    qc = QuantumCircuit(num)
    for i in range(num-1):
        qc.mcx(wires[:num-1-i],wires[num-1-i])
    qc.x(wires[0])
    return qc.to_gate(label=r'$+1$')

def decrement(num):
    qc = QuantumCircuit(num)
    qc.append(increment(num).inverse(),[i for i in range(num)])
    return qc.to_gate(label=r'$-1$')

def plus(num,q):
    wires = [i for i in range(num)]
    qc = QuantumCircuit(num)
    bits = format(q,f'0{num}b')[::-1]
    for i,b in enumerate(bits):
        if b=='1':
            qc.append(increment(num-i),wires[:num-i])
    label = f'+{q}'
    return qc.to_gate(label=label)

def minus(num,q):
    qc = QuantumCircuit(num)
    qc.append(plus(num,q).inverse(),[i for i in range(num)])
    label = f'-{q}'
    return qc.to_gate(label=label)

def larger(num,Q):
    qc = QuantumCircuit(num+1)
    c_wires = [i for i in range(num)]
    t_wire = num
    bit = format(Q,f'0{num}b')
    for i,b in enumerate(bit):
        if b=='0':
            ctrl_state = bit[:i]+'1'
            qc.append(MCXGate(i+1,ctrl_state=ctrl_state[::-1]),c_wires[:i+1]+[t_wire])
    label = rf'>{Q}'
    return qc.to_gate(label=label)

def CE(N,Q,q,num_qubits,register_x,register_y,register_a,register_d,register_c,register_r):
    qc = QuantumCircuit(num_qubits)
    num_d = len(register_d)

    #initial d and c
    for i in range(N-1):
        qc.append(plus(num_d,q[i]).control(1),[register_x[0][i]]+register_d)
    for i in range(N-1):
        qc.cnot(register_x[0][i],register_c[i])
        
    for t in range(1,N-1):
        # register demand
        for i in range(N-1):
            qc.append(plus(num_d,q[i]).control(1),[register_x[t][i]]+register_d)
        
        # control a_t
        qc.append(larger(num_d,Q).control(1,ctrl_state=0),[register_y[t]]+register_d+[register_a[t]])
        qc.cnot(register_y[t],register_a[t])
        
        # recover d
        if t < N-2:
            for i in range(N-1):
                qc.append(minus(num_d,q[i]).control(2),[register_a[t]]+[register_c[i]]+register_d)
            
        # recover c
        if t==1:
            for i in range(N-1):
                qc.mcx([register_x[0][i],register_a[1]],register_c[i])
        
        elif t>1 and t<N-2:
            for i in range(N-1):
                qc.mcx([register_a[t],register_c[i]],register_r[t][i])
                qc.cnot(register_r[t][i],register_c[i])
            
        # register c
        if t < N-2:
            for i in range(N-1):
                qc.cnot(register_x[t][i],register_c[i])
            
    return qc.to_gate(label=r'$U_E$')

def CE_dagger(N,Q,q,num_qubits,register_x,register_y,register_a,register_d,register_c,register_r):
    qc = QuantumCircuit(num_qubits)
    qc.append(CE(N,Q,q,num_qubits,register_x,register_y,register_a,register_d,register_c,register_r).inverse(),[i for i in range(num_qubits)])
    return qc.to_gate(label=r'$U_E^{\dagger}$')

def exp_minus_i_gamma_hc(num_qubits,terms,gamma):
    qc = QuantumCircuit(num_qubits)
    read_z_terms(qc,gamma,terms)
    return qc.to_gate(label=r'$exp(-i\gamma HC)$')

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

