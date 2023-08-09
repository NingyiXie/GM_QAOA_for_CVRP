from qiskit import execute, Aer, transpile
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

import numpy as np
import json
import os
from itertools import permutations

from terms import obj_terms
from operators import *

class GM_QAOA_CVRP:
    def __init__(self, locs, items, capacity, depth=1, shots=1024, optimizer='COBYLA', backend = 'qasm_simulator', load_path = '',save_path = ''):
        
        self.solution_string = ''
        self.solution_distance = np.inf
        self.buffer = {}
        self.res = None
        
        self.counts_his = []
        
        self.locs = locs
        self.items = items
        self.capacity = capacity
                    
        self.depth = depth
        
        if os.path.exists(load_path) and load_path != '':
            self.load(load_path)

        self.save_path = save_path
        
        self.N = len(locs)
        self.weights = np.zeros((self.N,self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i!=j:
                    x1,y1 = locs[i]
                    x2,y2 = locs[j]
                    self.weights[i,j] = np.sqrt((x1-x2)**2+(y1-y2)**2) # 计算两点之间的欧氏距离
        
        self.shots = shots
        self.backend = backend
        self.optimizer = optimizer

        self.num_qubits = 2*(self.N**2) - 5*self.N + 2 + np.math.ceil(np.log2(capacity+np.max(items)+1)) + self.N-2

        wires = np.arange(self.num_qubits).tolist()[::-1]

        register_x = {}
        for t in range(self.N-1):
            register_x[t] = []
            for _ in range(self.N-1):
                register_x[t].append(wires.pop())
                
        register_y = {}
        for t in range(1,self.N-1):
            register_y[t] = wires.pop()
        
        register_a = {}
        for t in range(1,self.N-1):
            register_a[t] = wires.pop()

        register_d = []
        for i in range(np.math.ceil(np.log2(capacity+np.max(items)+1))):
            register_d.append(wires.pop())

        register_c = []
        for i in range(self.N-1):
            register_c.append(wires.pop())

        register_r = {}
        for t in range(2,self.N-2):
            register_r[t] = []
            for _ in range(self.N-1):
                register_r[t].append(wires.pop())
        
        x_wires = []
        for t in range(self.N-1):
            x_wires += register_x[t]
        y_wires = []
        for t in range(1,self.N-1):
            y_wires.append(register_y[t])
        m_wires = x_wires+y_wires
            
        print(self.num_qubits)
        
        self.circuit = QuantumCircuit(self.num_qubits,len(m_wires))
        
        terms = obj_terms(self.weights,register_x,register_a)

        params = ParameterVector('P', 2*self.depth)
        
        #initial state
        self.circuit.append(ux(self.N-1),x_wires)
        for t in range(1,self.N-1):
            self.circuit.h(register_y[t])
        
        for p in range(self.depth):

            #Phase speration
            
                # condition encoding
            self.circuit.append(CE(self.N,self.capacity,self.items,self.num_qubits,register_x,register_y,register_a,register_d,register_c,register_r),[i for i in range(self.num_qubits)])

            self.circuit.append(exp_minus_i_gamma_hc(self.num_qubits,terms,params[p]),[i for i in range(self.num_qubits)])

            self.circuit.append(CE_dagger(self.N,self.capacity,self.items,self.num_qubits,register_x,register_y,register_a,register_d,register_c,register_r),[i for i in range(self.num_qubits)])

        
            # mixer
            self.circuit.append(ux_dagger(self.N-1),x_wires)
            for t in range(1,self.N-1):
                self.circuit.h(register_y[t])
                
            for i in m_wires:
                self.circuit.x(i)
                
            self.circuit.mcp(-params[self.depth+p],m_wires[:-1],m_wires[-1])

            for i in m_wires:
                self.circuit.x(i)

            self.circuit.append(ux(self.N-1),x_wires)
            for t in range(1,self.N-1):
                self.circuit.h(register_y[t])
            
        if self.backend != 'statevector_simulator':
            self.circuit.measure(m_wires,m_wires)
        
    def optimize_fun(self):
        def f(params):
            exp = 0
            while(exp==0):
                counts = self.execute_circuit(params)
                self.counts_his.append(counts)
                if self.save_path!='':
                    self.save(self.save_path)
                exp = self.expected_value(counts)
            return exp
        return f
    
    
    def execute_circuit(self,parameters):
        # bind parameters
        if len(parameters) == 2 * self.depth:
            bound_circuit = self.circuit.assign_parameters(parameters).reverse_bits()
            # bound_circuit = self.circuit.reverse_bits()
            if self.backend == 'qasm_simulator':
                return execute(bound_circuit, backend=Aer.get_backend('qasm_simulator'), shots=self.shots).result().get_counts()
            elif self.backend == 'statevector_simulator':
                state = np.asarray(execute(bound_circuit, backend=Aer.get_backend('statevector_simulator')).result().get_statevector()).reshape((-1, 1))
                probs = np.abs(state) ** 2
                counts = {}
                feasible_solutions = list(permutations(np.arange(self.N-1).tolist()))
                for s in feasible_solutions:
                    bit = ''
                    for n in s:
                        b = ['0' for _ in range(self.N-1)]
                        b[n] = '1'
                        bit += ''.join(b)
                    for i in range(2**(self.N-2)):
                        key = bit + format(i,f'0{self.N-2}b')
                        idx = int(key+''.join(['0' for _ in range(self.num_qubits-(self.N-1)**2-(self.N-2))]),2)
                        counts[key] = probs[idx][0]
                return counts
            else:
                sim_ = AerSimulator.from_backend(self.backend)
                tcirc = transpile(bound_circuit, sim_)
                counts = sim_.run(tcirc).result().get_counts(0)
                return counts
    
    
    def get_solution(self,key):
        if self.feasible(key):
            key0 = key[:(self.N-1)**2]
            key1 = key[(self.N-1)**2:]
            
            num = self.N-1
            
            s = np.array([int(i) for i in key0]).reshape(num,num)

            solution = [(0,np.argmax(s[0])+1)]

            l = self.items[np.argmax(s[0])]
            for i in range(num-1):
                if l+self.items[np.argmax(s[i+1])]<=self.capacity and key1[i]=='0':
                    solution.append((np.argmax(s[i])+1,np.argmax(s[i+1])+1))
                    l += self.items[np.argmax(s[i+1])]
                else:
                    solution.append((np.argmax(s[i])+1,0))
                    solution.append((0,np.argmax(s[i+1])+1))
                    l = self.items[np.argmax(s[i+1])]
            solution.append((np.argmax(s[-1])+1,0))
        else:
            solution=[]
        return solution
    
    
    def feasible(self,key):
        key0 = key[:(self.N-1)**2]
        solution = []
        array = np.array([int(i) for i in key0]).reshape(self.N-1,self.N-1)
        for row in array:
            idx = np.where(row==1)[0].tolist()
            if len(idx) == 1:
                solution += idx
            else:
                return False
        if len(set(solution))==len(solution) and len(solution)==self.N-1:
            return True
        else:
            return False
    
    
    def expected_value(self,counts):
        exp = 0
        feasible_shots = 0
            
        for key in counts.keys():

            # calculate distance if the solution is not in buffer
            if key not in self.buffer:
                solution = self.get_solution(key)
                    
                distance = 0
                for s in solution:
                    distance += self.weights[s[0]][s[1]]

                # save distance in buffer
                self.buffer[key] = distance
                    
                # best solution so far
                if distance < self.solution_distance and distance!=0:
                    self.solution_string = key
                    self.solution_distance = distance
                    # print(distance)
                        
            if self.buffer[key] != 0:
                exp += counts[key] * self.buffer[key]
                feasible_shots += counts[key]
                    
        exp = exp/feasible_shots
        return exp
    
    def run(self,init_params=[]):
        # initializing the parameter set
        if len(init_params)!= 2 * self.depth:
            # init_params = (np.pi*np.random.rand(self.depth,)).tolist()+(np.pi*np.random.rand(self.depth,)/2).tolist()
            init_params = (np.pi+np.pi*np.random.rand(self.depth,)).tolist()+(np.pi+np.pi*np.random.rand(self.depth,)).tolist()
        if self.backend == 'statevector_simulator':
            # self.res = minimize(self.optimize_fun(), np.asarray(init_params), method=self.optimizer)
            if self.N==4:
                self.res = minimize(self.optimize_fun(), np.asarray(init_params), method=self.optimizer)
            else:
                self.res = minimize(self.optimize_fun(), np.asarray(init_params), method=self.optimizer, tol=5e-2)
        else:
            self.res = minimize(self.optimize_fun(), np.asarray(init_params), method=self.optimizer, tol=1e-2)
        
        return self.res
        
    def save(self,path='./results.json'):
        save_dict = {}
        save_dict['locs'] = self.locs
        # save_dict['weights'] = self.weights
        save_dict['items'] = self.items
        save_dict['capacity'] = self.capacity
        save_dict['depth'] = self.depth
        save_dict['optimize_res'] = self.res
        save_dict['solution_string'] = self.solution_string
        save_dict['solution_distance'] = self.solution_distance
        save_dict['buffer'] = self.buffer
        save_dict['circuit_res'] = self.counts_his
        
        dumped = json.dumps(save_dict, cls=NpEncoder)
        with open(path, 'w') as result_file:
            result_file.write(dumped)
    
    def load(self,path='./results.json'):
        json_file = open(path, 'r')
        data_dict = json.load(json_file)
        
        self.locs = data_dict['locs']
        # self.weights = data_dict['weights']
        self.items = data_dict['items']
        self.capacity = data_dict['capacity']
        self.depth = data_dict['depth']
        self.res = data_dict['optimize_res']
        self.solution_string = data_dict['solution_string']
        self.solution_distance = data_dict['solution_distance']
        self.buffer = data_dict['buffer']
        self.counts_his = data_dict['circuit_res']
        
        
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.floating, np.complexfloating)):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.string_):
            return str(obj)
        return super(NpEncoder, self).default(obj)
