from qiskit import execute, Aer, transpile
from qiskit.circuit import QuantumCircuit, ParameterVector
from scipy.optimize import minimize
import math

import numpy as np
import json
import os
from itertools import permutations

from terms import obj_C,obj_C_value,obj_C_dict
from operators import read_z_terms
from multiprocessing import Pool, Manager
from tqdm.contrib.concurrent import process_map  # process_map 自动处理进度条
from tqdm import tqdm

class QAOA_CVRP:
    def __init__(self, locs, items, capacity, depth=1, optimizer='COBYLA', hamiltonian=''):
        self.solution = []
        self.solution_distance = np.inf
        self.res = None
        # self.probs_his = []
        self.feasible_idx = []
    
        # for p3s set, as the depot location is last coordinate in the locs, hence, we change it to the first:
        # locs.insert(0,locs.pop())
        
        self.items = items
        self.capacity = capacity
        
        self.N = len(locs)
        self.weights = np.zeros((self.N,self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i!=j:
                    x1,y1 = locs[i]
                    x2,y2 = locs[j]
                    self.weights[i,j] = np.sqrt((x1-x2)**2+(y1-y2)**2)
        self.locs = locs
        
        self.vehicle = math.ceil(np.sum(items)/capacity)

        self.time = self.N-2
        if self.N == 5:
            self.time = 2
        
        self.penalty = [(np.max(self.weights)*(self.N-1))**2,(np.max(self.weights)*(self.N-1))**2,np.max(self.weights)*(self.N-1)]
            
        self.num_qubits = self.vehicle*self.time*self.N+self.vehicle*math.ceil(np.log2(self.capacity+1))
        print(self.num_qubits)
        self.hamiltonian = np.zeros((1,2**self.num_qubits))
        
        self.depth = depth
        self.optimizer = optimizer
        
        wires = np.arange(self.num_qubits)[::-1].tolist()
        registers={}
        registers['x']={}
        registers['y']={}
        for k in range(self.vehicle):
            registers['x'][k] = {}
            for t in range(1,self.time+1):
                registers['x'][k][t] = []
                for _ in range(self.N):
                    registers['x'][k][t].append(wires.pop())
        for k in range(self.vehicle):
            registers['y'][k] = []
            for m in range(math.ceil(np.log2(self.capacity+1))):
                registers['y'][k].append(wires.pop())
            registers['y'][k] = registers['y'][k][::-1]
            
        self.registers = registers
        
        self.circuit = QuantumCircuit(self.num_qubits)
        
        terms = obj_C(self.weights,self.vehicle,self.time,self.capacity,self.items,self.penalty,registers)
        
        params = ParameterVector('P', 2*self.depth)
        
        #initial state
        for i in range(self.num_qubits):
            self.circuit.h(i)
        
        for p in range(self.depth):
            read_z_terms(self.circuit,params[p],terms.args)
        
            for i in range(self.num_qubits):
                self.circuit.rx(2*params[p+self.depth],i)        
        
        # self.feasible_idx == []
        # self.solution = []
        # self.solution_distance = np.inf
        
        if hamiltonian!='':
            if os.path.exists(f'{hamiltonian}.npy') and os.path.exists(f'{hamiltonian}_idx.npy'):
                self.hamiltonian = np.load(f'{hamiltonian}.npy')
                self.feasible_idx = np.load(f'{hamiltonian}_idx.npy').tolist()
            else:
                self.get_hamiltonian()
                # self.hamiltonian = terms_2_hamiltonian(obj_C_dict(self.weights,self.vehicle,self.time,self.capacity,self.items,self.penalty,registers),self.num_qubits)
                np.save(f'{hamiltonian}.npy',self.hamiltonian)
                np.save(f'{hamiltonian}_idx.npy',np.array(self.feasible_idx))
        else:
            print('no hamiltonian path')
        
        self.hamiltonian = np.round(self.hamiltonian,6)
        self.opt_value = np.min(self.hamiltonian[0])
        self.opt_idx = np.where(self.hamiltonian[0] == self.opt_value)[0].tolist()
                    
        self.probs_buffer = None
    
            
    def get_cost(self,key):
        return obj_C_value(self.weights,self.vehicle,self.time,self.capacity,self.items,self.penalty,self.registers,key)
    
    def process_key(self, num):
        key = format(num, f'0{self.num_qubits}b')
        cost = self.get_cost(key)
        result = (num, cost, self.feasible(key))
        return result

    def update_results(self, result):
        num, cost, is_feasible = result
        self.hamiltonian[0][num] = cost
        if is_feasible:
            self.feasible_idx.append(num)
            if self.solution_distance > cost:
                self.solution_distance = cost
                self.solution = self.get_solution(num)

    def get_hamiltonian(self):
        with Pool() as pool:
            results = process_map(self.process_key, range(2**self.num_qubits), max_workers=32, chunksize=75000)
            for result in results:
                self.update_results(result)
        
        
    def execute_circuit(self,parameters):
        # bind parameters
        if len(parameters) == 2 * self.depth:
            bound_circuit = self.circuit.assign_parameters(parameters).reverse_bits()
            state = np.asarray(execute(bound_circuit, backend=Aer.get_backend('statevector_simulator')).result().get_statevector()).reshape((-1, 1))
            probs = np.abs(state) ** 2
            return probs
    
    def expected_value(self,probs):
        exp = self.hamiltonian @ probs
        return exp[0][0]
    
    def optimize_fun(self):
        def f(params):
            probs = self.execute_circuit(params)
            self.probs_buffer = probs
            exp = self.expected_value(probs)
            print(exp)
            return exp
        return f
    
    def run(self,init_params=[]):
        # initializing the parameter set
        if len(init_params)!= 2 * self.depth:
            init_params = (np.pi*np.random.rand(self.depth,)).tolist()+(np.pi*np.random.rand(self.depth,)/2).tolist()
        self.res = minimize(self.optimize_fun(), np.asarray(init_params), method=self.optimizer)
    
    def feasible(self,key):
        encoding = [int(i) for i in key]
        for k in range(self.vehicle):
            for t in range(1,self.time+1):
                if np.sum([encoding[self.registers['x'][k][t][i]] for i in range(self.N)]) != 1:
                    return False
        
        for i in range(1,self.N):
            s = 0
            for k in range(self.vehicle):
                for t in range(1,self.time+1):
                    s += encoding[self.registers['x'][k][t][i]]
            if s != 1:
                return False
            
        for k in range(self.vehicle):
            demands = 0
            for t in range(1,self.time+1):
                for i in range(1,self.N):
                    if encoding[self.registers['x'][k][t][i]] == 1:
                        demands += self.items[i-1]
            if demands>self.capacity:
                return False
        return True
    
    
    def get_solution(self,idx):
        if idx in self.feasible_idx:
            key = format(idx,f'0{self.num_qubits}b')
            encoding = [int(i) for i in key]
            solution = []
            for k in range(self.vehicle):
                solution.append((0,np.argmax([encoding[self.registers['x'][k][1][i]] for i in range(self.N)])))
                for t in range(1,self.time):
                    solution.append((np.argmax([encoding[self.registers['x'][k][t][i]] for i in range(self.N)]), np.argmax([encoding[self.registers['x'][k][t+1][i]] for i in range(self.N)])))
                solution.append((np.argmax([encoding[self.registers['x'][k][self.time][i]] for i in range(self.N)]),0))
        return solution
    
    
    def get_distance(self,idx):
        if idx in self.feasible_idx:
            distance = 0
            solution = self.get_solution(idx)
            for s in solution:
                distance += self.weights[s[0]][s[1]]
            return distance
            
    
    def feasible_counts(self,probs):
        counts = {}
        for idx in self.feasible_idx:
            counts[idx] = probs[idx][0]
        return counts
    
    def opt_counts(self,probs):
        counts = {}
        for idx in self.opt_idx:
            counts[idx] = probs[idx][0]
        return counts
    
    
    def save(self,path='./results.json'):
        save_dict = {}
        save_dict['N'] = self.N
        save_dict['customer_num'] = self.N-1
        save_dict['coordinates'] = self.locs
        save_dict['distances'] = self.weights
        save_dict['item_weights'] = self.items
        save_dict['capacity'] = self.capacity
        save_dict['vehicle'] = self.vehicle
        save_dict['time'] = self.time
        save_dict['penalty'] = self.penalty
        save_dict['qubit_num'] = self.num_qubits
        
        if self.res!=None:
            save_dict['optimize_res'] = self.res
        if self.solution!=[]:
            save_dict['solution'] = self.solution
        if self.solution_distance!=np.inf:
            save_dict['solution_distance'] = self.solution_distance
        if self.res!=None:
            feasible_counts = self.feasible_counts(self.probs_buffer)
            save_dict['feasible_counts'] = feasible_counts
            opt_counts = self.opt_counts(self.probs_buffer)
            save_dict['opt_counts'] = opt_counts
            save_dict['feasibility_ratio'] = sum([feasible_counts[key] for key in feasible_counts.keys()])
            save_dict['optimality_ratio'] = sum([opt_counts[key] for key in opt_counts.keys()])    
            save_dict['opt_value'] = self.opt_value    
            save_dict['expected_value'] = self.res.fun
            save_dict['opt_gap'] = self.res.fun/self.opt_value-1
        
        dumped = json.dumps(save_dict, cls=NpEncoder)
        with open(path, 'w') as result_file:
            result_file.write(dumped)

        
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
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, timedelta):
            return str(obj)
        return super(NpEncoder, self).default(obj)
