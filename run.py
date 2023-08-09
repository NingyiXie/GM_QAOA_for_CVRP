from qaoa import GM_QAOA_CVRP
from problems import cvrp_problem

capacity = 3
num_vehicle= 2
N = 5
seed = 2345
locs,items = cvrp_problem(size=N,capacity=capacity,num_vehicle=num_vehicle,seed=seed)
q=GM_QAOA_CVRP(locs, items, capacity=capacity, depth=2, backend='statevector_simulator',save_path='./n5_q3_seed2345_p2_sv.json')
q.run()
q.save('./n5_q3_seed2345_p2_sv.json')
