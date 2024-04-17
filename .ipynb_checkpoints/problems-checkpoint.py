import numpy as np

def cvrp_problem(size,capacity,num_vehicle,seed,items=[]):
    np.random.seed(seed)
    locs = np.round(np.random.uniform(0,1,(size,2)),2).tolist() # 生成城市坐标

    if len(items) != size-1:
        items = np.random.randint(1,capacity,size-1)
        if num_vehicle!=0: # when num_vehicle neq 0, let the [SUM/Capacity]=num_vehicle
            while(num_vehicle!=np.math.ceil(np.sum(items)/capacity)):
                items = np.random.randint(1,capacity,size-1)
    np.random.seed()

    return locs,items
