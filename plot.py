import matplotlib.pyplot as plt

def plot_cvrp(locs,items):
    plt.figure(figsize=(3,3))
    plt.xlim(0,1)
    plt.ylim(0,1)
    for i in range(1,len(locs)):
        plt.scatter(locs[i][0],locs[i][1])
        plt.text(locs[i][0],locs[i][1]+0.005,f'C{i}')
        plt.text(locs[i][0],locs[i][1]-0.04,f'  +{items[i-1]}')
        
    plt.scatter(locs[0][0],locs[0][1])
    plt.text(locs[0][0],locs[0][1]+0.005,f'Depot')

def plot_cvrp_solution(solution,locs,items):
    plt.figure(figsize=(3,3))
    plt.xlim(0,1)
    plt.ylim(0,1)
        
    for s in solution:
        x1,y1 = locs[s[0]]
        x2,y2 = locs[s[1]]
        plt.plot([x1,x2],[y1,y2],'r--')
        
    for i in range(1,len(locs)):
        plt.scatter(locs[i][0],locs[i][1])
        plt.text(locs[i][0],locs[i][1]+0.005,f'C{i}')
        plt.text(locs[i][0],locs[i][1]-0.04,f'  +{items[i-1]}')
        
    plt.scatter(locs[0][0],locs[0][1])
    plt.text(locs[0][0],locs[0][1]+0.005,f'Depot')
