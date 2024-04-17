import sympy
import math
import numpy as np

def obj_C(w,K,T,Q,d,p,registers):
    N = w.shape[0]
    
    sympy_dict_x={}
    sympy_dict_y={}
    for k in range(K):
        sympy_dict_x[k] = {}
        for t in range(1,T+1):
            sympy_dict_x[k][t] = {}
            for i in range(N):
                sympy_dict_x[k][t][i] = 0.5*(1-sympy.symbols(str(registers['x'][k][t][i])))
        sympy_dict_y[k] = {}
        for m in range(math.ceil(np.log2(Q+1))):
            sympy_dict_y[k][m] = 0.5*(1-sympy.symbols(str(registers['y'][k][m])))
        
    
    func_obj = 0
    for k in range(K):
        for t in range(1,T):
            for i in range(N):
                for j in range(N):
                    func_obj += w[i][j] * sympy_dict_x[k][t][i] * sympy_dict_x[k][t+1][j]
        for i in range(N):
            func_obj += w[0][i] * sympy_dict_x[k][1][i]
            func_obj += w[i][0] * sympy_dict_x[k][T][i]
    
    func_c1 = 0
    for i in range(1,N):
        tmp = []
        for k in range(K):
            for t in range(1,T+1):
                tmp.append(sympy_dict_x[k][t][i])
        sub_func_c1 = 1
        for b in range(len(tmp)):
            sub_func_c1 -= tmp[b]
        for b in range(len(tmp)-1):
            for bd in range(b+1,len(tmp)):
                sub_func_c1 += 2*tmp[b]*tmp[bd]
        func_c1 += sub_func_c1
    
    func_c2 = 0
    for k in range(K):
        for t in range(1,T+1):
            sub_func_c2 = 1
            for i in range(N):
                sub_func_c2 -= sympy_dict_x[k][t][i]
            for i in range(N-1):
                for j in range(i+1,N):
                    sub_func_c2 += 2*sympy_dict_x[k][t][i]*sympy_dict_x[k][t][j]
            func_c2 += sub_func_c2
            
    func_c3 = 0
    for k in range(K):
        tmp = []
        for t in range(1,T+1):
            for i in range(1,N):
                tmp.append((sympy_dict_x[k][t][i],d[i-1]))
                
        sub_func_c3 = Q**2
        
        for b in range(len(tmp)):
            sub_func_c3 += tmp[b][0] * ((tmp[b][1]**2) - 2*tmp[b][1]*Q)
        
        for b in range(len(tmp)-1):
            for bd in range(b+1,len(tmp)):
                sub_func_c3 += 2*tmp[b][0]*tmp[bd][0]*tmp[b][1]*tmp[bd][1]
        
        for t in range(1,T+1):
            for i in range(1,N):
                for m in range(math.ceil(np.log2(Q+1))):
                    sub_func_c3 += 2*sympy_dict_x[k][t][i]*d[i-1]*(2**m)*sympy_dict_y[k][m]
        
        for m in range(math.ceil(np.log2(Q+1))):
            sub_func_c3 -= 2*(2**m)*sympy_dict_y[k][m]*Q
        
        for m in range(math.ceil(np.log2(Q+1))):
            sub_func_c3 += (2**m)*(2**m)*sympy_dict_y[k][m]
        
        for m in range(math.ceil(np.log2(Q+1))-1):
            for md in range(m+1,math.ceil(np.log2(Q+1))):
                sub_func_c3 += 2*sympy_dict_y[k][m]*sympy_dict_y[k][md]*(2**m)*(2**md)
        
        func_c3 += sub_func_c3
    
    func = func_obj + p[0]*func_c1 + p[1]*func_c2 + p[2]*func_c3
    
    return sympy.expand(func)


def obj_C_dict(w,K,T,Q,d,p,registers):
    N = w.shape[0]
    
    sympy_dict_x={}
    sympy_dict_y={}
    for k in range(K):
        sympy_dict_x[k] = {}
        for t in range(1,T+1):
            sympy_dict_x[k][t] = {}
            for i in range(N):
                sympy_dict_x[k][t][i] = 0.5*(1-sympy.symbols('Z'+str(registers['x'][k][t][i])))
        sympy_dict_y[k] = {}
        for m in range(math.ceil(np.log2(Q+1))):
            sympy_dict_y[k][m] = 0.5*(1-sympy.symbols('Z'+str(registers['y'][k][m])))
        
    
    func_obj = 0
    for k in range(K):
        for t in range(1,T):
            for i in range(N):
                for j in range(N):
                    func_obj += w[i][j] * sympy_dict_x[k][t][i] * sympy_dict_x[k][t+1][j]
        for i in range(N):
            func_obj += w[0][i] * sympy_dict_x[k][1][i]
            func_obj += w[i][0] * sympy_dict_x[k][T][i]
    
    func_c1 = 0
    for i in range(1,N):
        tmp = []
        for k in range(K):
            for t in range(1,T+1):
                tmp.append(sympy_dict_x[k][t][i])
        sub_func_c1 = 1
        for b in range(len(tmp)):
            sub_func_c1 -= tmp[b]
        for b in range(len(tmp)-1):
            for bd in range(b+1,len(tmp)):
                sub_func_c1 += 2*tmp[b]*tmp[bd]
        func_c1 += sub_func_c1
    
    func_c2 = 0
    for k in range(K):
        for t in range(1,T+1):
            sub_func_c2 = 1
            for i in range(N):
                sub_func_c2 -= sympy_dict_x[k][t][i]
            for i in range(N-1):
                for j in range(i+1,N):
                    sub_func_c2 += 2*sympy_dict_x[k][t][i]*sympy_dict_x[k][t][j]
            func_c2 += sub_func_c2
            
    func_c3 = 0
    for k in range(K):
        tmp = []
        for t in range(1,T+1):
            for i in range(1,N):
                tmp.append((sympy_dict_x[k][t][i],d[i-1]))
                
        sub_func_c3 = Q**2
        
        for b in range(len(tmp)):
            sub_func_c3 += tmp[b][0] * ((tmp[b][1]**2) - 2*tmp[b][1]*Q)
        
        for b in range(len(tmp)-1):
            for bd in range(b+1,len(tmp)):
                sub_func_c3 += 2*tmp[b][0]*tmp[bd][0]*tmp[b][1]*tmp[bd][1]
        
        for t in range(1,T+1):
            for i in range(1,N):
                for m in range(math.ceil(np.log2(Q+1))):
                    sub_func_c3 += 2*sympy_dict_x[k][t][i]*d[i-1]*(2**m)*sympy_dict_y[k][m]
        
        for m in range(math.ceil(np.log2(Q+1))):
            sub_func_c3 -= 2*(2**m)*sympy_dict_y[k][m]*Q
        
        for m in range(math.ceil(np.log2(Q+1))):
            sub_func_c3 += (2**m)*(2**m)*sympy_dict_y[k][m]
        
        for m in range(math.ceil(np.log2(Q+1))-1):
            for md in range(m+1,math.ceil(np.log2(Q+1))):
                sub_func_c3 += 2*sympy_dict_y[k][m]*sympy_dict_y[k][md]*(2**m)*(2**md)
        
        func_c3 += sub_func_c3
    
    func = func_obj + p[0]*func_c1 + p[1]*func_c2 + p[2]*func_c3
    
    return dict(sympy.expand(func).as_coefficients_dict())



def obj_C_value(w,K,T,Q,d,p,registers,key):
    N = w.shape[0]
    
    xv={}
    yv={}
    for k in range(K):
        xv[k] = {}
        for t in range(1,T+1):
            xv[k][t] = {}
            for i in range(N):
                xv[k][t][i] = int(key[registers['x'][k][t][i]])
        yv[k] = {}
        for m in range(math.ceil(np.log2(Q+1))):
            yv[k][m] = int(key[registers['y'][k][m]])        
    
    func_obj = 0
    for k in range(K):
        for t in range(1,T):
            for i in range(N):
                for j in range(N):
                    func_obj += w[i][j] * xv[k][t][i] * xv[k][t+1][j]
        for i in range(N):
            func_obj += w[0][i] * xv[k][1][i]
            func_obj += w[i][0] * xv[k][T][i]
    # print(func_obj)
    func_c1 = 0
    for i in range(1,N):
        tmp = []
        for k in range(K):
            for t in range(1,T+1):
                tmp.append(xv[k][t][i])
        sub_func_c1 = 1
        for b in range(len(tmp)):
            sub_func_c1 -= tmp[b]
        for b in range(len(tmp)-1):
            for bd in range(b+1,len(tmp)):
                sub_func_c1 += 2*tmp[b]*tmp[bd]
        func_c1 += sub_func_c1
    # print(func_c1)
    func_c2 = 0
    for k in range(K):
        for t in range(1,T+1):
            sub_func_c2 = 1
            for i in range(N):
                sub_func_c2 -= xv[k][t][i]
            for i in range(N-1):
                for j in range(i+1,N):
                    sub_func_c2 += 2*xv[k][t][i]*xv[k][t][j]
            func_c2 += sub_func_c2
    # print(func_c2)
    func_c3 = 0
    for k in range(K):
        tmp = []
        for t in range(1,T+1):
            for i in range(1,N):
                tmp.append((xv[k][t][i],d[i-1]))
                
        sub_func_c3 = Q**2
        
        for b in range(len(tmp)):
            sub_func_c3 += tmp[b][0] * ((tmp[b][1]**2) - 2*tmp[b][1]*Q)
        
        for b in range(len(tmp)-1):
            for bd in range(b+1,len(tmp)):
                sub_func_c3 += 2*tmp[b][0]*tmp[bd][0]*tmp[b][1]*tmp[bd][1]
        
        for t in range(1,T+1):
            for i in range(1,N):
                for m in range(math.ceil(np.log2(Q+1))):
                    sub_func_c3 += 2*xv[k][t][i]*d[i-1]*(2**m)*yv[k][m]
        
        for m in range(math.ceil(np.log2(Q+1))):
            sub_func_c3 -= 2*(2**m)*yv[k][m]*Q
        
        for m in range(math.ceil(np.log2(Q+1))):
            sub_func_c3 += (2**m)*(2**m)*yv[k][m]
        
        for m in range(math.ceil(np.log2(Q+1))-1):
            for md in range(m+1,math.ceil(np.log2(Q+1))):
                sub_func_c3 += 2*yv[k][m]*yv[k][md]*(2**m)*(2**md)
        
        func_c3 += sub_func_c3
    # print(func_c3)
    func = func_obj + p[0]*func_c1 + p[1]*func_c2 + p[2]*func_c3
    
    return func