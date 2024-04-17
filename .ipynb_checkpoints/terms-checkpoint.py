import sympy

def terms1(N,weights,wires_dict_b,wires_dict_a,t):
    func1 = 0
    for i in range(N-1):
        for j in range(N-1):
            if i!=j:
                func1 += weights[i+1][j+1]*wires_dict_b[t-1][i]*wires_dict_b[t][j]
    func1 *= (1-wires_dict_a[t])
    
    func2 = 0
    for i in range(N-1):
        func2 += weights[i+1][0]*wires_dict_b[t-1][i]
        func2 += weights[0][i+1]*wires_dict_b[t][i]
    func2 *= wires_dict_a[t]
    
    return func1 + func2


def terms2(N,weights,wires_dict_b):
    func = 0
    for i in range(N-1):
        func += weights[0][i+1]*wires_dict_b[0][i]
    return func


def terms3(N,weights,wires_dict_b):
    func = 0
    for i in range(N-1):
        func += weights[i+1][0]*wires_dict_b[N-2][i]
    return func


def obj_terms(weights,register_b,register_a):
    N = weights.shape[0]
    
    sympy_dict_b={}
    for t in range(N-1):
        sympy_dict_b[t] = {}
        for i in range(N-1):
            sympy_dict_b[t][i] = 0.5*(1-sympy.symbols(str(register_b[t][i])))
            
    sympy_dict_a={}
    for t in range(1,N-1):
        sympy_dict_a[t] = 0.5*(1-sympy.symbols(str(register_a[t])))
    
    d_terms = []
    
    func = terms2(N,weights,sympy_dict_b)
    d_terms.append(func)
    
    for t in range(1,N-1):
        func = terms1(N,weights,sympy_dict_b,sympy_dict_a,t)
        d_terms.append(func)
    
    func = terms3(N,weights,sympy_dict_b)
    d_terms.append(func)
    
    obj = 0
    for func in d_terms:
        obj += func

    return sympy.expand(obj).args