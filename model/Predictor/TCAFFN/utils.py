import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def multi_order(s_out, order_0, n):
        solves = []
        stats = []
        for i in range(1,int(n)):
            c = 6*(order_0**i)
            m = n-i
            order_low = (s_out/c)**(1/(m+1)) 
            order_up = (s_out/c)**(1/m)
            order_1 = order_up//1
            if (not ((order_1 <= order_up) and (order_1 > order_low))) or (order_1 == 1):
                continue
            else:
                solves.append([order_0,order_1,i,m])
                stats.append(order_0*i+order_1*m)
        idx = np.argmin(stats)
        solves = solves[idx]
        order_list = []
        for i in range(int(n)):
            idx = np.argmax(solves[2:4])
            order_list.append(int(solves[idx]))
            solves[2+idx] -= 1
        return order_list


def calculate_order(c_in, s_in, s_out, order_in, order_out):
    n_in = (np.log(s_in)/np.log(order_in))//1 #5
    order_out_low = (s_out/c_in)**(1/(1+n_in))
    order_out_up = (s_out/c_in)**(1/(n_in))
    order_out = order_out_up//1
    n_out = (np.log(s_out/2)/np.log(order_out))//1
    if (not ((order_out <= order_out_up) and (order_out > order_out_low))) or (order_out == 1):
        Warning('Order {} is not good for s_in, s_out')
        order_out_list = multi_order(s_out, order_out, n_in)
    else:
        order_out_list = [int(order_out)]*int(n_out)
    order_in_list = [int(order_in)]*int(n_in)
    return int(n_in), order_in_list, order_out_list

