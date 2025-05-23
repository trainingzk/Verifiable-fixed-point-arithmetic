import time
import random
import MatMul_renov as MatMul 
import Useful_functions as aux
from keras.datasets import mnist 
from Aggregated_range_proof import aggregated_range_proof 
import random 
import matplotlib.pyplot as plt 
import numpy as np


plt.rcParams.update({
    'font.family': 'Times New Roman',
    # 'font.size': 24,
    # 'axes.titlesize': 26,
    # 'axes.labelsize': 24,
    # 'xtick.labelsize': 16,
    # 'ytick.labelsize': 16,
    # 'legend.fontsize': 16
})

precision_list = [8, 8, 16, 16, 32, 32, 64, 64] # The number of bits to be removed (also known as s)
mantisa_list = [6, 14, 14, 30, 30, 62, 62, 126] # The number of bits in integer. We assume each number after multiplicaiton is representable in fixed-point mode

N_dim = 6 # Matrix dimension 
items = [12, 17, 15, 5, 8, 6, 18, 20] # Set the prime number and the group order 

prover_sum_check_times = [] 
verifier_sum_check_times = []
prover_range_proof_times = []  
verifier_range_proof_times = [] 
prover_MLE_times = []
verifier_MLE_times = []


for i in range(8): 
    print("******************************************** i = ", i) 
    [prime_number, order, factors, base_group] = aux.prime_order(items[i]) 
    precision = precision_list[i]  
    mantisa = mantisa_list[i]  
    bias = int((order - 1)/2)  
    # numbers are from -(prime_number - 1)/2 to +(prime_number - 1)/2. Thus we need bias in remainder calculations

    half_random_range = round((precision + mantisa - N_dim) / 2 - 0.5) 
    input_renov = [[random.randint(-half_random_range, half_random_range) for j in range(2**N_dim)] for k in range(2**N_dim)] 
    A_renov = [[random.randint(-half_random_range, half_random_range) for j in range(2**N_dim)] for k in range(2**N_dim)] 


    t1 = time.time() 
    B_renov = aux.matmul_list(input_renov, A_renov, order, bias) 
    V_error_1 = aux.error_vector(B_renov, precision) 
    C_renov = aux.round_vector(B_renov, V_error_1, precision, bias, order) 
    t2 = time.time() 
    print("Main task time = ", 1000 * (t2 - t1), "ms")    
    print("---------------------------------------------") 


    [MLE_v_time_GKR, MLE_p_time_GKR, MLE_v_time1, MLE_p_time1, p_time1, v_time1] = MatMul.Thaler_method(input_renov, A_renov, order, prime_number, factors, base_group) 
    [p_time2, v_time2] = aggregated_range_proof([V_error_1[0][i] + 2**(precision-1) for i in range(len(V_error_1[0]))], precision, order, prime_number, factors, base_group) 
    [p_time3, v_time3] = aggregated_range_proof([C_renov[0][i] + 2**(mantisa+1) for i in range(len(C_renov[0]))], mantisa+2, order, prime_number, factors, base_group) 


    print("Sum-check = ") 
    print("prover time = ", 1000 * p_time1, "ms") 
    print("verifier time = ", 1000 * v_time1, "ms") 
    print("---------------------------------------------") 

    print("Range proof = ") 
    print("prover time = ", 1000 * (p_time2 + p_time3), "ms") 
    print("verifier time = ", 1000 * (v_time2 + v_time3), "ms") 
    print("---------------------------------------------") 

    print("MLE = ") 
    print("prover time = ", 1000 * MLE_p_time1, "ms") 
    print("verifier time = ", 1000 * MLE_v_time1, "ms") 
    print("GKR prover time = ", 1000 * MLE_p_time_GKR, "ms") 
    print("GKR verifier time = ", 1000 * MLE_v_time_GKR, "ms") 

    prover_sum_check_times.append(p_time1)    
    verifier_sum_check_times.append(v_time1)
    prover_range_proof_times.append(p_time2 + p_time3)    
    verifier_range_proof_times.append(v_time2 + v_time3)
    prover_MLE_times.append(MLE_p_time1)    
    verifier_MLE_times.append(MLE_v_time1) 

labels = ["6,8", "14,8", "14,16", "30,16", "30,32", "62,32", "62,64", "126,64"] 


import pandas as pd
plotdata_p = pd.DataFrame({
    "Linear part":[prover_sum_check_times[i] + prover_MLE_times[i] for i in range(len(prover_MLE_times))],
    "Non-linear part":prover_range_proof_times}, 
    index=labels) 

plotdata_v = pd.DataFrame({
    "Linear part":[verifier_sum_check_times[i] + verifier_MLE_times[i] for i in range(len(verifier_MLE_times))],
    "Non-linear part":verifier_range_proof_times},
    index=labels) 

fig, axes = plt.subplots(nrows=1, ncols=2) 
plotdata_p.plot(kind='bar', stacked=False, ax=axes[0])  
axes[0].set_xlabel("Bit-length (int, frac)")
axes[0].set_ylabel("Prover Time (s)") 
axes[0].set_yscale("log")
plotdata_v.plot(kind='bar', stacked=False, ax=axes[1])  
axes[1].set_xlabel("Bit-length (int, frac)")
axes[1].set_ylabel("Verifier Time (s)") 
axes[1].set_yscale("log")

# Adjust the layout
plt.tight_layout()

# Save the plot to PDF (adjust the filename as needed)
plt.savefig('Figure_1.pdf', format='pdf')

plt.show() 

