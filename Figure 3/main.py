import time
import random
import MatMul_renov as MatMul 
import Useful_functions as aux
# from keras.datasets import mnist 
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



# precision_list_ours = [32, 32, 32, 32, 32, 32] # The number of bits to be removed (also known as s)
# mantisa_list_ours = [2, 6, 14, 30, 62, 126] # The number of bits in integer. We assume each number after multiplicaiton is representable in fixed-point mode
precision = 32 
mantisa = 6
Depth_list = [1, 2, 3, 4, 5, 6] 

N_dim = 6 # Matrix dimension 
items_ours = [10, 10, 10, 10, 10] # Set the prime number and the group order 
prime_length_ours = [44, 44, 44, 44, 44]
items_thal = [10, 23, 30, 32, 33] # Set the prime number and the group order 
prime_length_thal = [44, 77, 117, 145, 177]

prover_sum_check_times = [] 
verifier_sum_check_times = []
prover_range_proof_times = []  
verifier_range_proof_times = [] 
prover_MLE_times = []
verifier_MLE_times = []
communications_sum_check = []
communications_range_proof = []
GKR_prover_sum_check_times = [] 
GKR_verifier_sum_check_times = []
GKR_prover_MLE_times = []
GKR_verifier_MLE_times = []
GKR_communications_sum_check = []


for j in range(2): 
    if j == 0: 
        items = items_ours          
    if j == 1: 
        items = items_thal   
    for i in range(len(items_ours)): 
        print("******************************************** i = ", i) 
        [prime_number, order, factors, base_group] = aux.prime_order(items[i])         
        bias = int((order - 1)/2)  
        # numbers are from -(prime_number - 1)/2 to +(prime_number - 1)/2. Thus we need bias in remainder calculations

        if j == 0: 
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


        [MLE_v_time_GKR, MLE_p_time_GKR, MLE_v_time1, MLE_p_time1, p_time1, v_time1, comm1, comm_GKR] = MatMul.Thaler_method(input_renov, A_renov, order, prime_number, factors, base_group) 
        if j == 0:
            [p_time2, v_time2, comm2] = aggregated_range_proof([V_error_1[0][i] + 2**(precision-1) for i in range(len(V_error_1[0]))], precision, order, prime_number, factors, base_group) 
            [p_time3, v_time3, comm3] = aggregated_range_proof([C_renov[0][i] + 2**(mantisa+1) for i in range(len(C_renov[0]))], mantisa+2, order, prime_number, factors, base_group) 

        if j == 0: # Ours
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
            print("---------------------------------------------") 

            print("Commumication = ") 
            print("Sum-check = ", prime_length_ours[i] * comm1, "bits") # precision + mantisa + 3 is the minimum p size.
            print("Range proof = ", prime_length_ours[i] * (comm2 + comm3), "bits")

            prover_sum_check_times.append(p_time1 * Depth_list[i])    
            verifier_sum_check_times.append(v_time1 * Depth_list[i])
            prover_range_proof_times.append((p_time2 + p_time3) * Depth_list[i])    
            verifier_range_proof_times.append((v_time2 + v_time3) * Depth_list[i])
            prover_MLE_times.append(MLE_p_time1 * Depth_list[i])    
            verifier_MLE_times.append(MLE_v_time1 * Depth_list[i]) 
            communications_sum_check.append(prime_length_ours[i] * comm1 * Depth_list[i])  
            communications_range_proof.append(prime_length_ours[i] * (comm2 + comm3) * Depth_list[i]) 

        if j == 1: # GKR or Thaler 
            print("Sum-check = ") 
            print("prover time = ", 1000 * p_time1, "ms") 
            print("verifier time = ", 1000 * v_time1, "ms") 
            print("---------------------------------------------") 

            print("MLE = ") 
            print("GKR prover time = ", 1000 * MLE_p_time_GKR, "ms") 
            print("GKR verifier time = ", 1000 * MLE_v_time_GKR, "ms") 
            print("---------------------------------------------") 

            print("Commumication = ") 
            print("Sum-check = ", prime_length_thal[i] * comm_GKR, "bits") # precision + mantisa + 3 is the minimum p size.

            GKR_prover_sum_check_times.append(p_time1 * Depth_list[i])     
            GKR_verifier_sum_check_times.append(v_time1 * Depth_list[i]) 
            GKR_prover_MLE_times.append(MLE_p_time1 * Depth_list[i])    
            GKR_verifier_MLE_times.append(MLE_v_time1 * Depth_list[i]) 
            GKR_communications_sum_check.append(prime_length_thal[i] * comm_GKR * Depth_list[i] + prime_length_thal[i] * (comm1 - comm_GKR))  


labels = ["1", "2", "3", "4", "5", "6", "7"] 
labels = labels[0:len(communications_sum_check)] 


import pandas as pd
plotdata_p = pd.DataFrame({
    "Proposed method":[prover_sum_check_times[i] + prover_range_proof_times[i] + prover_MLE_times[i] for i in range(len(prover_sum_check_times))],
    "SOTA":[GKR_prover_sum_check_times[i] + GKR_prover_MLE_times[i] for i in range(len(GKR_prover_sum_check_times))]}, 
    index=labels) 

plotdata_v = pd.DataFrame({
    "Proposed method":[verifier_sum_check_times[i] + verifier_range_proof_times[i] + verifier_MLE_times[i] for i in range(len(verifier_sum_check_times))],
    "SOTA":[GKR_verifier_sum_check_times[i] + GKR_verifier_MLE_times[i] for i in range(len(GKR_verifier_sum_check_times))]},
    index=labels) 

plotdata_c = pd.DataFrame({
    "Proposed method":[(communications_sum_check[i] + communications_range_proof[i]) / 8000 for i in range(len(communications_sum_check))],
    "SOTA":[(GKR_communications_sum_check[i]) / 8000 for i in range(len(GKR_communications_sum_check))]},
    index=labels) 

fig, axes = plt.subplots(nrows=1, ncols=3) 
fig.tight_layout() 
plotdata_p.plot(kind='bar', stacked=False, ax=axes[0])  
axes[0].set_xlabel("# of layers") 
axes[0].set_ylabel("Time (s)") 
# axes[0].set_yscale("log")  
axes[0].set_title("Prover") 
plotdata_v.plot(kind='bar', stacked=False, ax=axes[1])  
axes[1].set_xlabel("# of layers")
axes[1].set_ylabel("Time (s)")
# axes[1].set_yscale("log")
axes[1].set_title("Verifier") 
plotdata_c.plot(kind='bar', stacked=False, ax=axes[2])  
axes[2].set_xlabel("# of layers")
axes[2].set_ylabel("Communication Cost (KB)") 
# axes[2].set_yscale("log")
axes[2].set_title("Communication")

# Adjust the layout
plt.tight_layout()

# Save the plot to PDF (adjust the filename as needed)
plt.savefig('Figure_3_v1.pdf', format='pdf')

plt.show() 



