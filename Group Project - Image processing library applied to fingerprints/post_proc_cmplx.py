# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:40:44 2023

@author: maxime
"""
import numpy as np
import matplotlib.pyplot as plt


def local_mean(values):
    N = len(values)
    return np.array(values).sum()/N



# Reading file
f = open("convo_compare.txt", 'r')
lines = f.readlines()
f.close()

# Initilizing accumulators
indices = []
times = []
times2 = []

#Filling raw data
for line in lines[1:]:
    l = line.split('\t')
    indices.append(int(l[0]))
    times.append(float(l[1]))
    times2.append(float(l[2]))

# Plot everything
plt.plot(np.array(indices), np.array(times),'-x', label="Classic convolution")
plt.plot(np.array(indices), np.array(times2), '-x', label="DFT convolution")

h = np.array([2**i for i in range(1,len(indices)+1)])
plt.plot(np.array(indices), h, '--',alpha=0.7, label=r"Theoric $\mathcal{O}(x^1)$")
plt.plot(np.array(indices), h**2, '--',alpha=0.7, label=r"Theoric $\mathcal{O}(x^2)$")


plt.xscale('log')
plt.yscale('log')
plt.xlabel("Side length (pixel)")
plt.ylabel("Running time (ms)")
plt.title("Convolution methods runtime (log-log scale)")
plt.legend()

plt.savefig("complexity_graphs/convolution_comparison_complexity.png", dpi=800)

plt.show()
# Plot everything
plt.plot(np.array(indices), np.array(times),'-x', label="Classic convolution")
plt.plot(np.array(indices), np.array(times2), '-x', label="DFT convolution")
plt.xlabel("Side length (pixel)")
plt.ylabel("Running time (ms)")
plt.title("Convolution methods runtime")
plt.legend()


"""
h = np.array([2**i for i in range(1,len(indices)+1)])
# Plot everything
plt.plot(np.array(indices), np.array(times),'-x', label="Computation time")
plt.plot(np.array(indices), h, '--',alpha=0.7, label=r"Theoric $\mathcal{O}(x^1)$")
plt.plot(np.array(indices), h**2, '--',alpha=0.7, label=r"Theoric $\mathcal{O}(x^2)$")
plt.xlabel("Side length (pixel)")
plt.ylabel("Running time (ms)")
plt.xscale('log')
plt.yscale('log')
plt.title("DFT convolution running time (log-log scale)")
plt.legend()"""

plt.savefig("complexity_graphs/convolution_comparison_runtime.png", dpi=800)

