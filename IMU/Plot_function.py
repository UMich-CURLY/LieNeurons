import torch 
import matplotlib.pyplot as plt

def plt3dvec(x, data, title = "Data", data_label_1 = "x", data_label_2 = "y", data_label_3 = "z"):
    fig, ax = plt.subplots(3,1)
    ax[0].plot(x, data[:,0], label=data_label_1)
    ax[1].plot(x, data[:,1], label=data_label_2)
    ax[2].plot(x, data[:,2], label=data_label_3)
    
    ax[0].set_title(title)
    ax[2].set_xlabel('time (s)')
    
    ax[0].set_ylabel(data_label_1)
    ax[1].set_ylabel(data_label_2)
    ax[2].set_ylabel(data_label_3)
    plt.legend()
    plt.show(block=False)