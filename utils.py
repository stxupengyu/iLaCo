import time
import numpy as np
import matplotlib.pyplot as plt
import os
plt.style.use('seaborn-notebook')

def loss_display(clean_stat, name, args):
    #density plot
    plt.figure(figsize=(6,4))
    plt.hist(clean_stat, bins=20, density=False)
    plt.grid()
    plt.xlabel('Loss',fontsize='16')
    plt.ylabel('Frequency',fontsize='16')
    plt.title(name)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    # plt.legend(loc="upper left",ncol=1,fontsize=15)
    # plt.show()
    save_path = os.path.join(args.plot_dir, '%s.png'%name)
    plt.savefig(save_path)
    print('plot saved to %s'%save_path)
    plt.clf()

def loss_display2(clean_stat, noise_stat, epoch, args):
    #density plot
    plt.figure(figsize=(6,4))
    plt.hist([clean_stat, noise_stat], bins=20, density=False, label=['Clean Loss', 'Noise Loss'])
    plt.xlabel('Loss')
    plt.ylabel('Density')
    plt.title('Distribution of Loss')
    plt.legend()
    # plt.show()
    # np.save(save_path.replace('png', 'npy'), np.array(loss_collector))
    save_path = os.path.join(args.plot_dir, '%s_stat_epoch%d.png'%(args.mode, epoch))
    plt.savefig(save_path)
    print('plot saved to %s'%save_path)
    plt.clf()

def normalize_array(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def time_since(start_time):
    #adaptive return hours, minutes or seconds
    seconds = int(time.time()-start_time)
    if seconds < 60:
        return '%d seconds'%(seconds)
    elif seconds < 3600:
        return '%d minutes'%(seconds//60)
    else:
        return '%d hours'%(seconds//3600)

def cprint(name, value):
    #print a varible clearly
    print('=============%s=============='%name)
    print(value)
    print('=============%s=============='%name)

def ndarray_memory(ndarray_name, ndarray):
    #get the memory of a ndarrat
    memory_in_bytes = np.array(ndarray).nbytes  
    memory_in_gb = memory_in_bytes / (1024 ** 3)  
    print(ndarray_name + ": Memory occupied by ndarray:", round(memory_in_gb, 3), "GB")


