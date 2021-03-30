import multiprocessing
from joblib import Parallel, delayed
from prototype import predicting_newtons_cooling_law

import time





def myfunction(num_list):

    total = 0
    for num in num_list:
        for nummm in range(20):
            for nummm in range(40):
                total += num
    print(total)
    return total

num_cores = multiprocessing.cpu_count()
inputs = [1,2,3,4,5,6]



if __name__ == "__main__":
    start = time.time()
    processed_list = Parallel(n_jobs=30)(delayed(predicting_newtons_cooling_law)("data/train/newtons_cooling_law.csv", True) for i in range(30))
    print("--- %s seconds ---" % (time.time() - start))
