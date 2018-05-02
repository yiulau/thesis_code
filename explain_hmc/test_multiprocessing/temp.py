# test putting a chain on each cpu.

import numpy,multiprocessing,time, pickle

class testclass(object):
    # simulates behaviour
    def simulate_chain(self,seed):
        numpy.random.seed(seed)
        #time.sleep(5)
        out = numpy.random.randn(5)
        return(out)


v_object = testclass()

num_cpu = multiprocessing.cpu_count()
agents = 2
#chunksize = 3
parallel_time = time.time()
with multiprocessing.Pool(processes=agents) as pool:
    result_parallel = pool.map(v_object.simulate_chain, range(10))

parallel_time = time.time() - parallel_time
print("parallel time {}".format(parallel_time))
print(result_parallel)

result_sequential = []
sequential_time = time.time()
for i in range(10):
    result_sequential.append(v_object.simulate_chain(i))

sequential_time = time.time() - sequential_time
print("sequential time {}".format(sequential_time))
print(result_sequential)



# how to dump data

with open('test_experimentdata.pkl', 'wb') as f:
    pickle.dump(result_sequential, f)

mod = pickle.load(open('test_experimentdata.pkl', 'rb'))

print(mod)
exit()
pool = multiprocessing.Pool(processes=2)
p1 = pool.apply_async(v_object.simulate_chain())
p1 = pool.apply_async(bar)

output = multiprocessing.Queue()
# Setup a list of processes that we want to run
processes = [multiprocessing.Process(target=rand_string, args=(5, x, output)) for x in range(4)]

# Run processes
for p in processes:
    p.start()

# Exit the completed processes
for p in processes:
    p.join()

# Get process results from the output queue
results = [output.get() for p in processes]


#out = v_object.simulate_chain(0)
#print(out)



