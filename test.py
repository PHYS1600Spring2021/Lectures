import timeit
import sys
import LogisticMap as lm

start = timeit.default_timer()
l = lm.LogisticMap()
l.iterate(5000000)
stop = timeit.default_timer()
total_time = stop - start
# output running time in a nice format.
sys.stdout.write("Regular: {} s \n".format(total_time))


start = timeit.default_timer()
lg = lm.LogisticMap()
lg.iterate_gen(5000000)

stop = timeit.default_timer()
total_time = stop - start
# output running time in a nice format.
sys.stdout.write("generator: {} s \n".format(total_time))

start = timeit.default_timer()
li = lm.LogisticMap_iter()
l = [next(li) for i in range(5000000)]
stop = timeit.default_timer()
total_time = stop - start

# output running time in a nice format.

sys.stdout.write("Iterator class: {} s \n".format(total_time))
