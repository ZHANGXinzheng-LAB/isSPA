#!/usr/bin/env python3

import multiprocessing as mp

def f(b, q):
    s = b**2
    q.put(s)

def main():
    q = mp.Queue()
    results = []

    num_cores = mp.cpu_count() - 4
    with mp.Pool(processes=num_cores) as pool:
        pool.starmap(f, [(j, q) for j in range(1,21)])

    while not q.empty():
        results.append(q.get())

    print(results)

if __name__ == "__main__":
    main()