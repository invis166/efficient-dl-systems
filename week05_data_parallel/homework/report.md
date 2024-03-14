# Experimental setup

* torch version: 2.1.2
* GPU: 2x nvidia RTX A4000
* number of processes by default is 2 if other is not said

# Training loop benchmark
Selfmade distributed training
| Rank | Time                       | Max Memory Allocated |
|------|----------------------------|----------------------|
| 0    | 29.588 seconds | 170099200 bytes      |
| 1    | 29.632 seconds | 170098688 bytes      |
loss: 3.2136, acc: 0.2734375


pytorch Distributed Data Parallel
| Rank | Time                     | Max Memory Allocated |
|------|--------------------------|----------------------|
| 0    | 29.227 seconds | 173903360 bytes     |
| 1    | 29.279 seconds | 174062592 bytes     |
loss: 3.2149, acc: 0.27734375


how to reproduce pytoch version: `CUDA_VISIBLE_DEVICES=X,Y torchrun --nproc_per_node 2 ddp_cifar100.py --num-epochs 16 --grad-accum-steps 4 --distributed-impl selfmade`

how to reproduce selfmade version: `CUDA_VISIBLE_DEVICES=X,Y torchrun --nproc_per_node 2 ddp_cifar100.py --num-epochs 16 --grad-accum-steps 4 --distributed-impl pytorch`

In practice, those results (especially loss and acc) are not reproducible, each run coverges to a different optimum, alhtough the seed is fixed. Anyway, on a average both implementations converges near to the same result. Performance is the same and pytorch has a slightly larger memory consumtion.

# SyncBatchNorm benchmark
10000 runs for each parameter set

Pytorch BN
| batch_size | hidden_dim | num_workers | median time | max memory allocated |
|------------|------------|-------------|-------------|----------------------|
| 32         | 128        | 2           | 0.00052     | 41984                |
| 32         | 256        | 2           | 0.00047     | 80896                |
| 32         | 512        | 2           | 0.00053     | 158720               |
| 32         | 1024       | 2           | 0.00047     | 314368               |
| 64         | 128        | 2           | 0.00045     | 74752                |
| 64         | 256        | 2           | 0.00046     | 146432               |
| 64         | 512        | 2           | 0.00048     | 289792               |
| 64         | 1024       | 2           | 0.00048     | 576512               |


Selfmade BN
| batch_size | hidden_dim | num_workers | median time | max memory allocated |
|------------|------------|-------------|-------------|----------------------|
| 32         | 128        | 2           | 0.00037     | 101888               |
| 32         | 256        | 2           | 0.00046     | 203264               |
| 32         | 512        | 2           | 0.00037     | 406016               |
| 32         | 1024       | 2           | 0.00040     | 811520               |
| 64         | 128        | 2           | 0.00036     | 200192               |
| 64         | 256        | 2           | 0.00036     | 399872               |
| 64         | 512        | 2           | 0.00039     | 799232               |
| 64         | 1024       | 2           | 0.00038     | 1597952              |

to reproduce: `CUDA_VISIBLE_DEVICES=X,Y python syncbn_benchmark.py`

in general, custom implementation is turned out to be faster but is has larger memory consumption.

# Ring all reduce

Comprasion with given butterfly all reduce

| world_size | tensor_size | ring_max_time_consumed | butterfly_max_time_consumed | ring_max_memory | butterfly_max_memory | diff_ring | diff_butterfly |
|------------|-------------|-----------------------|-----------------------------|----------------|----------------------|-----------|------------|
| 1 | 1 | 0.017 | 0.012 | 379424.0 | 379424.0 | 0.0000 | 0.0000 |
| 2 | 2 | 0.024 | 0.026 | 380787.2 | 380966.4 | 0.0000 | 0.0000 |
| 4 | 4 | 0.026 | 0.027 | 380989.6 | 380996.0 | 2.3592e-16 | 0.0000 |
| 8 | 8 | 0.034 | 0.031 | 382106.8 | 382218.8 | 2.5535e-15 | 1.6653e-15 |
| 16 | 16 | 0.043 | 0.036 | 382266.0 | 382289.6 | 1.8666e-14 | 5.0515e-15 |
| 32 | 32 | 0.057 | 0.048 | 382398.4 | 382439.6 | 3.7367e-14 | 1.6990e-14 |

Comprasion with pytorch all reduce

| world_size | tensor_size | ring_max_time_consumed | torch_max_time_consumed | ring_max_memory | torch_max_memory | diff_ring | diff_torch |
|------------|-------------|-----------------------|------------------------|----------------|-----------------|-----------|------------|
| 1 | 1024 | 0.015 | 0.020 | 382460.0 | 382460.0 | 0.0000 | 0.0000 |
| 2 | 2048 | 0.028 | 0.025 | 382460.0 | 382460.0 | 0.0000 | 0.0000 |
| 4 | 4096 | 0.024 | 0.026 | 382523.2 | 382586.4 | 2.4260e-12 | 2.7529e-12 |
| 8 | 16384 | 0.030 | 0.032 | 384578.4 | 385378.8 | 1.0071e-11 | 1.1579e-11 |
| 16 | 131072 | 0.359 | 0.258 | 405353.2 | 410945.6 | 7.6576e-11 | 8.7605e-11 |

to reproduce: `python allredce.py`