# Data Loader
This is a data loader for machine learning tasks. It can read
the images asynchronously and do preprocessing in parallel.
It also support feeding data for distributed learning.
Currently this loader is configured to load SceneFlow dataset.
You can easily move to other datasets.

## How to use
You can directly use this loader.
```
loader = Loader(data_root, dataset_name, val,ratio, preproc_args, sys_args).start()
```

You can also run `loader.py` to launch a server. The clients
can use `LoaderClient` to get the data.
```
loader = LoaderClient(addr, port, auth)
```

`preproc_args` can be customized. `sys_args` consists of
the following arguments.
- num_worker: Number of workers.
- queue_capacity: Capacity of the buffers.
- port: Port for connection. Only for server mode.
- auth: Password for connection. Only for server mode.

## Support for other data sets
You need to modify `gen_sample_list` and `read` which
perform data set specific operations.