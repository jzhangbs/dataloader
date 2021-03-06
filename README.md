# Data Loader
This is a data loader for machine learning tasks. It can read
the images asynchronously and do preprocessing in parallel.
It also support feeding data for distributed learning.
Currently this loader is configured to load SceneFlow dataset.
You can easily move to other datasets.

## How to use
You first need to create an instance.
```
loader = Loader(data_root, dataset_name, val,ratio, preproc_args, sys_args).start()
```

The loader will scan the directory, launch the workers and
read the images on creation.

You can get the number of samples by
`loader.training_sample_size` and `loader.validation_sample_size`.
Get a mini batch by
```
train_batch = loader.get_batch(batch_size, 'train')
val_batch = loader.get_batch(batch_size, 'val')
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

They need to support dot access. You can use `dotdict` in
`loader.py`, which is a dictionary with dot access.

## Support for other data sets
You need to modify `gen_sample_list` and `read` which
perform data set specific operations.