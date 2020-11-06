## Large numpy data loader
It's almost impossible to load all **large amounts of date** into memory **at once**, we try to **store** the data into a file and build a **data loader** to load a **batch** of data from the file each step when training and evaluating deep learning model.

### Packages
- numpy
- pickle
- tqdm
- os
- time
- warnings

### Important functions
- pickle.load()
- pickle.dump()
- np.random.randint()
- np.random.randn()
- x,y=zip(*data)
- np.stack()
- np.mean()
- yield()

### Special code
```python
with open(file,'rb') as f:
    total_samples = 0
    for index in range(1000000):
        try:
            data=pickle.load(f)
        except EOFError:
            break
        else:
            total_samples += 1
```

### Main process
- preprocess raw data and store them into the file step by step
- load a batch of data and feed them to the deep learning model.

### Dataset
We generate raw data by `np.random.randn()` and `np.random.randint()` method

### Run
You should create relevant data and store them into a file firstly with function `create_data` and then get batched data each time with function `get_loader`