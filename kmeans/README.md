## Classic kmeans code
Build a classic **kmeans** program to get the **clustering** results from raw two dimensional data, you can see the **variation** of the clustering center on the plot figure during the calculation.

### Packages
- numpy
- random
- matplotlib

### Important functions
- random.randint()
- random.sample()
- np.argmin()
- np.mean()
- plt.ion()
- plt.cla()
- plt.scatter(x,y,s,c)
- plt.pause()
- plt.show()

### Main process
- get the clustering center by random sampling.
- calculate the distance between the points from raw data and clustering center and group relevant points into specific clusters.
- recalculate the clustering center and show the center with raw data on the plot figure.
- redo the process from step 2 to step 3 until you find very little difference between the previous clustering center and new calculated clustering centers. 

### Dataset
We generate raw data by `random.randint()` method

### Run
You should create relevant data firstly with function `generate_data` and then get the clustering center with function `get_cluster_centers`