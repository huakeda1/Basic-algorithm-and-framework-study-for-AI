## Basic regression model
We build a simple template regression model in this project, it will make you be more familiar with the basic structure of regression model, you can take a reference to build more complicated model later. 

### Packages
- numpy
- tensorflow
- os
- time
- sklearn
- pandas

### Important functions
- np.loadtxt(delimiter=',',dtype=np.float32)
- sklearn.model_selection.train_test_split()
- sklearn.preprocessing.StardardScaler().fit_transform()
- sklearn.preprocessing.StardardScaler().transform()
- tf.data.Dataset.from_tensor_slices()
- dataset.shuffle().batch()
- tf.keras.layers.Layer
- tf.keras.layers.Activation()
- tf.keras.Sequential()
- tf.keras.layers.Dense()
- tf.train.Checkpoint()
- tf.GradientTape().gradient()
- optimizer.apply_gradients(zip())
- tf.keras.optimizers.schedules.LearningRateSchedule
- tf.keras.optimizers.Adam()
- tf.keras.losses.MeanSquareError()
- tf.keras.metrics.MeanAbsoluteError()
- tf.keras.metrics.Mean()
- history=model.fit()
- pandas.DataFrame(history.history).plot(figsize=(8,5))

### Special code
```python    
def train_step(inputs, labels, model, optimizer, loss_func, train_loss, l2_loss_coeff=0.0001):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss_pure = loss_func(labels, predictions)
        loss_regularization = []
        # l2 regularization
        for p in model.trainable_variables:
            loss_regularization.append(tf.nn.l2_loss(p))
        loss_regularization = tf.reduce_sum(tf.stack(loss_regularization))
        loss =loss_pure + l2_loss_coeff*loss_regularization
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(loss_pure)
    
def plot_learning_curve(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()
```

### Main process
- get raw data and normalize them to specific range(-1~1)
- build data loader
- create the model
- train and evaluate the model.

### Dataset
We get data from `cal_housing.data` by `np.loadtxt()` and normalize them to standard format with StandardScaler, we use `tf.data.Dataset.from_tensor_slices` to build a data loader.

### Model
We stack many dense layers with activation `relu` together to build a simple template regression model.

### Run
You can run the code directly to train, save and evaluate the template regression model.