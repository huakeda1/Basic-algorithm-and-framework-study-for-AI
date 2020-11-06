## Basic classification model
We build a simple template classification model in this project, it will make you be more familiar with the basic structure of classification model, you can take a reference to build more complicated model later. 

### Packages
- numpy
- tensorflow
- os
- time

### Important functions
- np.load()
- tf.convert_to_tensor()
- tf.data.Dataset.from_tensor_slices()
- dataset.shuffle().batch()
- tf.keras.Sequential()
- tf.keras.layers.Dense()
- tf.train.Checkpoint()
- tf.GradientTape().gradient()
- optimizer.apply_gradients(zip())
- tf.keras.optimizers.schedules.LearningRateSchedule
- tf.keras.optimizers.Adam()
- tf.keras.losses.SparseCategoricalCrossentropy()
- tf.keras.metrics.SparseCategoricalAccuracy()
- tf.keras.metrics.Mean()

### Special code
```python
class ExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_rate, decay_steps, decay_rate):
        super().__init__()
        self.initial_rate = tf.cast(initial_rate, dtype=tf.float32)
        self.decay_steps = tf.cast(decay_steps, dtype=tf.float32)
        self.decay_rate = tf.cast(decay_rate, dtype=tf.float32)

    def __call__(self, step):
        return self.initial_rate * self.decay_rate ** (step / self.decay_steps)
    
def train_step(inputs, labels, model, optimizer, loss_func, train_acc, train_loss):
    with tf.GradientTape() as tape:
        inputs = tf.reshape(inputs, (-1, 28 * 28))
        predictions = model(inputs)
        loss = loss_func(labels, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(loss)
    train_acc(labels, predictions)
```

### Main process
- get raw data and normalize them to specific range(-1~1)
- build data loader
- create the model
- train and evaluate the model.

### Dataset
We get data from `mnist.npz` by `np.load()` and normalize them to standard format with scaling method, we use `tf.convert_to_tensor` to convert the data to what the model is expecting.

### Model
We stack many dense layers with activation `relu` together to build a simple template classification model.

### Run
You can run the code directly to train, save and evaluate the deep learning model.