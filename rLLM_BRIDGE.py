import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import numpy as np
from sklearn.metrics import accuracy_score
from typing import Tuple, Generator
import random

# Constants
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 1000
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

# Data Engine Layer
class BaseTable:
    def __init__(self, user_data: np.ndarray, movie_data: np.ndarray, rating_data: np.ndarray):
        self.user_data = user_data
        self.movie_data = movie_data
        self.rating_data = rating_data
    
    def _generator(self) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
        while True:
            idx = random.randint(0, len(self.user_data) - 1)
            user = self.user_data[idx]
            movie = random.choice(self.movie_data)
            rating = random.choice(self.rating_data)
            yield user, movie, rating
    
    def get_dataset(self) -> tf.data.Dataset:
        return tf.data.Dataset.from_generator(
            self._generator,
            output_signature=(
                tf.TensorSpec(shape=(5,), dtype=tf.float32),
                tf.TensorSpec(shape=(11,), dtype=tf.float32),
                tf.TensorSpec(shape=(4,), dtype=tf.float32)
            )
        ).batch(BATCH_SIZE).shuffle(SHUFFLE_BUFFER_SIZE)

class BaseGraph:
    def __init__(self, graph_data: np.ndarray, edge_index: np.ndarray):
        self.graph_data = graph_data
        self.edge_index = edge_index

# Module Layer
class GraphTransform(layers.Layer):
    def call(self, x: tf.Tensor, edge_index: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return tf.nn.l2_normalize(x, axis=1), tf.cast(edge_index, tf.int32)

class GraphConv(layers.Layer):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.conv = layers.Dense(out_features)
    
    def call(self, x: tf.Tensor, edge_index: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(x)[0]
        row, col = edge_index
        row = tf.clip_by_value(row, 0, batch_size - 1)
        col = tf.clip_by_value(col, 0, batch_size - 1)
        x_sum = tf.gather(x, row) + tf.gather(x, col)
        return self.conv(x_sum)

class GraphEncoder(models.Model):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.transform = GraphTransform()
        self.conv = GraphConv(in_features, out_features)
    
    def call(self, x: tf.Tensor, edge_index: tf.Tensor) -> tf.Tensor:
        x, edge_index = self.transform(x, edge_index)
        return self.conv(x, edge_index)

class TableTransform(layers.Layer):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = layers.Dense(output_dim, activation='relu')
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.fc(x)

class TableConv(layers.Layer):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = layers.Dense(output_dim, activation='relu')
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.fc(x)

# Model Layer
class BRIDGE(models.Model):
    def __init__(self, table_encoder: models.Model, graph_encoder: models.Model):
        super().__init__()
        self.table_encoder = table_encoder
        self.graph_encoder = graph_encoder
    
    def call(self, table_data: tf.Tensor, graph_data: tf.Tensor, edge_index: tf.Tensor) -> tf.Tensor:
        table_embeddings = self.table_encoder(table_data)
        graph_embeddings = self.graph_encoder(graph_data, edge_index)
        graph_embeddings = tf.reduce_mean(graph_embeddings, axis=1, keepdims=True)
        graph_embeddings = tf.tile(graph_embeddings, [1, tf.shape(table_embeddings)[0], 1])
        graph_embeddings = tf.reshape(graph_embeddings, [tf.shape(table_embeddings)[0], -1])
        return tf.concat([table_embeddings, graph_embeddings], axis=1)

# Utility functions
@tf.function
def train_step(model: BRIDGE, table_data: tf.Tensor, graph_data: tf.Tensor, edge_index: tf.Tensor, 
               optimizer: optimizers.Optimizer, loss_fn: losses.Loss) -> tf.Tensor:
    with tf.GradientTape() as tape:
        outputs = model(table_data, graph_data, edge_index)
        labels = tf.random.uniform(shape=(tf.shape(outputs)[0],), maxval=7, dtype=tf.int32)
        loss = loss_fn(labels, outputs)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def train_model(model: BRIDGE, dataset: tf.data.Dataset, optimizer: optimizers.Optimizer, 
                loss_fn: losses.Loss, graph_data: np.ndarray, edge_index: np.ndarray, num_epochs: int = NUM_EPOCHS):
    for epoch in range(num_epochs):
        epoch_loss = 0
        for user, movie, rating in dataset:
            table_data = tf.concat([user, movie, rating], axis=1)
            valid_indices = tf.where(user[:, 0] >= 0)
            valid_user_indices = tf.gather(user[:, 0], valid_indices)
            graph_data_batch = tf.gather(graph_data, tf.cast(valid_user_indices, tf.int32))
            loss = train_step(model, table_data, graph_data_batch, edge_index, optimizer, loss_fn)
            epoch_loss += loss.numpy()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(list(dataset))}")

def evaluate_model(model: BRIDGE, dataset: tf.data.Dataset, graph_data: np.ndarray, edge_index: np.ndarray):
    all_labels, all_preds = [], []
    for user, movie, rating in dataset:
        table_data = tf.concat([user, movie, rating], axis=1)
        valid_indices = tf.where(user[:, 0] >= 0)
        valid_user_indices = tf.gather(user[:, 0], valid_indices)
        graph_data_batch = tf.gather(graph_data, tf.cast(valid_user_indices, tf.int32))
        outputs = model(table_data, graph_data_batch, edge_index)
        labels = tf.random.uniform(shape=(tf.shape(outputs)[0],), maxval=7, dtype=tf.int32)
        preds = tf.argmax(outputs, axis=1)
        all_labels.extend(labels.numpy())
        all_preds.extend(preds.numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Evaluation Accuracy: {accuracy}")

# Example Usage
if __name__ == "__main__":
    user_data = np.random.randn(6040, 5).astype(np.float32)
    movie_data = np.random.randn(3883, 11).astype(np.float32)
    rating_data = np.random.randn(1000209, 4).astype(np.float32)
    graph_data = np.random.randn(6040, 16).astype(np.float32)
    edge_index = np.random.randint(0, 6040, (2, 10000))

    dataset = BaseTable(user_data, movie_data, rating_data).get_dataset()
    table_transform = TableTransform(20, 32)
    table_encoder = models.Sequential([table_transform, TableConv(32, 64)])
    graph_encoder = GraphEncoder(16, 32)
    model = BRIDGE(table_encoder, graph_encoder)
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)

    train_model(model, dataset, optimizer, loss_fn, graph_data, edge_index, num_epochs=NUM_EPOCHS)
    evaluate_model(model, dataset, graph_data, edge_index)