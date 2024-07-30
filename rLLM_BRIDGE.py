import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import numpy as np
from sklearn.metrics import accuracy_score
from typing import Tuple, Dict, Any
import random
from dataclasses import dataclass
from absl import app, flags, logging
import os
import json

# Define flags for command-line arguments
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 64, 'Batch size for training')
flags.DEFINE_integer('shuffle_buffer', 1000, 'Buffer size for dataset shuffling')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate')
flags.DEFINE_integer('num_epochs', 10, 'Number of training epochs')
flags.DEFINE_string('model_dir', 'model_checkpoints', 'Directory to save model checkpoints')
flags.DEFINE_string('data_dir', 'data', 'Directory containing input data')
flags.DEFINE_boolean('use_tpu', False, 'Whether to use TPU for training')
flags.DEFINE_string('tpu_name', None, 'Name of the TPU to use')

# Data structures
@dataclass
class DataConfig:
    user_dim: int = 5
    movie_dim: int = 11
    rating_dim: int = 4
    graph_dim: int = 16

# Data Engine Layer
class BaseTable:
    def __init__(self, user_data: tf.Tensor, movie_data: tf.Tensor, rating_data: tf.Tensor):
        self.user_data = user_data
        self.movie_data = movie_data
        self.rating_data = rating_data
    
    def _generator(self):
        while True:
            idx = tf.random.uniform((), 0, tf.shape(self.user_data)[0], dtype=tf.int32)
            user = self.user_data[idx]
            movie = tf.random.shuffle(self.movie_data)[0]
            rating = tf.random.shuffle(self.rating_data)[0]
            yield user, movie, rating
    
    def get_dataset(self) -> tf.data.Dataset:
        return tf.data.Dataset.from_generator(
            self._generator,
            output_signature=(
                tf.TensorSpec(shape=(DataConfig.user_dim,), dtype=tf.float32),
                tf.TensorSpec(shape=(DataConfig.movie_dim,), dtype=tf.float32),
                tf.TensorSpec(shape=(DataConfig.rating_dim,), dtype=tf.float32)
            )
        ).batch(FLAGS.batch_size).shuffle(FLAGS.shuffle_buffer).prefetch(tf.data.AUTOTUNE)

class BaseGraph:
    def __init__(self, graph_data: tf.Tensor, edge_index: tf.Tensor):
        self.graph_data = graph_data
        self.edge_index = edge_index

# Module Layer
class GraphTransform(layers.Layer):
    def call(self, x: tf.Tensor, edge_index: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return tf.nn.l2_normalize(x, axis=1), edge_index

class GraphConv(layers.Layer):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.conv = layers.Dense(out_features, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
    
    def call(self, x: tf.Tensor, edge_index: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(x)[0]
        row, col = tf.unstack(edge_index, axis=0)
        row = tf.clip_by_value(row, 0, batch_size - 1)
        col = tf.clip_by_value(col, 0, batch_size - 1)
        x_sum = tf.gather(x, row) + tf.gather(x, col)
        return self.conv(x_sum)

class GraphEncoder(models.Model):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.transform = GraphTransform()
        self.conv1 = GraphConv(in_features, out_features * 2)
        self.conv2 = GraphConv(out_features * 2, out_features)
        self.dropout = layers.Dropout(0.5)
    
    def call(self, x: tf.Tensor, edge_index: tf.Tensor, training: bool = False) -> tf.Tensor:
        x, edge_index = self.transform(x, edge_index)
        x = self.conv1(x, edge_index)
        x = self.dropout(x, training=training)
        return self.conv2(x, edge_index)

class TableEncoder(models.Model):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.dense1 = layers.Dense(hidden_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dense2 = layers.Dense(output_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dropout = layers.Dropout(0.5)
        self.batch_norm = layers.BatchNormalization()
    
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.batch_norm(x, training=training)
        return self.dense2(x)

# Model Layer
class BRIDGE(models.Model):
    def __init__(self, table_encoder: models.Model, graph_encoder: models.Model, output_dim: int = 7):
        super().__init__()
        self.table_encoder = table_encoder
        self.graph_encoder = graph_encoder
        self.final_dense = layers.Dense(output_dim)
    
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor], training: bool = False) -> tf.Tensor:
        table_data, graph_data, edge_index = inputs
        table_embeddings = self.table_encoder(table_data, training=training)
        graph_embeddings = self.graph_encoder(graph_data, edge_index, training=training)
        graph_embeddings = tf.reduce_mean(graph_embeddings, axis=1, keepdims=True)
        graph_embeddings = tf.tile(graph_embeddings, [1, tf.shape(table_embeddings)[0], 1])
        graph_embeddings = tf.reshape(graph_embeddings, [tf.shape(table_embeddings)[0], -1])
        combined = tf.concat([table_embeddings, graph_embeddings], axis=1)
        return self.final_dense(combined)

# Training and evaluation functions
@tf.function
def train_step(model: BRIDGE, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor], 
               optimizer: optimizers.Optimizer, loss_fn: losses.Loss) -> Tuple[tf.Tensor, tf.Tensor]:
    table_data, graph_data, edge_index = inputs
    with tf.GradientTape() as tape:
        outputs = model((table_data, graph_data, edge_index), training=True)
        labels = tf.random.uniform(shape=(tf.shape(outputs)[0],), maxval=7, dtype=tf.int32)
        loss = loss_fn(labels, outputs)
        loss += sum(model.losses)  # Add regularization losses
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(outputs, axis=1), labels), tf.float32))
    return loss, accuracy

def train_model(model: BRIDGE, dataset: tf.data.Dataset, optimizer: optimizers.Optimizer, 
                loss_fn: losses.Loss, graph_data: tf.Tensor, edge_index: tf.Tensor, 
                num_epochs: int = FLAGS.num_epochs) -> Dict[str, Any]:
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.Mean()
        
        for user, movie, rating in dataset:
            table_data = tf.concat([user, movie, rating], axis=1)
            valid_indices = tf.where(user[:, 0] >= 0)
            valid_user_indices = tf.gather(user[:, 0], valid_indices)
            graph_data_batch = tf.gather(graph_data, tf.cast(valid_user_indices, tf.int32))
            loss, accuracy = train_step(model, (table_data, graph_data_batch, edge_index), optimizer, loss_fn)
            epoch_loss_avg.update_state(loss)
            epoch_accuracy.update_state(accuracy)
        
        history['loss'].append(epoch_loss_avg.result().numpy())
        history['accuracy'].append(epoch_accuracy.result().numpy())
        
        # Validation step
        val_loss, val_accuracy = evaluate_model(model, dataset, graph_data, edge_index)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        logging.info(f"Epoch {epoch+1}, Loss: {epoch_loss_avg.result():.4f}, Accuracy: {epoch_accuracy.result():.4f}, "
                     f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Save checkpoint
        if epoch % 5 == 0:
            model.save_weights(os.path.join(FLAGS.model_dir, f'checkpoint_epoch_{epoch}'))
    
    return history

def evaluate_model(model: BRIDGE, dataset: tf.data.Dataset, graph_data: tf.Tensor, edge_index: tf.Tensor) -> Tuple[float, float]:
    loss_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.Mean()
    loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)
    
    for user, movie, rating in dataset:
        table_data = tf.concat([user, movie, rating], axis=1)
        valid_indices = tf.where(user[:, 0] >= 0)
        valid_user_indices = tf.gather(user[:, 0], valid_indices)
        graph_data_batch = tf.gather(graph_data, tf.cast(valid_user_indices, tf.int32))
        outputs = model((table_data, graph_data_batch, edge_index), training=False)
        labels = tf.random.uniform(shape=(tf.shape(outputs)[0],), maxval=7, dtype=tf.int32)
        loss = loss_fn(labels, outputs)
        loss_avg.update_state(loss)
        accuracy.update_state(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(outputs, axis=1), labels), tf.float32)))
    
    return loss_avg.result().numpy(), accuracy.result().numpy()

def load_data(data_dir: str) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    user_data = tf.convert_to_tensor(np.load(os.path.join(data_dir, 'user_data.npy')), dtype=tf.float32)
    movie_data = tf.convert_to_tensor(np.load(os.path.join(data_dir, 'movie_data.npy')), dtype=tf.float32)
    rating_data = tf.convert_to_tensor(np.load(os.path.join(data_dir, 'rating_data.npy')), dtype=tf.float32)
    graph_data = tf.convert_to_tensor(np.load(os.path.join(data_dir, 'graph_data.npy')), dtype=tf.float32)
    edge_index = tf.convert_to_tensor(np.load(os.path.join(data_dir, 'edge_index.npy')), dtype=tf.int32)
    return user_data, movie_data, rating_data, graph_data, edge_index

# Main execution
def main(argv):
    del argv  # Unused

    # Set up logging
    logging.set_verbosity(logging.INFO)

    # Load data
    user_data, movie_data, rating_data, graph_data, edge_index = load_data(FLAGS.data_dir)

    # Create dataset
    dataset = BaseTable(user_data, movie_data, rating_data).get_dataset()

    # Create model
    config = DataConfig()
    table_encoder = TableEncoder(config.user_dim + config.movie_dim + config.rating_dim, 32, 64)
    graph_encoder = GraphEncoder(config.graph_dim, 32)
    model = BRIDGE(table_encoder, graph_encoder)

    # Set up optimizer and loss function
    optimizer = optimizers.Adam(learning_rate=FLAGS.learning_rate)
    loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)

    # Set up TPU strategy if using TPU
    if FLAGS.use_tpu:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu_name)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        with strategy.scope():
            model = BRIDGE(table_encoder, graph_encoder)
            optimizer = optimizers.Adam(learning_rate=FLAGS.learning_rate)
            loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)

    # Train model
    history = train_model(model, dataset, optimizer, loss_fn, graph_data, edge_index, num_epochs=FLAGS.num_epochs)

    # Save training history
    with open(os.path.join(FLAGS.model_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)

    # Final evaluation
    test_loss, test_accuracy = evaluate_model(model, dataset, graph_data, edge_index)
    logging.info(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Save final model
    model.save(os.path.join(FLAGS.model_dir, 'final_model'))

if __name__ == "__main__":
    app.run(main)