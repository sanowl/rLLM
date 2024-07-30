import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, losses
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from dataclasses import dataclass
from absl import app, flags, logging
import os
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime

# Define flags for command-line arguments
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 64, 'Batch size for training')
flags.DEFINE_integer('shuffle_buffer', 1000, 'Buffer size for dataset shuffling')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate')
flags.DEFINE_integer('num_epochs', 10, 'Number of training epochs')
flags.DEFINE_string('model_dir', 'model_checkpoints', 'Directory to save model checkpoints')
flags.DEFINE_boolean('use_tpu', False, 'Whether to use TPU for training')
flags.DEFINE_string('tpu_name', None, 'Name of the TPU to use')
flags.DEFINE_float('dropout_rate', 0.5, 'Dropout rate for neural network layers')
flags.DEFINE_integer('early_stopping_patience', 5, 'Number of epochs with no improvement after which training will be stopped')
flags.DEFINE_boolean('use_mixed_precision', False, 'Whether to use mixed precision training')

# Data structures
@dataclass
class DataConfig:
    user_dim: int = 5
    movie_dim: int = 11
    rating_dim: int = 4
    graph_dim: int = 16
    num_users: int = 1000
    num_movies: int = 500
    num_ratings: int = 10000
    num_graph_nodes: int = 1000

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
    
    def get_dataset(self, batch_size: int, shuffle_buffer: int) -> tf.data.Dataset:
        return tf.data.Dataset.from_generator(
            self._generator,
            output_signature=(
                tf.TensorSpec(shape=(DataConfig.user_dim,), dtype=tf.float32),
                tf.TensorSpec(shape=(DataConfig.movie_dim,), dtype=tf.float32),
                tf.TensorSpec(shape=(DataConfig.rating_dim,), dtype=tf.float32)
            )
        ).batch(batch_size).shuffle(shuffle_buffer).prefetch(tf.data.AUTOTUNE)

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
    def __init__(self, in_features: int, out_features: int, dropout_rate: float):
        super().__init__()
        self.transform = GraphTransform()
        self.conv1 = GraphConv(in_features, out_features * 2)
        self.conv2 = GraphConv(out_features * 2, out_features)
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, x: tf.Tensor, edge_index: tf.Tensor, training: bool = False) -> tf.Tensor:
        x, edge_index = self.transform(x, edge_index)
        x = self.conv1(x, edge_index)
        x = self.dropout(x, training=training)
        return self.conv2(x, edge_index)

class TableEncoder(models.Model):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_rate: float):
        super().__init__()
        self.dense1 = layers.Dense(hidden_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dense2 = layers.Dense(output_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dropout = layers.Dropout(dropout_rate)
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
    
    predictions = tf.cast(tf.argmax(outputs, axis=1), tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
    return loss, accuracy

class BRIDGETrainer:
    def __init__(self, model: BRIDGE, dataset: tf.data.Dataset, optimizer: optimizers.Optimizer, 
                 loss_fn: losses.Loss, graph_data: tf.Tensor, edge_index: tf.Tensor):
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.graph_data = graph_data
        self.edge_index = edge_index
        self.history: Dict[str, List[float]] = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        for epoch in range(num_epochs):
            if self._train_epoch(epoch):
                logging.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        return self.history

    def _train_epoch(self, epoch: int) -> bool:
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.Mean()
        
        for user, movie, rating in self.dataset:
            table_data = tf.concat([user, movie, rating], axis=1)
            valid_indices = tf.where(user[:, 0] >= 0)
            valid_user_indices = tf.gather(user[:, 0], valid_indices)
            graph_data_batch = tf.gather(self.graph_data, tf.cast(valid_user_indices, tf.int32))
            loss, accuracy = train_step(self.model, (table_data, graph_data_batch, self.edge_index), self.optimizer, self.loss_fn)
            epoch_loss_avg.update_state(loss)
            epoch_accuracy.update_state(accuracy)
        
        self.history['loss'].append(epoch_loss_avg.result().numpy())
        self.history['accuracy'].append(epoch_accuracy.result().numpy())
        
        val_loss, val_accuracy = self.evaluate()
        self.history['val_loss'].append(val_loss)
        self.history['val_accuracy'].append(val_accuracy)
        
        logging.info(f"Epoch {epoch+1}, Loss: {epoch_loss_avg.result():.4f}, Accuracy: {epoch_accuracy.result():.4f}, "
                     f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        if epoch % 5 == 0:
            self.model.save_weights(os.path.join(FLAGS.model_dir, f'checkpoint_epoch_{epoch}'))

        # Early stopping logic
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= FLAGS.early_stopping_patience:
                return True  # Stop training

        return False  # Continue training

    def evaluate(self) -> Tuple[float, float]:
        loss_avg = tf.keras.metrics.Mean()
        accuracy = tf.keras.metrics.Mean()
        
        for user, movie, rating in self.dataset:
            table_data = tf.concat([user, movie, rating], axis=1)
            valid_indices = tf.where(user[:, 0] >= 0)
            valid_user_indices = tf.gather(user[:, 0], valid_indices)
            graph_data_batch = tf.gather(self.graph_data, tf.cast(valid_user_indices, tf.int32))
            outputs = self.model((table_data, graph_data_batch, self.edge_index), training=False)
            labels = tf.random.uniform(shape=(tf.shape(outputs)[0],), maxval=7, dtype=tf.int32)
            loss = self.loss_fn(labels, outputs)
            loss_avg.update_state(loss)
            predictions = tf.cast(tf.argmax(outputs, axis=1), tf.int32)
            accuracy.update_state(tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32)))
        
        return loss_avg.result().numpy(), accuracy.result().numpy()

def generate_synthetic_data(config: DataConfig) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    user_data = tf.random.normal((config.num_users, config.user_dim))
    movie_data = tf.random.normal((config.num_movies, config.movie_dim))
    rating_data = tf.random.normal((config.num_ratings, config.rating_dim))
    graph_data = tf.random.normal((config.num_graph_nodes, config.graph_dim))
    edge_index = tf.random.uniform((2, config.num_ratings), 0, config.num_graph_nodes, dtype=tf.int32)
    return user_data, movie_data, rating_data, graph_data, edge_index

def setup_model_and_training(config: DataConfig, strategy: tf.distribute.Strategy) -> Tuple[BRIDGE, tf.data.Dataset, optimizers.Optimizer, losses.Loss, tf.Tensor, tf.Tensor]:
    with strategy.scope():
        table_encoder = TableEncoder(config.user_dim + config.movie_dim + config.rating_dim, 32, 64, FLAGS.dropout_rate)
        graph_encoder = GraphEncoder(config.graph_dim, 32, FLAGS.dropout_rate)
        model = BRIDGE(table_encoder, graph_encoder)
        optimizer = optimizers.Adam(learning_rate=FLAGS.learning_rate)
        loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)

    user_data, movie_data, rating_data, graph_data, edge_index = generate_synthetic_data(config)
    
    # Split data into train and validation sets
    user_train, user_val, movie_train, movie_val, rating_train, rating_val = train_test_split(
        user_data, movie_data, rating_data, test_size=0.2, random_state=42)
    
    train_dataset = BaseTable(user_train, movie_train, rating_train).get_dataset(FLAGS.batch_size, FLAGS.shuffle_buffer)
    val_dataset = BaseTable(user_val, movie_val, rating_val).get_dataset(FLAGS.batch_size, FLAGS.shuffle_buffer)
    
    return model, train_dataset, val_dataset, optimizer, loss_fn, graph_data, edge_index

def plot_training_history(history: Dict[str, List[float]], save_path: str):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_model_summary(model: BRIDGE, save_path: str):
    with open(save_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

def main(argv):
    del argv  # Unused

    logging.set_verbosity(logging.INFO)

    # Set up mixed precision if requested
    if FLAGS.use_mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logging.info("Mixed precision enabled")

    config = DataConfig()

    if FLAGS.use_tpu:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu_name)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        logging.info("TPU strategy enabled")
    else:
        strategy = tf.distribute.get_strategy()
        logging.info(f"Using default strategy: {strategy}")

    model, train_dataset, val_dataset, optimizer, loss_fn, graph_data, edge_index = setup_model_and_training(config, strategy)

    # Create a timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(FLAGS.model_dir, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    # Save model summary
    save_model_summary(model, os.path.join(run_dir, 'model_summary.txt'))

    # Set up TensorBoard
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(run_dir, 'logs'))

    trainer = BRIDGETrainer(model, train_dataset, optimizer, loss_fn, graph_data, edge_index)
    history = trainer.train(num_epochs=FLAGS.num_epochs)

    # Save training history
    with open(os.path.join(run_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)

    # Plot and save training history
    plot_training_history(history, os.path.join(run_dir, 'training_history.png'))

    # Final evaluation
    test_loss, test_accuracy = trainer.evaluate()
    logging.info(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Save final model
    model.save(os.path.join(run_dir, 'final_model'))

    # Save test results
    with open(os.path.join(run_dir, 'test_results.json'), 'w') as f:
        json.dump({'test_loss': test_loss, 'test_accuracy': test_accuracy}, f)

    logging.info(f"Training completed. Results saved in {run_dir}")

if __name__ == "__main__":
    app.run(main)
    
    