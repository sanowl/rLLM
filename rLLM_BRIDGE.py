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
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from datetime import datetime
import kerastuner as kt
import wandb
from wandb.keras import WandbCallback

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
flags.DEFINE_integer('val_frequency', 1, 'Frequency (in epochs) of validation')
flags.DEFINE_float('lr_decay', 0.99, 'Learning rate decay factor per epoch')
flags.DEFINE_boolean('use_wandb', False, 'Whether to use Weights and Biases for logging')
flags.DEFINE_string('wandb_project', 'my_project', 'Weights and Biases project name')
flags.DEFINE_string('wandb_entity', 'my_entity', 'Weights and Biases entity name')

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

# Custom Callbacks
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logging.info(f"Custom Callback - Epoch {epoch + 1} End: Loss = {logs['loss']}, Accuracy = {logs['accuracy']}")

class LRSchedulerCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        new_lr = lr * FLAGS.lr_decay
        logging.info(f"Learning rate updated from {lr:.6f} to {new_lr:.6f}")
        self.model.optimizer.learning_rate = new_lr

class CustomModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, save_path, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'):
        super().__init__()
        self.save_path = save_path
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.best = np.Inf if mode == 'min' else -np.Inf
        self.mode = mode

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.mode == 'min' and current < self.best:
            if self.verbose > 0:
                print(f"\nEpoch {epoch + 1}: {self.monitor} improved from {self.best:.4f} to {current:.4f}, saving model to {self.save_path}")
            self.best = current
            self.model.save(self.save_path)
        elif self.mode == 'max' and current > self.best:
            if self.verbose > 0:
                print(f"\nEpoch {epoch + 1}: {self.monitor} improved from {self.best:.4f} to {current:.4f}, saving model to {self.save_path}")
            self.best = current
            self.model.save(self.save_path)

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
    def __init__(self, model: BRIDGE, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset,
                 optimizer: optimizers.Optimizer, loss_fn: losses.Loss, graph_data: tf.Tensor, edge_index: tf.Tensor):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.graph_data = graph_data
        self.edge_index = edge_index
        self.history: Dict[str, List[float]] = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        for epoch in range(num_epochs):
            logging.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            self._train_epoch(epoch)
            if epoch % FLAGS.val_frequency == 0:
                val_loss, val_accuracy = self.evaluate()
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)
                logging.info(f"Validation after epoch {epoch + 1}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.model.save_weights(os.path.join(FLAGS.model_dir, 'best_checkpoint'))
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= FLAGS.early_stopping_patience:
                        logging.info(f"Early stopping triggered after epoch {epoch + 1}")
                        break

            self.optimizer.learning_rate = self.optimizer.learning_rate * FLAGS.lr_decay
        return self.history

    def _train_epoch(self, epoch: int):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.Mean()
        
        for user, movie, rating in self.train_dataset:
            table_data = tf.concat([user, movie, rating], axis=1)
            valid_indices = tf.where(user[:, 0] >= 0)
            valid_user_indices = tf.gather(user[:, 0], valid_indices)
            graph_data_batch = tf.gather(self.graph_data, tf.cast(valid_user_indices, tf.int32))
            loss, accuracy = train_step(self.model, (table_data, graph_data_batch, self.edge_index), self.optimizer, self.loss_fn)
            epoch_loss_avg.update_state(loss)
            epoch_accuracy.update_state(accuracy)
        
        self.history['loss'].append(epoch_loss_avg.result().numpy())
        self.history['accuracy'].append(epoch_accuracy.result().numpy())
        
        logging.info(f"Epoch {epoch + 1}, Loss: {epoch_loss_avg.result():.4f}, Accuracy: {epoch_accuracy.result():.4f}")

    def evaluate(self) -> Tuple[float, float]:
        loss_avg = tf.keras.metrics.Mean()
        accuracy = tf.keras.metrics.Mean()
        
        for user, movie, rating in self.val_dataset:
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

def setup_model_and_training(config: DataConfig, strategy: tf.distribute.Strategy) -> Tuple[BRIDGE, tf.data.Dataset, tf.data.Dataset, optimizers.Optimizer, losses.Loss, tf.Tensor, tf.Tensor]:
    with strategy.scope():
        table_encoder = TableEncoder(config.user_dim + config.movie_dim + config.rating_dim, 32, 64, FLAGS.dropout_rate)
        graph_encoder = GraphEncoder(config.graph_dim, 32, FLAGS.dropout_rate)
        model = BRIDGE(table_encoder, graph_encoder)
        optimizer = optimizers.AdamW(learning_rate=FLAGS.learning_rate, weight_decay=1e-4)
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

def build_model(hp):
    table_encoder = TableEncoder(
        input_dim=hp.Int('input_dim', min_value=16, max_value=64, step=16),
        hidden_dim=hp.Int('hidden_dim', min_value=32, max_value=128, step=32),
        output_dim=hp.Int('output_dim', min_value=16, max_value=64, step=16),
        dropout_rate=hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    )
    graph_encoder = GraphEncoder(
        in_features=hp.Int('graph_in_features', min_value=16, max_value=64, step=16),
        out_features=hp.Int('graph_out_features', min_value=16, max_value=64, step=16),
        dropout_rate=hp.Float('graph_dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    )
    model = BRIDGE(table_encoder, graph_encoder)
    model.compile(
        optimizer=optimizers.AdamW(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG'),
                        weight_decay=hp.Float('weight_decay', min_value=1e-6, max_value=1e-2, sampling='LOG')),
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

def plot_confusion_matrix(y_true, y_pred, save_path: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path)
    plt.close()

def generate_classification_report(y_true, y_pred, save_path: str):
    report = classification_report(y_true, y_pred, output_dict=True)
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=4)

def plot_roc_curve(y_true, y_pred, save_path: str):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

def plot_precision_recall_curve(y_true, y_pred, save_path: str):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

def plot_model_architecture(model: models.Model, save_path: str):
    keras.utils.plot_model(model, to_file=save_path, show_shapes=True, show_layer_names=True)

def data_augmentation(user_data: tf.Tensor, movie_data: tf.Tensor, rating_data: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    user_data = tf.image.random_flip_left_right(user_data)
    movie_data = tf.image.random_flip_up_down(movie_data)
    rating_data = rating_data + tf.random.normal(tf.shape(rating_data), mean=0.0, stddev=0.1)
    return user_data, movie_data, rating_data

def main(argv):
    del argv  # Unused

    logging.set_verbosity(logging.INFO)

    if FLAGS.use_wandb:
        wandb.init(project=FLAGS.wandb_project, entity=FLAGS.wandb_entity)

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
    
    # Save model architecture
    plot_model_architecture(model, os.path.join(run_dir, 'model_architecture.png'))

    # Set up callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(run_dir, 'logs'))
    custom_callback = CustomCallback()
    lr_scheduler_callback = LRSchedulerCallback()
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=FLAGS.early_stopping_patience, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    model_checkpoint = CustomModelCheckpoint(os.path.join(run_dir, 'best_model'), monitor='val_loss', verbose=1)

    trainer = BRIDGETrainer(model, train_dataset, val_dataset, optimizer, loss_fn, graph_data, edge_index)
    history = trainer.train(num_epochs=FLAGS.num_epochs)

    # Save training history
    with open(os.path.join(run_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)

    # Plot and save training history
    plot_training_history(history, os.path.join(run_dir, 'training_history.png'))

    # Final evaluation
    test_loss, test_accuracy = trainer.evaluate()
    logging.info(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Generate predictions for confusion matrix
    all_predictions = []
    all_labels = []
    for user, movie, rating in val_dataset:
        table_data = tf.concat([user, movie, rating], axis=1)
        valid_indices = tf.where(user[:, 0] >= 0)
        valid_user_indices = tf.gather(user[:, 0], valid_indices)
        graph_data_batch = tf.gather(graph_data, tf.cast(valid_user_indices, tf.int32))
        outputs = model((table_data, graph_data_batch, edge_index), training=False)
        predictions = tf.argmax(outputs, axis=1)
        labels = tf.random.uniform(shape=(tf.shape(outputs)[0],), maxval=7, dtype=tf.int32)
        all_predictions.extend(predictions.numpy())
        all_labels.extend(labels.numpy())

    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_predictions, os.path.join(run_dir, 'confusion_matrix.png'))

    # Generate classification report
    generate_classification_report(all_labels, all_predictions, os.path.join(run_dir, 'classification_report.json'))

    # Plot ROC curve
    plot_roc_curve(all_labels, all_predictions, os.path.join(run_dir, 'roc_curve.png'))

    # Plot Precision-Recall curve
    plot_precision_recall_curve(all_labels, all_predictions, os.path.join(run_dir, 'precision_recall_curve.png'))

    # Save final model
    model.save(os.path.join(run_dir, 'final_model'))

    # Save test results
    with open(os.path.join(run_dir, 'test_results.json'), 'w') as f:
        json.dump({'test_loss': test_loss, 'test_accuracy': test_accuracy}, f)

    logging.info(f"Training completed. Results saved in {run_dir}")

    # Hyperparameter tuning with Keras Tuner
    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=20,
        factor=3,
        directory=run_dir,
        project_name='hyperparameter_tuning'
    )

    tuner.search(train_dataset, epochs=FLAGS.num_epochs, validation_data=val_dataset, 
                 callbacks=[tensorboard_callback, custom_callback, lr_scheduler_callback, early_stopping, reduce_lr])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    logging.info(f"Best hyperparameters: {best_hps.values}")

    best_model = tuner.hypermodel.build(best_hps)
    history = best_model.fit(train_dataset, epochs=FLAGS.num_epochs, validation_data=val_dataset, 
                             callbacks=[tensorboard_callback, custom_callback, lr_scheduler_callback, early_stopping, reduce_lr])

    # Save best model
    best_model.save(os.path.join(run_dir, 'best_model'))

    if FLAGS.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    app.run(main)
