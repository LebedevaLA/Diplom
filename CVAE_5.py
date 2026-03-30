
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from GenerateFunc_1 import Unimodal, Periodic, Piecewise  
from Create_TestFunc_4 import ComplexUnimodal, ComplexPeriodic

# ==== КЛАСС CVAE ====
class CVAE(tf.keras.Model):
    def __init__(self, latent_dim=10, num_classes=3, encoder_layers=[128, 64],
                 decoder_layers=[64, 128], input_dim=100, beta=0.1):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.beta = beta

        self.encoder = self.build_encoder(encoder_layers)
        self.decoder = self.build_decoder(decoder_layers)
        self.classifier = self.build_classifier()

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.classification_loss_tracker = tf.keras.metrics.Mean(
            name="classification_loss")

    def build_encoder(self, encoder_layers):
        inputs = tf.keras.Input(shape=(self.input_dim,))
        labels = tf.keras.Input(shape=(self.num_classes,))

        x = tf.keras.layers.concatenate([inputs, labels])

        for units in encoder_layers:
            x = tf.keras.layers.Dense(units)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Dropout(0.3)(x)

        z_mean = tf.keras.layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = tf.keras.layers.Dense(self.latent_dim, name="z_log_var")(x)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = tf.keras.backend.random_normal(
                shape=(tf.shape(z_mean)[0], self.latent_dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        z = tf.keras.layers.Lambda(sampling)([z_mean, z_log_var])

        return tf.keras.Model([inputs, labels], [z_mean, z_log_var, z], name="encoder")

    def build_decoder(self, decoder_layers):
        latent_inputs = tf.keras.Input(shape=(self.latent_dim,))
        labels = tf.keras.Input(shape=(self.num_classes,))

        x = tf.keras.layers.concatenate([latent_inputs, labels])

        for i, units in enumerate(decoder_layers):
            x = tf.keras.layers.Dense(units, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            # Добавляем dropout после каждого слоя, кроме последнего
            if i < len(decoder_layers) - 1:
                x = tf.keras.layers.Dropout(0.3)(x)

        outputs = tf.keras.layers.Dense(self.input_dim, activation='linear')(x)

        return tf.keras.Model([latent_inputs, labels], outputs, name="decoder")

    def build_classifier(self):
        inputs = tf.keras.Input(shape=(self.input_dim,))

        x = tf.keras.layers.Dense(256, activation='relu')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.1)(x)

        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dense(16, activation='relu')(x)

        outputs = tf.keras.layers.Dense(
            self.num_classes, activation='softmax')(x)

        return tf.keras.Model(inputs, outputs, name="classifier")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.classification_loss_tracker,
        ]

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([x, y])
            reconstructed = self.decoder([z, y])
            class_pred = self.classifier(x)

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(x - reconstructed), axis=1)
            )

            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) -
                              tf.exp(z_log_var), axis=1)
            )

            classification_loss = tf.keras.losses.categorical_crossentropy(
                y, class_pred)
            classification_loss = tf.reduce_mean(classification_loss)

            total_loss = reconstruction_loss + self.beta * kl_loss + classification_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.classification_loss_tracker.update_state(classification_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "classification_loss": self.classification_loss_tracker.result(),
        }

    def test_step(self, data):
        x, y = data

        z_mean, z_log_var, z = self.encoder([x, y])
        reconstructed = self.decoder([z, y])
        class_pred = self.classifier(x)

        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(x - reconstructed), axis=1)
        )

        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) -
                          tf.exp(z_log_var), axis=1)
        )

        classification_loss = tf.keras.losses.categorical_crossentropy(
            y, class_pred)
        classification_loss = tf.reduce_mean(classification_loss)

        total_loss = reconstruction_loss + self.beta * kl_loss + classification_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "classification_loss": classification_loss,
        }

    def encode(self, x, y):
        return self.encoder.predict([x, y], verbose=0)

    def decode(self, z, y):
        return self.decoder.predict([z, y], verbose=0)

    def predict_class(self, x):
        return self.classifier.predict(x, verbose=0)

    def save_weights(self, filepath):
        weights_dict = {
            'encoder': self.encoder.get_weights(),
            'decoder': self.decoder.get_weights(),
            'classifier': self.classifier.get_weights()
        }
        with open(filepath, 'wb') as f:
            pickle.dump(weights_dict, f)
        print(f"Веса сохранены в {filepath}")

    def load_weights(self, filepath):
        with open(filepath, 'rb') as f:
            weights_dict = pickle.load(f)

        self.encoder.set_weights(weights_dict['encoder'])
        self.decoder.set_weights(weights_dict['decoder'])
        self.classifier.set_weights(weights_dict['classifier'])
        print(f"Веса загружены из {filepath}")


# ==== КЛАССИФИКАТОР ====

class FunctionClassifier:
    def __init__(self, latent_dim=20, encoder_layers=[256, 128, 64],
                 decoder_layers=[64, 128, 256], input_dim=100, num_classes=3):
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.scaler = StandardScaler()

        self.cvae = CVAE(
            latent_dim=latent_dim,
            num_classes=num_classes,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            input_dim=input_dim,
            beta=0.1
        )

        self.cvae.compile(optimizer=Adam(learning_rate=0.001))

    def load_train_data(self, train_file='train_functions.pkl'):
        """
        Загружает обучающие данные из файла train_functions.pkl
        Разделяет на train/val (без test)
        """
        print("Загрузка обучающих данных...")
        
        with open(train_file, 'rb') as f:
            train_data = pickle.load(f)
        
        X_list = []
        y_list = []
        
        class_mapping = {
            'unimodal': 0,
            'periodic': 1,
            'piecewise': 2
        }
        
        # Собираем все функции
        for class_name, func_list in train_data.items():
            if class_name in class_mapping:
                class_idx = class_mapping[class_name]
                for func in func_list:
                    X_list.append(func['y'])
                    y_list.append(class_idx)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"Загружено {len(X)} функций для обучения")
        print(f"Распределение по классам:")
        for class_name, idx in class_mapping.items():
            count = np.sum(y == idx)
            print(f"  {class_name}: {count} функций")
        
        # Разделяем ТОЛЬКО на train и val (80/20)
        # НЕ создаем test из обучающих данных!
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Масштабирование
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Преобразуем метки в one-hot
        y_train_onehot = tf.keras.utils.to_categorical(y_train, self.num_classes)
        y_val_onehot = tf.keras.utils.to_categorical(y_val, self.num_classes)
        
        print(f"\nРазмеры данных:")
        print(f"  Train: {X_train_scaled.shape} ({len(X_train_scaled)} функций)")
        print(f"  Val: {X_val_scaled.shape} ({len(X_val_scaled)} функций)")
        
        return (X_train_scaled, X_val_scaled, 
                y_train_onehot, y_val_onehot, 
                y_train, y_val)

    def load_test_data(self, test_file='test_functions.pkl'):
        """
        Загружает тестовые данные из файла test_functions.pkl
        """
        print("\nЗагрузка тестовых данных...")
        
        with open(test_file, 'rb') as f:
            test_data = pickle.load(f)
        
        X_list = []
        y_list = []
        
        class_mapping = {
            'test_unimodal': 0,
            'test_periodic': 1,
            'test_piecewise': 2
        }
        
        for class_name, func_list in test_data.items():
            if class_name in class_mapping:
                class_idx = class_mapping[class_name]
                for func in func_list:
                    X_list.append(func['y'])
                    y_list.append(class_idx)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"Загружено {len(X)} функций для тестирования")
        print(f"Распределение по классам:")
        for class_name, idx in class_mapping.items():
            count = np.sum(y == idx)
            print(f"  {class_name}: {count} функций")
        
        X_scaled = self.scaler.transform(X)
        
        print(f"\nРазмер тестовых данных: {X_scaled.shape}")
        
        return X_scaled, y

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):  # ИЗМЕНЕНО: больше эпох
        print("\nОбучение модели CVAE...")

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        history = self.cvae.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return history


    def evaluate(self, X_test, y_test, data_name="Test",test_file='test_functions.pkl'):
        """
        Оценивает модель на тестовых данных
        """
        y_pred_proba = self.cvae.predict_class(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        print("\n" + "="*60)
        print(f"ОЦЕНКА НА {data_name.upper()} ДАННЫХ")
        print("="*60)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                    target_names=['unimodal', 'periodic', 'piecewise']))
        
        # Матрица ошибок
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['unimodal', 'periodic', 'piecewise'],
                    yticklabels=['unimodal', 'periodic', 'piecewise'])
        plt.title(f'Confusion Matrix ({data_name} Data)')
        plt.tight_layout()
        plt.savefig(f'{data_name.lower()}_confusion_matrix.png', dpi=150)
        plt.show()
        
        # Точность по классам
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        print("\nТочность по классам:")
        for i, name in enumerate(['unimodal', 'periodic', 'piecewise']):
            print(f"  {name}: {class_accuracy[i]:.2%}")
        
        print(f"\nОбщая точность: {np.mean(y_pred == y_test):.2%}")
        print("\n" + "="*60)
        print("Анализ по каждой функции")
        print("="*60)
        # Загружаем исходные данные для получения id и типа
        with open(test_file, 'rb') as f:
            test_data = pickle.load(f)
        
        # Собираем все функции в порядке индексов
        all_functions = []
        for class_name, func_list in test_data.items():
            for func in func_list:
                all_functions.append({
                    'id': func.get('id', 0),
                    'type': func['type'],
                    'true_class': class_name.replace('test_', ''),
                    'func_obj': func['func_obj'],
                    'x': func['x'],
                    'y': func['y'],
                    'true_params': func.get('true_params', [])
                })
        
        pred_class_names = ['unimodal', 'periodic', 'piecewise']
        
        # Структура для хранения только правильно классифицированных функций
        correct_classify = {
            'unimodal': [],
            'periodic': [],
            'piecewise': []
        }
        
        
        print("\n" + "="*60)
        print("ДЕТАЛЬНЫЙ АНАЛИЗ ПО КАЖДОЙ ФУНКЦИИ")
        print("="*60)
        
        correct_count = 0
        total_count = len(y_test)
        
        for idx, (true_label, pred_label) in enumerate(zip(y_test, y_pred)):
            func_info = all_functions[idx]
            true_class_name = func_info['true_class']
            pred_class_name = pred_class_names[pred_label]
            prob = y_pred_proba[idx][pred_label]
            is_correct = (true_label == pred_label)
            
            if is_correct:
                correct_count += 1
                # Сохраняем функцию в правильный формат
                saved_func = {
                    'func_obj': func_info['func_obj'],
                    'x': func_info['x'],
                    'y': func_info['y'],
                    'true_params': func_info['true_params'],
                    'class': true_class_name,
                    'id': func_info['id'],
                    'type':func_info['type']
                }
                correct_classify[true_class_name].append(saved_func)
                status = "✓"
            else:
                status = "✗"
            
            print(f"  {status} id={func_info['id']:3d} | {func_info['type']:20s} | "
                f"Истинный: {true_class_name:10s} | Предсказано: {pred_class_name:10s} | "
                f"Уверенность: {prob:.2%}")
        
        print("\n" + "="*60)
        print(f"ИТОГ: верно классифицировано {correct_count} из {total_count} функций "
            f"({correct_count/total_count:.2%})")
        print("="*60)
        
        # Статистика по классам
        print("\nСтатистика по классам:")
        for class_name in ['unimodal', 'periodic', 'piecewise']:
            class_funcs = correct_classify[class_name]
            print(f"  {class_name}: {len(class_funcs)}/{len([f for f in all_functions if f['true_class'] == class_name])} = {len(class_funcs)/len([f for f in all_functions if f['true_class'] == class_name]):.2%}")
        
        # Сохраняем правильно классифицированные функции в файл
        output_file = 'correctly_classified.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(correct_classify, f)
        print(f"\nСохранено {correct_count} правильно классифицированных функций в {output_file}")
        for class_name in correct_classify:
            print(f"  {class_name}: {len(correct_classify[class_name])} функций")
        
        return y_pred_proba, y_pred

    def plot_training_history(self, history):
        """Визуализация истории обучения"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        metrics = ['loss', 'reconstruction_loss', 'kl_loss', 'classification_loss']
        titles = ['Total Loss', 'Reconstruction Loss', 'KL Loss', 'Classification Loss']

        for i, metric in enumerate(metrics):
            if metric in history.history:
                axes[i].plot(history.history[metric], label='Train')
                if f'val_{metric}' in history.history:
                    axes[i].plot(history.history[f'val_{metric}'], label='Validation')
                axes[i].set_title(titles[i])
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel('Loss')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.show()


# Main

def main():
    print("="*70)
    print("ОБУЧЕНИЕ КЛАССИФИКАТОРА ФУНКЦИЙ")
    print("="*70)
    
    classifier = FunctionClassifier(
    latent_dim=100,                    # было 30
    encoder_layers=[512, 256, 128],   # больше слоев
    decoder_layers=[128, 256, 512],
    input_dim=100,
    num_classes=3
    )

    # 2. Загрузка обучающих данных
    X_train, X_val, y_train, y_val, y_train_raw, y_val_raw = classifier.load_train_data('train_functions.pkl')

    # 3. Проверка наличия сохраненных весов
    weights_file = 'cvae_weights.pkl'
    train_model = True

    if os.path.exists(weights_file):
        response = input(f"\nНайден файл весов '{weights_file}'. Загрузить веса? (y/n): ")
        if response.lower() == 'y':
            classifier.cvae.load_weights(weights_file)
            train_model = False
            print("Веса успешно загружены!")

    # 4. Обучение модели
    if train_model:
        print("\n" + "="*70)
        print("НАЧАЛО ОБУЧЕНИЯ")
        print("="*70)
        
        history = classifier.train(
            X_train, y_train,
            X_val, y_val,
            epochs=300,        
            batch_size=32
        )

        # Сохранение весов
        classifier.cvae.save_weights(weights_file)
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(classifier.scaler, f)
        print("Scaler сохранен в scaler.pkl")
        # Визуализация истории обучения
        classifier.plot_training_history(history)

    # 5. Оценка на тестовых данных
    X_test, y_test = classifier.load_test_data('test_functions.pkl')
    classifier.evaluate(X_test, y_test, data_name="Test")

    print("\n" + "="*70)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("="*70)


if __name__ == "__main__":
    main()