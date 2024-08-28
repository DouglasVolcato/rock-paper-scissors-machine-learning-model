import tensorflow as tf
import pandas as pd


class ClassificationModel:
    def convert_to_int(self, play: str) -> int:
        if play == 'R':
            return 0
        elif play == 'P':
            return 1
        else:
            return 2

    def convert_to_str(self, play: int) -> str:
        if play == 0:
            return 'R'
        elif play == 1:
            return 'P'
        else:
            return 'S'

    def get_next_winning_move(self, next_play: str) -> str:
        if next_play == 'R':
            return 'P'
        elif next_play == 'P':
            return 'S'
        else:
            return 'R'

    def train(self) -> None:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(
                units=128, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=3, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        data = []
        labels = []
        for i in range(1, 5):
            df = pd.read_csv(f"cache/player{i}.csv")
            letters_array = df['last_play'].tolist()
            for j in range(0, len(letters_array) - 11, 10):
                input_sequence = letters_array[j:j + 10]
                next_play = letters_array[j + 10]

                data.append([self.convert_to_int(x) for x in input_sequence])
                labels.append(self.convert_to_int(
                    self.get_next_winning_move(next_play)))
                
        for i in range(1, 5):
            df = pd.read_csv(f"cache/player{i}.csv")
            letters_array = df['last_play'].tolist()
            for j in range(0, len(letters_array) - 11, 10):
                input_sequence = letters_array[j:j + 10]
                next_play = letters_array[j + 10]

                data.append([self.convert_to_int(x) for x in input_sequence])
                labels.append(self.convert_to_int(
                    self.get_next_winning_move(next_play)))
                
        for i in range(1, 5):
            df = pd.read_csv(f"cache/player{i}.csv")
            letters_array = df['last_play'].tolist()
            for j in range(0, len(letters_array) - 11, 10):
                input_sequence = letters_array[j:j + 10]
                next_play = letters_array[j + 10]

                data.append([self.convert_to_int(x) for x in input_sequence])
                labels.append(self.convert_to_int(
                    self.get_next_winning_move(next_play)))

        data = tf.convert_to_tensor(data, dtype=tf.float32)
        labels = tf.convert_to_tensor(labels, dtype=tf.int64)

        # Split data into train and test
        train_size = int(0.9 * len(data))
        train_data = data[:train_size]
        train_labels = labels[:train_size]
        test_data = data[train_size:]
        test_labels = labels[train_size:]

        model.fit(train_data, train_labels, epochs=50, validation_split=0.2)

        print("====================")
        print(model.summary())
        test_loss, test_accuracy = model.evaluate(test_data, test_labels)
        print(f"Test accuracy: {test_accuracy}")
        print(f"Trained model saved at models/model.keras")

        model.save(f"models/model.keras")

    def predict(self, prev_plays: list[str]) -> str:
        prev_plays = [self.convert_to_int(x) for x in prev_plays]
        prev_plays = tf.convert_to_tensor([prev_plays], dtype=tf.float32)
        model = tf.keras.models.load_model(f"models/model.keras")
        prediction = model.predict(prev_plays)
        predicted_class = tf.argmax(prediction[0], output_type=tf.int32)
        return self.convert_to_str(predicted_class.numpy())
