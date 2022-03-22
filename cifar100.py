import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


# Beskriv input og label for modellen
INPUT_SHAPE = (32, 32, 3)
NUMBER_OF_CLASSES = 100


def train(args):
    """Train and save model"""

    # Last inn trenings- og valideringsdata
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

    # Skaler input fra [0, 255] til [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Konverter label fra 1 til [0 1 0 0 0 0 0 0 0 0]
    y_train = to_categorical(y_train, NUMBER_OF_CLASSES)
    y_test = to_categorical(y_test, NUMBER_OF_CLASSES)

    # Opprett modellen
    model = build_model()

    # Stopp trening automatisk dersom den stagnerer
    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        verbose=1,
        patience=args.patience,
        restore_best_weights=True,
    )

    # Tren modellen
    model.fit(
        x_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=1,
        validation_split=0.2,
        callbacks=[early_stopping],
    )


    # Evaluer modellen på valideringsdata
    score = model.evaluate(x_test, y_test, verbose=0)

    print(f'Accuracy: {score[1]}')

    return model


def build_model():
    """Builds the model"""

    model = Sequential([
        Conv2D(32, (3, 3), padding='same', input_shape=INPUT_SHAPE),
        Activation('relu'),
        Conv2D(32, (3, 3), padding='same'),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),

        Conv2D(64, (3, 3), padding='same'),
        Activation('relu'),
        Conv2D(64, (3, 3), padding='same'),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),

        Flatten(),
        Dense(512),
        Activation('relu'),
        Dropout(0.5),
        Dense(100, activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model


def save(model, args):
    """Save model to file system"""

    if args.save:
        model.save(args.model)


def predict(args):
    """Predict the classes of an image"""

    model = load_model(args.model)

    image = Image.open(args.image, 'r')
    image_resized = image.resize((32, 32))
    image_data = np.expand_dims(
        np.asarray(image_resized).astype('float32') / 255.0,
        axis=0,
    )

    results = model.predict(image_data)[0]

    plot_result(image, results)


def plot_result(image, result):
    """Show image and result"""

    with open('cifar100-labels.txt', 'r', encoding='utf-8') as file:
        labels = file.read().splitlines()

    sorted_results = sorted(
        enumerate(result), key=lambda x: x[1], reverse=True,
    )

    top_five_results = [
        (labels[index], probability) for (index, probability) in sorted_results[:5]
    ]

    result_text = '\n'.join([
        f'{label}: {(probability * 100):.2f}%' for (label, probability) in top_five_results
    ])

    _, axs = plt.subplots(nrows=2, sharex=True, figsize=(3, 5))

    axs[0].imshow(image, origin='upper')
    axs[0].axis('off')

    axs[1].text(0, 0, result_text)
    axs[1].axis('off')

    plt.show()


def run():
    """Run software"""

    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=50)

    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--model', type=str, default='model')
    parser.add_argument('--image', type=str, default=None)

    args = parser.parse_args()

    if args.train:
        model = train(args)

        save(model, args)

    if args.predict:
        predict(args)


if __name__ == '__main__':
    run()
