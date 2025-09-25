import os
from collections import deque
import concurrent.futures
import concurrent.futures._base
from concurrent.futures import ProcessPoolExecutor
from typing import Any, AnyStr, Callable, NamedTuple

import librosa
import librosa.effects
import librosa.feature
import numpy as np
from numpy.typing import NDArray

from emotion import Emotion

os.environ["KERAS_BACKEND"] = "jax"
import keras  # type: ignore
import jax


class AudioFileItem(NamedTuple):
    path: str | os.PathLike[Any]
    label: np.int32


class Model:
    _pending: deque[AudioFileItem] = deque()
    # Should change _mfccs, _labels to numpy array with dynamic expanding someday
    _mfccs: list[NDArray[np.float32]] = []
    _labels: list[np.int32] = []
    _sample_rate: float
    _n_mfcc: int
    model = None

    def __init__(self, sample_rate: float, n_mfcc: int):
        self._sample_rate = sample_rate
        self._n_mfcc = n_mfcc

    def load_dataset(self, root_dir: os.PathLike[AnyStr] | str,
                     code_emotion_map: dict[Any, Emotion],
                     code_derive_function: Callable[[AnyStr], Any]):
        def __scan(dir_path):
            for e in os.scandir(dir_path):
                if e.is_dir():
                    __scan(e.path)  # recurse
                elif e.is_file():
                    if not e.name.lower().endswith(".wav"):
                        continue
                    try:
                        code = code_derive_function(os.path.basename(e.path))
                        emotion = code_emotion_map[code]
                    except IndexError:
                        raise ValueError(f"Malformed filename: {e.path}")
                    except KeyError:
                        raise ValueError(f"Emotion type not recognized: {e.path}")
                    self._pending.append(AudioFileItem(
                        path=os.path.abspath(e.path),
                        label=np.int32(emotion.value)))

        __scan(root_dir)

    @staticmethod
    def extract_mfcc(y: np.ndarray, sr: float, n_mfcc: int) -> NDArray[np.float32]:
        return np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T, axis=0).astype(np.float32)

    @staticmethod
    def _generate_augmented_audios(y: np.ndarray, sr: float, target_length=None) \
        -> list[NDArray[np.float32]]:
        augmented = []
        for n_steps in [-2, -1, 1, 2]:
            y_ps = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
            augmented.append(y_ps)
        for rate in [0.9, 1.1]:
            y_ts = librosa.effects.time_stretch(y=y, rate=rate)
            if target_length:
                if len(y_ts) > target_length:
                    y_ts = y_ts[:target_length]
                else:
                    y_ts = np.pad(y_ts, (0, target_length - len(y_ts)))
            augmented.append(y_ts)
        augmented.append(y + np.random.normal(0, 0.005, len(y)))
        return augmented

    def generate_mfccs(self) -> None:
        while self._pending:
            path, label = self._pending.popleft()
            print(f"Remaining: {len(self._pending)} {path}", end="\r")
            y, _ = librosa.load(path, sr=self._sample_rate)
            self._mfccs.append(self.extract_mfcc(y, self._sample_rate, self._n_mfcc))
            self._labels.append(label)
            for augmented_y in self._generate_augmented_audios(y, self._sample_rate):
                self._mfccs.append(self.extract_mfcc(augmented_y, self._sample_rate, self._n_mfcc))
                self._labels.append(label)

    @staticmethod
    def _get_mfcc_with_augmented(path: str | os.PathLike[Any], sr: float, n_mfcc: int) -> list[NDArray[np.float32]]:
        __y, _ = librosa.load(path, sr=sr)
        __mfccs = [Model.extract_mfcc(__y, sr, n_mfcc)]
        for aug_y in Model._generate_augmented_audios(__y, sr):
            __mfccs.append(Model.extract_mfcc(aug_y, sr, n_mfcc))
        return __mfccs

    @staticmethod
    def _get_mfcc_with_augmented_label(path, sr, n_mfcc, label) -> tuple[list[NDArray[np.float32]], np.int32]:
        return Model._get_mfcc_with_augmented(path, sr, n_mfcc), label

    def generate_mfccs_parallel(self, n_worker=os.cpu_count()) -> None:
        in_flight: set[concurrent.futures._base.Future[tuple[list[NDArray[np.float32]], np.int32]]] = set()
        with ProcessPoolExecutor(max_workers=n_worker) as executor:
            while self._pending or in_flight:
                while self._pending and len(in_flight) < n_worker:
                    path, label = self._pending.popleft()
                    in_flight.add(executor.submit(Model._get_mfcc_with_augmented_label, path=path, sr=self._sample_rate,
                                                  n_mfcc=self._n_mfcc, label=label))
                    print(f"Remaining: {len(self._pending)} {path}", end="\r")
                done, in_flight = concurrent.futures.wait(in_flight, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    try:
                        mfccs, label = future.result()
                        for mfcc in mfccs:
                            self._mfccs.append(mfcc)
                            self._labels.append(label)
                    except Exception as ex:
                        print(f"Error processing audio: {ex}")

    def save_mfccs_to_file(self, mfccs_filename: str | os.PathLike[str],
                           labels_filename: str | os.PathLike[str]) -> None:
        np.save(mfccs_filename, np.array(self._mfccs))
        np.save(labels_filename, np.array(self._labels))

    def load_mfccs_from_file(self, mfccs_filename: str | os.PathLike[str],
                             labels_filename: str | os.PathLike[str]) -> None:
        self._mfccs.extend(np.load(mfccs_filename))
        self._labels.extend(np.load(labels_filename))

    @staticmethod
    def get_available_gpu() -> None:
        # print("Available devices:", len(tensorflow.config.list_physical_devices('GPU')))
        # for device in tensorflow.config.list_physical_devices('GPU'):
        #     print(f"Device found: {device}")
        print("Default backend:", jax.default_backend())
        print("Available devices:", jax.devices())

    def init_model(self, summary=True) -> None:
        keras.backend.clear_session()
        self.model = keras.models.Sequential([
            keras.layers.Input(shape=(40,)),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(len(Emotion), activation='softmax')
        ])
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        if summary:
            self.model.summary()

    def fit(self, evaluate=False, show_plot=False, verbose=2) -> Any:
        if self.model is None:
            raise ValueError("Model is not initialized. Call init_model() first.")
        fit_history = self.model.fit(
            np.array(self._mfccs, dtype=np.float32),
            np.array(self._labels, dtype=np.int32),
            validation_split=0.2,
            epochs=50,
            batch_size=128,
            verbose=verbose,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                              restore_best_weights=True)
            ]
        )
        if evaluate:
            loss, accuracy = self.evaluate(verbose=verbose)
            print(f"Train loss: {loss}")
            print(f"Train accuracy: {accuracy}")
        if show_plot:
            import matplotlib.pyplot as plt
            plt.subplots(1, 2, figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(fit_history.history['loss'], label='loss')
            plt.plot(fit_history.history['val_loss'], label='val_loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(fit_history.history['accuracy'], label='accuracy')
            plt.plot(fit_history.history['val_accuracy'], label='val_accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.ylim(0, 1.0)
            plt.show()
        print(
            f"Approx. model memory: {self.model.count_params() * np.dtype(self.model.trainable_weights[0].dtype).itemsize / 1024 ** 2:.2f} MB")
        return fit_history

    def evaluate(self, verbose=0) -> tuple[float, float]:
        if self.model is None:
            raise ValueError("Model is not initialized. Call init_model() first.")
        return self.model.evaluate(
            np.array(self._mfccs, dtype=np.float32),
            np.array(self._labels, dtype=np.float32),
            verbose=verbose)

    def predict(self, mfcc: NDArray[np.float32], verbose=0) -> None:
        if self.model is None:
            raise ValueError("Model is not initialized. Call init_model() first.")
        mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension
        prediction = self.model.predict(mfcc, verbose=verbose)[0]
        print(prediction)
        for i, prob in enumerate(prediction):
            print(f"{Emotion(i).name}: {prob * 100:.2f}%")

    def predict_from_file(self, f: str | os.PathLike[str], verbose=0) -> None:
        y, _ = librosa.load(f, sr=self._sample_rate)
        mfcc = self.extract_mfcc(y, self._sample_rate, self._n_mfcc)
        print(f"Predict file: {f}")
        self.predict(mfcc, verbose=verbose)

    def save_model(self, f: str | os.PathLike[str]) -> None:
        if self.model is None:
            raise ValueError("Model is not initialized. Call init_model() first.")
        self.model.save(f)

    def load_model(self, f: str | bytes | os.PathLike[str]) -> None:
        import keras
        self.model = keras.saving.load_model(f)

    def clear_mfccs(self) -> None:
        self._mfccs.clear()
        self._labels.clear()
