# batchtrainingbooster/core/weights.py
from typing import Optional, Sequence, Tuple, Dict
from threading import Lock
from numpy import (
    ndarray,
    float64,
    isnan,
    floating,
    bincount,
    array,
    asarray,
    unique,
    fromiter,
    issubdtype,
)

KeyType = Tuple[bytes, bytes, float, bool]


class OptimizedWeightCalculator:
    """
    Calcule des poids 'balanced' avec cache.
    - poids(c) = N / (K * n_c)
    - labels_all : cohérence inter-batchs
    - smoothing  : lissage additif
    - normalize  : moyenne des poids = 1
    """

    __slots__ = ("_cache", "_lock")

    def __init__(self) -> None:
        self._cache: Dict[KeyType, ndarray] = {}
        self._lock = Lock()

    @staticmethod
    def _as_1d(y) -> ndarray:
        y = asarray(y)
        if y.ndim != 1:
            y = y.ravel()
        return y

    @staticmethod
    def _validate(y: ndarray) -> None:
        if y.size == 0:
            return
        if issubdtype(y.dtype, floating) and isnan(y).any():
            raise ValueError("y_train_batch contient des NaN.")

    @staticmethod
    def _key_from_arrays(
        classes: ndarray,
        counts: ndarray,
        smoothing: float,
        normalize: bool,
    ) -> KeyType:
        """
        Clé compacte/stable. On canonicalise le dtype des classes pour éviter
        des clés distinctes à contenu identique (int32 vs int64, object vs <U).
        """
        kind = classes.dtype.kind  # 'i','u','f','U','S','O',...
        if kind in ("i", "u"):
            classes_bytes = asarray(classes, dtype="int64").tobytes()
        elif kind == "f":
            classes_bytes = asarray(classes, dtype=float64).tobytes()
        else:  # textes/objets
            classes_bytes = asarray(classes, dtype="U").tobytes()

        counts_bytes = counts.astype(float64, copy=False).tobytes()
        return (classes_bytes, counts_bytes, float(smoothing), bool(normalize))

    def calculate_sample_weights(
        self,
        y_train_batch,
        labels_all: Optional[Sequence] = None,
        smoothing: float = 0.0,
        normalize: bool = True,
    ) -> ndarray:
        y = self._as_1d(y_train_batch)
        self._validate(y)

        if y.size == 0:
            return array([], dtype=float64)

        # Espace des classes + indices inverses
        if labels_all is None:
            classes, inv = unique(y, return_inverse=True)
        else:
            classes = asarray(labels_all)
            idx_map = {c: i for i, c in enumerate(classes)}
            try:
                inv = fromiter((idx_map[c] for c in y), dtype=int, count=y.size)
            except KeyError as e:
                raise ValueError(
                    f"Label inconnu {e.args[0]} par rapport à labels_all={classes}."
                ) from None

        K = classes.shape[0]
        N = y.shape[0]

        # Effectifs par classe
        counts = bincount(inv, minlength=K).astype(float64, copy=False)
        if smoothing > 0.0:
            counts = counts + float(smoothing)
        # borne de sécurité si labels_all fourni et classe absente
        counts[counts == 0.0] = 1.0

        # --- CACHE ---
        cache_key = self._key_from_arrays(classes, counts, smoothing, normalize)
        with self._lock:
            class_weights = self._cache.get(cache_key)
            if class_weights is None:
                class_weights = (N / (K * counts)).astype(float64, copy=False)
                self._cache[cache_key] = class_weights

        # Projection par échantillon
        w = class_weights[inv].astype(float64, copy=False)

        # Normalisation : inutile quand labels_all is None (moyenne déjà = 1)
        if normalize and labels_all is not None and w.size > 0:
            w *= w.size / w.sum()

        return w

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def get_cache_size(self) -> int:
        with self._lock:
            return len(self._cache)
