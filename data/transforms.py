import collections
import random

import numpy as np
from scipy.ndimage import rotate


class Base(object):
    def sample(self, *shape):
        return shape

    def tf(self, img, k=0):
        return img

    def __call__(self, img, dim=3, reuse=False):
        if not reuse:
            im = img if isinstance(img, np.ndarray) else img[0]
            shape = im.shape[1 : dim + 1]
            self.sample(*shape)

        if isinstance(img, collections.abc.Sequence):
            return [self.tf(x, k) for k, x in enumerate(img)]

        return self.tf(img)

    def __str__(self):
        return "Identity()"


Identity = Base


class RandomRotion(Base):
    def __init__(self, angle_spectrum=10):
        assert isinstance(angle_spectrum, int)
        self.angle_spectrum = angle_spectrum
        self.axes = [(1, 0), (2, 1), (2, 0)]

    def sample(self, *shape):
        self.axes_buffer = self.axes[np.random.choice(list(range(len(self.axes))))]
        self.angle_buffer = np.random.randint(-self.angle_spectrum, self.angle_spectrum)
        return list(shape)

    def tf(self, img, k=0):
        batch_size = img.shape[0]
        for batch_idx in range(batch_size):
            if k == 0:
                channels = [
                    rotate(
                        img[batch_idx, :, :, :, channel_idx],
                        self.angle_buffer,
                        axes=self.axes_buffer,
                        reshape=False,
                        order=0,
                        mode="constant",
                        cval=0.0,
                    )
                    for channel_idx in range(img.shape[4])
                ]
                img[batch_idx, ...] = np.stack(channels, axis=-1)
            elif k == 1:
                img[batch_idx, ...] = rotate(
                    img[batch_idx, ...],
                    self.angle_buffer,
                    axes=self.axes_buffer,
                    reshape=False,
                    order=0,
                    mode="constant",
                    # Outside the rotated label volume is background, not a
                    # negative index that could be interpreted as ET later.
                    cval=0,
                )
        return img

    def __str__(self):
        return f"RandomRotion(axes={self.axes_buffer}, angle={self.angle_buffer})"


class RandomFlip(Base):
    def __init__(self, axis=0):
        self.axis = (1, 2, 3)
        self.x_buffer = None
        self.y_buffer = None
        self.z_buffer = None

    def sample(self, *shape):
        self.x_buffer = np.random.choice([True, False])
        self.y_buffer = np.random.choice([True, False])
        self.z_buffer = np.random.choice([True, False])
        return list(shape)

    def tf(self, img, k=0):
        if self.x_buffer:
            img = np.flip(img, axis=self.axis[0])
        if self.y_buffer:
            img = np.flip(img, axis=self.axis[1])
        if self.z_buffer:
            img = np.flip(img, axis=self.axis[2])
        return img


class CenterCrop(Base):
    def __init__(self, size):
        self.size = size
        self.buffer = None

    def sample(self, *shape):
        size = self.size
        start = [(s - size) // 2 for s in shape]
        self.buffer = [slice(None)] + [slice(s, s + size) for s in start]
        return [size] * len(shape)

    def tf(self, img, k=0):
        return img[tuple(self.buffer)]


class RandCrop3D(CenterCrop):
    def sample(self, *shape):
        assert len(self.size) == 3
        size = list(self.size) if not isinstance(self.size, list) else self.size
        start = [random.randint(0, s - i) for i, s in zip(size, shape)]
        self.buffer = [slice(None)] + [slice(s, s + k) for s, k in zip(start, size)]
        return size

    def __str__(self):
        return f"RandCrop({self.size})"


class RandomIntensityChange(Base):
    def __init__(self, factor):
        shift, scale = factor
        assert (shift > 0) and (scale > 0)
        self.shift = shift
        self.scale = scale

    def tf(self, img, k=0):
        if k == 1:
            return img

        shift_factor = np.random.uniform(
            -self.shift, self.shift, size=[1, img.shape[1], 1, 1, img.shape[4]]
        )
        scale_factor = np.random.uniform(
            1.0 - self.scale,
            1.0 + self.scale,
            size=[1, img.shape[1], 1, 1, img.shape[4]],
        )
        return img * scale_factor + shift_factor

    def __str__(self):
        return "RandomIntensityChange()"


class NumpyType(Base):
    def __init__(self, types, num=-1):
        self.types = types
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img
        return img.astype(self.types[k])

    def __str__(self):
        joined = ", ".join([str(item) for item in self.types])
        return f"NumpyType(({joined}))"


class Compose(Base):
    def __init__(self, ops):
        if not isinstance(ops, collections.abc.Sequence):
            ops = (ops,)
        self.ops = ops

    def sample(self, *shape):
        for op in self.ops:
            shape = op.sample(*shape)
        return shape

    def tf(self, img, k=0):
        for op in self.ops:
            img = op.tf(img, k)
        return img

    def __str__(self):
        ops = ", ".join([str(op) for op in self.ops])
        return f"Compose([{ops}])"
