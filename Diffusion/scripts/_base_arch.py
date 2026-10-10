"""Base diffusion model architecture (from diffusion-multiclass.ipynb).

Class names are kept identical to the notebook so that the auto-generated
layer names match the saved `.weights.h5` keys.
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Layer

CHANNELS = 3


class SinusoidalPosEmb(Layer):
    def __init__(self, dim=64, max_positions=10000):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions

    def call(self, x, training=True):
        x = tf.cast(x, tf.float32)
        half_dim = self.dim // 2
        emb = tf.math.log(tf.cast(self.max_positions, tf.float32)) / (half_dim - 1)
        emb = tf.math.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = tf.concat([tf.math.sin(emb), tf.math.cos(emb)], axis=-1)
        return emb


class TimeEmbedding(Layer):
    def __init__(self, dim=64, **kwargs):
        super().__init__(**kwargs)
        self.time_emb = 4 * dim
        self.emb = SinusoidalPosEmb(dim)
        self.dense1 = layers.Dense(self.time_emb)
        self.act = layers.Activation("swish")
        self.dense2 = layers.Dense(self.time_emb)

    def call(self, x, training=True):
        x = self.emb(x)
        x = self.dense1(x)
        x = self.act(x)
        x = self.dense2(x)
        return x


class LabelEmbedding(Layer):
    def __init__(self, num_classes, dim=64, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.time_emb_dim = 4 * dim
        self.emb_layer = layers.Embedding(num_classes, self.time_emb_dim)

    def call(self, x, training=True):
        return self.emb_layer(x)


class Attention(Layer):
    def __init__(self, dim, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=self.head_dim)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        h, w = x.shape[1], x.shape[2]
        flat_x = tf.reshape(x, (batch_size, h * w, self.dim))
        attn_x = self.mha(flat_x, flat_x)
        return tf.reshape(attn_x, (batch_size, h, w, self.dim))


class Block(Layer):
    def __init__(self, dim=64, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(dim, (3, 3), padding="same")
        self.norm = layers.GroupNormalization(groups=8)
        self.act = layers.Activation("swish")

    def call(self, x, training=True):
        return self.act(self.norm(self.conv(x)))


class ResNetBlock(layers.Layer):
    def __init__(self, out_channels=64, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.conv_block1 = Block(out_channels)
        self.conv_block2 = Block(out_channels)
        self.res_conv = layers.Conv2D(out_channels, 1, padding="same")
        self.context_dense = layers.Dense(2 * out_channels)
        self.context_act = layers.Activation("swish")

    def call(self, x, context_emb=None):
        input_x = x
        h = self.conv_block1(input_x)
        if context_emb is not None:
            c = context_emb[:, 0, :]
            c = self.context_act(c)
            c = self.context_dense(c)
            c = tf.expand_dims(tf.expand_dims(c, 1), 1)
            gamma, beta = tf.split(c, num_or_size_splits=2, axis=-1)
            h = h * (1.0 + gamma) + beta
        h = self.conv_block2(h)
        return h + self.res_conv(input_x)


def build_unet(img_size):
    dim = 64
    dim_scale_factor = (1, 2, 4, 8)
    ATTN_RESOLUTIONS = (16, 8)
    num_groups = 8
    channels = CHANNELS

    image_input = layers.Input(shape=(img_size, img_size, channels), name="image_input")
    time_input = layers.Input(shape=(1,), name="time_input")
    label_input = layers.Input(shape=(1,), name="label_input")

    t_emb = TimeEmbedding(dim)(time_input)
    l_emb = LabelEmbedding(num_classes=2, dim=dim)(label_input)
    context_emb = layers.Concatenate(axis=-1)([t_emb, l_emb])

    x = layers.Conv2D(dim, kernel_size=7, padding="same")(image_input)

    skips = []
    dims = [dim * i for i in dim_scale_factor]

    for index, current_dim in enumerate(dims):
        current_res = img_size // (2 ** index)
        x = ResNetBlock(current_dim)(x, context_emb)
        x = ResNetBlock(current_dim)(x, context_emb)
        if current_res in ATTN_RESOLUTIONS:
            attn = Attention(current_dim)(x)
            norm = layers.GroupNormalization(groups=num_groups)(attn)
            x = norm + x
        skips.append(x)
        if index != len(dims) - 1:
            x = layers.Conv2D(current_dim, kernel_size=4, strides=2, padding="same")(x)

    mid_dim = dims[-1]
    x = ResNetBlock(mid_dim)(x, context_emb)
    attn = Attention(mid_dim)(x)
    norm = layers.GroupNormalization(groups=num_groups)(attn)
    x = norm + x
    x = ResNetBlock(mid_dim)(x, context_emb)

    for index, current_dim in enumerate(reversed(dims)):
        current_res = img_size // (2 ** (len(dims) - 1 - index))
        skip_layer = skips.pop()
        x = layers.Concatenate()([skip_layer, x])
        x = ResNetBlock(current_dim)(x, context_emb)
        x = ResNetBlock(current_dim)(x, context_emb)
        if current_res in ATTN_RESOLUTIONS:
            attn = Attention(current_dim)(x)
            norm = layers.GroupNormalization(groups=num_groups)(attn)
            x = norm + x
        if index != len(dims) - 1:
            x = layers.Conv2DTranspose(current_dim, kernel_size=4, strides=2, padding="same")(x)

    x = ResNetBlock(dim)(x, context_emb)
    output = layers.Conv2D(channels, kernel_size=1, padding="same", dtype="float32")(x)
    return models.Model(inputs=[image_input, time_input, label_input], outputs=output, name="UNet")
