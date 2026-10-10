"""Upsampler (cascade) model architecture (from cascading-diffusion-2.ipynb).

Class names are kept identical to the notebook so that the auto-generated
layer names match the saved `.weights.h5` keys.
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Layer

CHANNELS = 3


class ZeroConv(Layer):
    def __init__(self, filters, kernel_size, padding="same", **kwargs):
        super().__init__(**kwargs)
        self.layer = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            kernel_initializer=tf.keras.initializers.Zeros(),
            bias_initializer=tf.keras.initializers.Zeros(),
        )

    def call(self, x):
        return self.layer(x)


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
        self.norm = layers.GroupNormalization(groups=8)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim // num_heads)
        self.proj_out = ZeroConv(dim, 1)

    def call(self, x):
        b, h, w = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        x_norm = self.norm(x)
        flat = tf.reshape(x_norm, [b, h * w, self.dim])
        attn = self.mha(flat, flat)
        attn = tf.reshape(attn, [b, h, w, self.dim])
        attn = self.proj_out(attn)
        return x + attn


class Block(Layer):
    def __init__(self, dim=64, **kwargs):
        super().__init__(**kwargs)
        self.norm = layers.GroupNormalization(groups=8)
        self.act = layers.Activation("swish")
        self.conv = layers.Conv2D(dim, (3, 3), padding="same")

    def call(self, x, training=True):
        return self.conv(self.act(self.norm(x)))


class ResNetBlock(Layer):
    def __init__(self, out_channels=64, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.norm1 = layers.GroupNormalization(groups=8)
        self.norm2 = layers.GroupNormalization(groups=8)
        self.act = layers.Activation("swish")
        self.conv1 = layers.Conv2D(out_channels, 3, padding="same")
        self.conv2 = ZeroConv(out_channels, 3)
        self.res_conv = layers.Conv2D(out_channels, 1, padding="same")
        self.dropout = layers.Dropout(dropout_rate)
        self.context_dense = layers.Dense(2 * out_channels)
        self.context_act = layers.Activation("swish")

    def call(self, x, context_emb=None, training=True):
        residual = x if x.shape[-1] == self.out_channels else self.res_conv(x)
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        h = self.norm2(h)
        if context_emb is not None:
            c = context_emb[:, 0, :]
            c = self.context_act(c)
            c = self.context_dense(c)
            c = tf.expand_dims(tf.expand_dims(c, 1), 1)
            gamma, beta = tf.split(c, 2, axis=-1)
            h = h * (1.0 + gamma) + beta
        h = self.act(h)
        h = self.dropout(h, training=training)
        h = self.conv2(h)
        return (residual + h) / tf.sqrt(2.0)


def build_unet(img_size):
    dim = 96
    dim_scale_factor = (1, 2, 2, 4)
    ATTN_RESOLUTIONS = (32,)
    num_groups = 8
    channels = CHANNELS

    image_input = layers.Input(shape=(img_size, img_size, channels + channels), name="image_input")
    time_input = layers.Input(shape=(1,), name="time_input")
    blur_input = layers.Input(shape=(1,), name="blur_input")
    label_input = layers.Input(shape=(1,), name="label_input")

    t_emb = TimeEmbedding(dim)(time_input)
    b_emb = TimeEmbedding(dim)(blur_input)
    l_emb = LabelEmbedding(num_classes=2, dim=dim)(label_input)
    context_emb = layers.Concatenate(axis=-1)([t_emb, l_emb, b_emb])

    x = layers.Lambda(lambda tensor: tf.nn.space_to_depth(tensor, block_size=2))(image_input)
    x = layers.Conv2D(dim, kernel_size=7, padding="same")(x)

    skips = []
    dims = [dim * i for i in dim_scale_factor]

    for index, current_dim in enumerate(dims):
        current_res = img_size // (2 ** index)
        x = ResNetBlock(current_dim)(x, context_emb)
        x = ResNetBlock(current_dim)(x, context_emb)
        x = ResNetBlock(current_dim)(x, context_emb)
        if current_res in ATTN_RESOLUTIONS:
            x = Attention(current_dim)(x)
        skips.append(x)
        if index != len(dims) - 1:
            x = layers.Conv2D(current_dim, kernel_size=3, strides=2, padding="same")(x)

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
        x = ResNetBlock(current_dim)(x, context_emb)
        if current_res in ATTN_RESOLUTIONS:
            x = Attention(current_dim)(x)
        if index != len(dims) - 1:
            x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
            x = layers.Conv2D(current_dim, kernel_size=3, padding="same")(x)

    x = layers.GroupNormalization(groups=8)(x)
    x = layers.Activation("swish")(x)
    x = layers.Lambda(lambda tensor: tf.nn.depth_to_space(tensor, block_size=2))(x)
    output = ZeroConv(channels, 3)(x)
    return models.Model(
        inputs=[image_input, time_input, blur_input, label_input], outputs=output, name="CascadeUNet"
    )
