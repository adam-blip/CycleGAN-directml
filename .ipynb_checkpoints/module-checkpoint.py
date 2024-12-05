import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras

BATCH_NORM_EPSILON = 1e-3
BATCH_NORM_MOMENTUM = 0.97


# TODO(ianstenbit): Remove this method once we're using CSPDarkNet backbone
# (Calls to it should be inlined in the detector head)
def apply_conv_bn(
    inputs,
    output_channel=3,
    kernel_size=1,
    strides=1,
    activation="swish",
    name="conv_bn",
):
    if kernel_size > 1:
        inputs = keras.layers.ZeroPadding2D(
            padding=kernel_size // 2, name=f"{name}_pad"
        )(inputs)

    x = keras.layers.Conv2D(
        filters=output_channel,
        kernel_size=kernel_size,
        strides=strides,
        padding="valid",
        use_bias=False,
        name=f"{name}_conv",
    )(inputs)
    x = keras.layers.BatchNormalization(
        momentum=BATCH_NORM_MOMENTUM,
        epsilon=BATCH_NORM_EPSILON,
        name=f"{name}_bn",
    )(x)
    x = keras.layers.Activation(activation, name=name)(x)
    return x

# TODO(ianstenbit): Remove this method once we're using CSPDarkNet backbone
# Calls to it should instead call the CSP block from the DarkNet implementation.
def apply_csp_block(
    inputs,
    channels=-1,
    depth=2,
    shortcut=True,
    expansion=0.5,
    activation="swish",
    name="csp_block",
):
    channel_axis = -1
    channels = channels if channels > 0 else inputs.shape[channel_axis]
    hidden_channels = int(channels * expansion)

    pre = apply_conv_bn(
        inputs,
        hidden_channels * 2,
        kernel_size=1,
        activation=activation,
        name=f"{name}_pre",
    )
    short, deep = keras.layers.Split(pre, 2, axis=channel_axis)

    out = [short, deep]
    for id in range(depth):
        deep = apply_conv_bn(
            deep,
            hidden_channels,
            kernel_size=3,
            activation=activation,
            name=f"{name}_pre_{id}_1",
        )
        deep = apply_conv_bn(
            deep,
            hidden_channels,
            kernel_size=3,
            activation=activation,
            name=f"{name}_pre_{id}_2",
        )
        deep = (out[-1] + deep) if shortcut else deep
        out.append(deep)
    out = ops.concatenate(out, axis=channel_axis)
    out = apply_conv_bn(
        out,
        channels,
        kernel_size=1,
        activation=activation,
        name=f"{name}_output",
    )
    return out

def apply_spatial_pyramid_pooling_fast(
    inputs, pool_size=5, activation="swish", name="spp_fast"
):
    channel_axis = -1
    input_channels = inputs.shape[channel_axis]
    hidden_channels = int(input_channels // 2)

    x = apply_conv_bn(
        inputs,
        hidden_channels,
        kernel_size=1,
        activation=activation,
        name=f"{name}_pre",
    )
    pool_1 = keras.layers.MaxPooling2D(
        pool_size=pool_size, strides=1, padding="same", name=f"{name}_pool1"
    )(x)
    pool_2 = keras.layers.MaxPooling2D(
        pool_size=pool_size, strides=1, padding="same", name=f"{name}_pool2"
    )(pool_1)
    pool_3 = keras.layers.MaxPooling2D(
        pool_size=pool_size, strides=1, padding="same", name=f"{name}_pool3"
    )(pool_2)

    out = ops.concatenate([x, pool_1, pool_2, pool_3], axis=channel_axis)
    out = apply_conv_bn(
        out,
        input_channels,
        kernel_size=1,
        activation=activation,
        name=f"{name}_output",
    )
    return out

# ==============================================================================
# =                                  networks                                  =
# ==============================================================================

def _get_norm_layer(norm):
    if norm == 'none':
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return keras.layers.BatchNormalization
    elif norm == 'instance_norm':
        return tfa.layers.InstanceNormalization
    elif norm == 'layer_norm':
        return keras.layers.LayerNormalization


def ResnetGenerator(input_shape=(256, 256, 3),
                    output_channels=3,
                    dim=64,
                    n_downsamplings=2,
                    n_blocks=9,
                    norm='instance_norm'):
    Norm = _get_norm_layer(norm)

    def _residual_block(x):
        dim = x.shape[-1]
        h = x

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        h = Norm()(h)

        return keras.layers.add([x, h])

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(dim, 7, padding='valid', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.relu(h)

    # 2
    for _ in range(n_downsamplings):
        dim *= 2
        h = keras.layers.Conv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    # 3
    for _ in range(n_blocks):
        h = _residual_block(h)

    # 4
    for _ in range(n_downsamplings):
        dim //= 2
        h = keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    # 5
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(output_channels, 7, padding='valid')(h)
    h = tf.tanh(h)

    return keras.Model(inputs=inputs, outputs=h)


def ConvDiscriminator(input_shape=(256, 256, 3),
                      dim=64,
                      n_downsamplings=3,
                      norm='instance_norm'):
    dim_ = dim
    Norm = _get_norm_layer(norm)

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1
    h = keras.layers.Conv2D(dim, 4, strides=2, padding='same')(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    for _ in range(n_downsamplings - 1):
        dim = min(dim * 2, dim_ * 8)
        h = keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.leaky_relu(h, alpha=0.2)

    # 2
    dim = min(dim * 2, dim_ * 8)
    h = keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    # 3
    h = keras.layers.Conv2D(1, 4, strides=1, padding='same')(h)

    return keras.Model(inputs=inputs, outputs=h)

def yoloGenerator(input_shape=(256, 256, 3),):
    h = inputs = keras.Input(shape=input_shape)
    keras.layers.Conv2D(3, 3, strides=2, padding='same', activation='swish')(h)

    
    
    '''
    h = apply_csp_block(
        h,
        depth=2,
        expansion=0.5,
        activation='swish',
    )
    h = apply_spatial_pyramid_pooling_fast(
        h,
        pool_size=5,
        activation='swish',
    )
    '''
    return keras.Model(inputs=inputs, outputs=h)

#def yoloDiscriminator
# ==============================================================================
# =                          learning rate scheduler                           =
# ==============================================================================

class LinearDecay(keras.optimizers.schedules.LearningRateSchedule):
    # if `step` < `step_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero

    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: self._initial_learning_rate * (1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)),
            false_fn=lambda: self._initial_learning_rate
        ))
        return self.current_learning_rate
