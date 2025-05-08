import tensorflow as tf
from tensorflow.keras import layers, models, losses
import numpy as np
import tensorflow_addons as tfa

# --------------------------------------
# ConvNeXt Block
def convnext_block(x, dim):
    shortcut = x
    x = layers.LayerNormalization()(x)
    x = layers.Conv2D(dim, 7, padding='same', groups=dim)(x)  # depthwise
    x = layers.LayerNormalization()(x)
    x = layers.Dense(4 * dim, activation='gelu')(x)
    x = layers.Dense(dim)(x)
    return layers.Add()([x, shortcut])

# --------------------------------------
# Squeeze-and-Excite Block
def squeeze_excite_block(inputs, ratio=16):
    filters = inputs.shape[-1]
    se = layers.GlobalAveragePooling2D()(inputs)
    se = layers.Dense(filters // ratio, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    return layers.Multiply()([inputs, tf.expand_dims(tf.expand_dims(se, 1), 1)])

# --------------------------------------
# Self-Attention Block
def self_attention_block(x, num_heads=4):
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1])(x, x)
    x = layers.Add()([x, attn_output])
    return x

# --------------------------------------
# Spatial Transformer Network
def spatial_transformer_network(input_tensor, name="stn"):
    with tf.name_scope(name):
        loc = layers.Conv2D(16, (7, 7), padding='same', activation='relu')(input_tensor)
        loc = layers.MaxPooling2D((2, 2))(loc)
        loc = layers.Conv2D(32, (5, 5), padding='same', activation='relu')(loc)
        loc = layers.MaxPooling2D((2, 2))(loc)
        loc = layers.Flatten()(loc)
        loc = layers.Dense(64, activation='relu')(loc)

        # Regressor for 6 affine parameters (initialized to identity)
        initial_weights = np.zeros((64, 6), dtype='float32')
        initial_bias = np.array([1, 0, 0, 0, 1, 0], dtype='float32')
        theta = layers.Dense(6, weights=[initial_weights, initial_bias])(loc)

        input_shape = tf.shape(input_tensor)
        H, W = input_tensor.shape[1], input_tensor.shape[2]
        transformed = tfa.image.transform(input_tensor, theta, interpolation='BILINEAR')
        transformed = tf.reshape(transformed, [-1, H, W, input_tensor.shape[-1]])
        return transformed

# --------------------------------------
# Full Emotion Detection Model
def build_emotion_model(input_shape=(48, 48, 1), num_classes=7):
    inputs = tf.keras.Input(shape=input_shape)

    # STN Module (Face Alignment)
    x = spatial_transformer_network(inputs)

    # Initial Conv Block
    x = layers.Conv2D(64, 7, strides=2, padding='same')(x)

    # Stage 1
    for _ in range(2):
        x = convnext_block(x, 64)
    x = squeeze_excite_block(x)
    x = layers.Conv2D(128, 3, strides=2, padding='same')(x)

    # Stage 2
    x = layers.LayerNormalization()(x)
    for _ in range(3):
        x = convnext_block(x, 128)
    shape2d = tf.shape(x)
    x_reshaped = layers.Reshape((-1, x.shape[-1]))(x)
    x_attn = self_attention_block(x_reshaped)
    x = layers.Reshape((shape2d[1], shape2d[2], -1))(x_attn)
    x = squeeze_excite_block(x)
    x = layers.Conv2D(256, 3, strides=2, padding='same')(x)

    # Stage 3
    x = layers.LayerNormalization()(x)
    for _ in range(4):
        x = convnext_block(x, 256)

    # Global Feature Pooling
    avg_pool = layers.GlobalAveragePooling2D()(x)
    max_pool = layers.GlobalMaxPooling2D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])

    # Emotion-specific Attention
    attn = layers.Dense(256, activation='tanh')(x)
    attn = layers.Dense(256, activation='sigmoid')(attn)
    x = layers.Multiply()([x, attn])

    # Final Classifier
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

# --------------------------------------
# Focal Loss with Label Smoothing
def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=7)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1. - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        return tf.reduce_mean(weight * cross_entropy)
    return loss

# --------------------------------------
# Build & Compile Model
model = build_emotion_model()
model.compile(
    optimizer='adam',
    loss=focal_loss(gamma=2.0),
    metrics=['accuracy']
)

model.summary()