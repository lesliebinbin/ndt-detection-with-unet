import keras
import numpy as np
import keras_hub
from .backbones import UnetBackbone
from .models import unet_model

# ECA实例
@keras.saving.register_keras_serializable()
class ECALayer(keras.layers.Layer):
    def __init__(self, k_size=3, **kwargs):
        super().__init__(**kwargs)
        self.k_size = k_size
        self.gap = keras.layers.GlobalAveragePooling2D()
        self.reshape1 = keras.layers.Reshape((-1, 1))
        self.conv1d = keras.layers.Conv1D(
            1, kernel_size=k_size, padding="same", use_bias=False
        )
        self.sigmoid = keras.layers.Activation("sigmoid")
        self.reshape2 = keras.layers.Reshape((1, 1, -1))

    def call(self, inputs):
        x = self.gap(inputs)  # [B, C]
        x = self.reshape1(x)  # [B, C, 1]
        x = self.conv1d(x)  # [B, C, 1]
        x = self.sigmoid(x)  # [B, C, 1]
        x = self.reshape2(x)  # [B, 1, 1, C]
        return inputs * x


def build_simple_cnn_with_eca(input_shape, num_classes):

    return keras.Sequential(
        [
            keras.layers.Input(shape=input_shape, name="intput_layer"),
            keras.layers.Rescaling(
                1.0 / 255, name="scaling_layer"
            ),  # Normalize pixel values [0,1]
            keras.layers.Normalization(mean=0.5, variance=0.25, name="normalize_layer"),
            keras.layers.Conv2D(
                32, kernel_size=(3, 3), activation="relu", name="conv2d_01"
            ),
            keras.layers.Conv2D(
                32,
                kernel_size=(3, 3),
                strides=(2, 2),
                activation="relu",
                name="conv2d_downscaling01",
            ),
            keras.layers.Conv2D(
                64, kernel_size=(3, 3), activation="relu", name="conv2d_02"
            ),
            keras.layers.Conv2D(
                64,
                kernel_size=(3, 3),
                activation="relu",
                strides=(2, 2),
                name="conv2d_02_downscaling",
            ),
            keras.layers.Conv2D(
                128, kernel_size=(3, 3), activation="relu", name="conv2d_03"
            ),
            ECALayer(k_size=3, name="eca_01"),  # ECA insert
            keras.layers.Conv2D(
                128,
                kernel_size=(3, 3),
                activation="relu",
                strides=(2, 2),
                name="conv2d_03_downscaling",
            ),
            ECALayer(k_size=3, name="eca_02"),  # ECA insert
            keras.layers.Conv2D(
                256, kernel_size=(3, 3), activation="relu", name="conv2d_04"
            ),
            keras.layers.GlobalAveragePooling2D(name="global_avg_pooling"),
            keras.layers.Dropout(0.5, name="final_dropout"),
            keras.layers.Dense(num_classes, activation="softmax", name="output_layer"),
        ]
    )


def build_simple_cnn(input_shape, num_classes, final_dropout=0.5):

    return keras.Sequential(
        [
            keras.layers.Input(shape=input_shape, name="intput_layer"),
            keras.layers.Rescaling(
                1.0 / 255, name="scaling_layer"
            ),  # Normalize pixel values [0,1]
            keras.layers.Conv2D(
                32, kernel_size=(3, 3), activation="relu", name="conv2d_01"
            ),
           
            keras.layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_01"),
            keras.layers.Conv2D(
                64, kernel_size=(3, 3), activation="relu", name="conv2d_02"
            ),
           
            keras.layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_02"),
            keras.layers.Conv2D(
                128, kernel_size=(3, 3), activation="relu", name="conv2d_03"
            ),
            
            keras.layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_03"),
            keras.layers.Conv2D(
                256, kernel_size=(3, 3), activation="relu", name="conv2d_04"
            ),
            keras.layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_04"),
            keras.layers.BatchNormalization(),
           
            keras.layers.GlobalAveragePooling2D(name="global_avg_pooling"),
            keras.layers.Dropout(final_dropout, name="final_dropout"),
            keras.layers.Dense(num_classes, activation="softmax", name="output_layer"),
        ]
    )


def build_mlp_mixer(
    input_shape=(512, 512, 1),
    num_classes=10,
    patch_size=16,
    hidden_dim=512,
    tokens_mlp_dim=256,
    channels_mlp_dim=2048,
    num_blocks=8,
):
    """
    Builds an MLP-Mixer model for grayscale images.

    Args:
        input_shape (tuple): Shape of the input image (H, W, C).
        num_classes (int): Number of output classes.
        patch_size (int): Size of each image patch.
        hidden_dim (int): Dimension of the patch embeddings.
        tokens_mlp_dim (int): Hidden dimension for token-mixing MLP.
        channels_mlp_dim (int): Hidden dimension for channel-mixing MLP.
        num_blocks (int): Number of Mixer blocks.

    Returns:
        keras.Model: Compiled MLP-Mixer model.
    """
    height, width, channels = input_shape
    assert (
        height % patch_size == 0 and width % patch_size == 0
    ), "Image dimensions must be divisible by patch size"

    num_patches = (height // patch_size) * (width // patch_size)

    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255)(inputs)

    # Patch embedding
    x = keras.layers.Conv2D(
        filters=hidden_dim, kernel_size=patch_size, strides=patch_size, padding="valid"
    )(x)
    x = keras.layers.Reshape((num_patches, hidden_dim))(x)

    # Mixer blocks
    for _ in range(num_blocks):
        # Token-mixing MLP
        y = keras.layers.LayerNormalization()(x)
        y = keras.layers.Permute((2, 1))(y)
        y = keras.layers.Dense(tokens_mlp_dim, activation="gelu")(y)
        y = keras.layers.Dense(num_patches)(y)
        y = keras.layers.Permute((2, 1))(y)
        x = keras.layers.Add()([x, y])

        # Channel-mixing MLP
        y = keras.layers.LayerNormalization()(x)
        y = keras.layers.Dense(channels_mlp_dim, activation="gelu")(y)
        y = keras.layers.Dense(hidden_dim)(y)
        x = keras.layers.Add()([x, y])

    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="mlp_mixer")
    return model


@keras.saving.register_keras_serializable()
class RandomSmoothingLoss(keras.losses.Loss):
    def __init__(
        self, base_loss, smoothing_range=(0.0, 0.1), name="random_smoothing_loss"
    ):
        super().__init__(name=name)
        self.base_loss = base_loss
        self.smoothing_range = smoothing_range

    def call(self, y_true, y_pred):
        smoothing = np.random.uniform(*self.smoothing_range)
        num_classes = y_true.shape[-1]
        y_true_smoothed = y_true * (1.0 - smoothing) + (smoothing / num_classes)
        return self.base_loss(y_true_smoothed, y_pred)


@keras.saving.register_keras_serializable()
class RandomSmoothingModel(keras.Model):
    def compute_loss(
        self,
        x=None,
        y=None,
        y_pred=None,
        sample_weight=None,
        training=True,
    ):
        if training:
            loss = self.loss(y, y_pred, sample_weight)
        else:
            loss = self.loss.base_loss(y, y_pred, sample_weight)
        if self.losses:
            loss += keras.ops.sum(self.losses)
        return loss


@keras.saving.register_keras_serializable()
class CBAM(keras.layers.Layer):
    def __init__(self, reduction_ratio=8, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channel = input_shape[-1]

        # Channel Attention
        self.avg_pool = keras.layers.GlobalAveragePooling2D()
        self.max_pool = keras.layers.GlobalMaxPooling2D()
        self.shared_dense_one = keras.layers.Dense(
            channel // self.reduction_ratio, activation="relu"
        )
        self.shared_dense_two = keras.layers.Dense(channel)

        # Spatial Attention
        self.conv_spatial = keras.layers.Conv2D(
            filters=1, kernel_size=7, padding="same", activation="sigmoid"
        )

    def call(self, inputs):
        # ----- Channel Attention -----
        avg_pool = self.avg_pool(inputs)
        max_pool = self.max_pool(inputs)

        avg_fc = self.shared_dense_two(self.shared_dense_one(avg_pool))
        max_fc = self.shared_dense_two(self.shared_dense_one(max_pool))

        channel_attention = keras.layers.Activation("sigmoid")(avg_fc + max_fc)
        channel_attention = keras.layers.Reshape((1, 1, -1))(channel_attention)
        x = keras.layers.Multiply()([inputs, channel_attention])

        # ----- Spatial Attention -----
        avg_pool_spatial = keras.ops.mean(x, axis=-1, keepdims=True)
        max_pool_spatial = keras.ops.max(x, axis=-1, keepdims=True)
        concat = keras.ops.concatenate([avg_pool_spatial, max_pool_spatial], axis=-1)
        spatial_attention = self.conv_spatial(concat)
        x = keras.layers.Multiply()([x, spatial_attention])

        return x


def build_cbam_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape, name="input_layer")

    x = keras.layers.Rescaling(1.0 / 255, name="rescaling_layer")(inputs)

    x = keras.layers.Conv2D(32, kernel_size=3, activation="relu", name="conv2d_01")(x)
    x = keras.layers.Conv2D(
        32, kernel_size=3, strides=2, name="downscaling_01", activation="relu"
    )(x)

    x = keras.layers.Conv2D(64, kernel_size=3, activation="relu", name="conv2d_02")(x)
    x = keras.layers.Conv2D(
        64, kernel_size=3, strides=2, activation="relu", name="downscaling_02"
    )(x)
    x = CBAM(name="cbam_02")(x)

    x = keras.layers.Conv2D(128, kernel_size=3, activation="relu", name="conv2d_03")(x)
    x = keras.layers.Conv2D(
        128, kernel_size=3, strides=2, activation="relu", name="downscaling_03"
    )(x)
    x = keras.layers.Conv2D(256, kernel_size=3, activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D(name="global_avg")(x)
    x = keras.layers.Dropout(0.5, name="final_dropout")(x)
    outputs = keras.layers.Dense(
        num_classes, activation="softmax", name="final_output"
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def build_simple_unet(input_shape, num_classes=1):
    """
    构建一个轻量级的U-Net分割模型，基于你提供的CNN结构。
    Args:
        input_shape: 输入图像尺寸，例如 (256, 256, 3)
        num_classes: 输出类别数。1（缺陷/背景）使用sigmoid，>1使用softmax。
    """
    # --- 编码器 (Encoder) --- 使用你的CNN结构，但记录中间输出
    inputs = keras.Input(shape=input_shape, name="input_layer")
    x = keras.layers.Rescaling(1.0 / 255, name="scaling_layer")(inputs)
    
    # Block 1
    x1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='enc_conv1_1')(x)
    x1 = keras.layers.BatchNormalization(name='enc_bn1_1')(x1) # 添加BN
    x1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='enc_conv1_2')(x1)
    x1 = keras.layers.BatchNormalization(name='enc_bn1_2')(x1)
    p1 = keras.layers.MaxPooling2D((2, 2), name='enc_pool1')(x1)
    
    # Block 2
    x2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='enc_conv2_1')(p1)
    x2 = keras.layers.BatchNormalization(name='enc_bn2_1')(x2)
    x2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='enc_conv2_2')(x2)
    x2 = keras.layers.BatchNormalization(name='enc_bn2_2')(x2)
    p2 = keras.layers.MaxPooling2D((2, 2), name='enc_pool2')(x2)
    
    # Block 3 (中心瓶颈)
    x3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='enc_conv3_1')(p2)
    x3 = keras.layers.BatchNormalization(name='enc_bn3_1')(x3)
    x3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='enc_conv3_2')(x3)
    x3 = keras.layers.BatchNormalization(name='enc_bn3_2')(x3)
    # 注意：这里我们不再进行池化，保留尺寸以备解码

    # --- 解码器 (Decoder) ---
    # 上采样路径 1 (将特征图尺寸放大2倍)
    u2 = keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', name='dec_deconv2')(x3)
    u2 = keras.layers.Concatenate(name='dec_concat2')([u2, x2]) # 跳跃连接，融合编码器Block2的特征
    u2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='dec_conv2_1')(u2)
    u2 = keras.layers.BatchNormalization(name='dec_bn2_1')(u2)
    u2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='dec_conv2_2')(u2)
    u2 = keras.layers.BatchNormalization(name='dec_bn2_2')(u2)
    
    # 上采样路径 2 (再次将特征图尺寸放大2倍，恢复到原始尺寸的1/2)
    u1 = keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', name='dec_deconv1')(u2)
    u1 = keras.layers.Concatenate(name='dec_concat1')([u1, x1]) # 跳跃连接，融合编码器Block1的特征
    u1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='dec_conv1_1')(u1)
    u1 = keras.layers.BatchNormalization(name='dec_bn1_1')(u1)
    u1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='dec_conv1_2')(u1)
    u1 = keras.layers.BatchNormalization(name='dec_bn1_2')(u1)
    
    # --- 最终上采样和输出 ---
    # 如果还需要放大到原始尺寸，可以使用UpSampling2D
    # 计算最终上采样倍数：如果原始输入是HxW，而u1是(H/2)x(W/2)，则需要2倍上采样
    # outputs = layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name='final_upsample')(u1)
    
    # 使用1x1卷积将通道数映射到类别数，每个像素点都会产生一个类别预测
    outputs = keras.layers.Conv2D(num_classes, (1, 1), activation='sigmoid', padding='same', name='output_layer')(u1)
    # 如果 num_classes > 1，使用 softmax: activation='softmax'

    model = keras.Model(inputs, outputs, name='Simple_U-Net')
    return model



def build_resnet(
    input_shape,
    num_classes,
    preset,
    means=None,
    variances=None,
    should_scale=False,
    base_backbone=None,
):
    inputs = keras.Input(shape=input_shape)
    if should_scale:
        if (means is None) and (variances is None):
            x = keras.layers.Rescaling(1.0 / 255, name="rescaling_layer")(inputs)
        else:
            x = keras.layers.Normalization(mean=means, variance=variances)(inputs)
    else:
        x = inputs

    backbone = keras_hub.models.ResNetBackbone.from_preset(
        preset=preset, name=preset, image_shape=input_shape, load_weights=False
    )
    if base_backbone is not None:
        for new_layer, base_layer in zip(backbone.layers, base_backbone.layers):
            if (
                new_layer.name == base_layer.name
                and new_layer.trainable
                and base_layer.trainable
            ):
                new_layer.set_weights(base_layer.get_weights())
                print(f"setting weights for {new_layer.name} from base backbone")

    x = backbone(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model


def build_efficientnet(input_shape, num_classes, preset, means=None, variances=None, should_scale=True, load_weights=False, unfreeze_last_n_layers=0):
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255, name="rescaling_layer")(inputs)
    if should_scale:
        x = keras.layers.Normalization(mean=means, variance=variances)(inputs)
    else:
        x = inputs

    backbone = keras_hub.models.EfficientNetBackbone.from_preset(
        preset=preset, name=preset, input_shape=input_shape, load_weights=load_weights
    )
    backbone.trainable = False
    if unfreeze_last_n_layers > 0:
            backbone_layers = backbone.layers
            total_layers = len(backbone_layers)
            
            print(f"Backbone总层数: {total_layers}")
            print(f"解冻最后 {unfreeze_last_n_layers} 层:")
            
            # 解冻最后n层
            for i in range(total_layers - unfreeze_last_n_layers, total_layers):
                layer = backbone_layers[i]
                layer.trainable = True
                print(f"  - {layer.name} (可训练)")
    x = backbone(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model


def build_custom_resnet(
    input_conv_filters=[64],  # Filters in the initial conv layer
    input_conv_kernel_sizes=[7],  # Kernel size for initial conv
    stackwise_num_filters=[64],  # Filters for the single residual stage
    stackwise_num_blocks=[1],  # 1 block in the only stage
    stackwise_num_strides=[1],  # No downsampling (stride=1)
    block_type="basic_block",  # Basic residual block (no bottleneck)
    use_pre_activation=False,  # Vanilla ResNet (post-activation)
    input_shape=(512, 512, 1),  # Input shape (grayscale)
    num_classes=10,
):  # Input shape (grayscale))
    backbone = keras_hub.models.ResNetBackbone(
        input_conv_filters=input_conv_filters,
        input_conv_kernel_sizes=input_conv_kernel_sizes,
        stackwise_num_filters=stackwise_num_filters,
        stackwise_num_blocks=stackwise_num_blocks,
        stackwise_num_strides=stackwise_num_strides,
        block_type=block_type,
        use_pre_activation=use_pre_activation,
        image_shape=input_shape,
    )

    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255, name="rescaling_layer")(inputs)
    x = backbone(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model


class Distiller(keras.Model):
    def __init__(self, student, teacher, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(
        self, x=None, y=None, y_pred=None, sample_weight=None, training=False
    ):
        student_loss = self.student_loss_fn(y, y_pred)
        if not training:
            return student_loss

        teacher_pred = self.teacher(x, training=False)

        distillation_loss = self.distillation_loss_fn(
            keras.ops.softmax(teacher_pred / self.temperature, axis=1),
            keras.ops.softmax(y_pred / self.temperature, axis=1),
        ) * (self.temperature**2)

        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        return loss

    def call(self, *args, **kwargs):
        return self.student(*args, **kwargs)


class RandomDistiller(Distiller):
    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha_range=(0.1, 0.5),
        temperature_range=(3, 10),
    ):
        super().compile(
            optimizer=optimizer,
            metrics=metrics,
            student_loss_fn=student_loss_fn,
            distillation_loss_fn=distillation_loss_fn,
        )
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha_range = alpha_range
        self.temperature_range = temperature_range

    def compute_loss(
        self, x=None, y=None, y_pred=None, sample_weight=None, training=False
    ):
        student_loss = self.student_loss_fn(y, y_pred)
        if not training:
            return student_loss

        temperature = np.random.uniform(*self.temperature_range)
        alpha = np.random.uniform(*self.alpha_range)

        teacher_pred = self.teacher(x, training=False)

        distillation_loss = self.distillation_loss_fn(
            keras.ops.softmax(teacher_pred / temperature, axis=1),
            keras.ops.softmax(y_pred / temperature, axis=1),
        ) * (temperature**2)

        loss = alpha * student_loss + (1 - alpha) * distillation_loss
        return loss


class SaveStudentCallback(keras.callbacks.Callback):
    def __init__(self, filepath, monitor="val_acc", save_best_only=True):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best_value = -float("inf") if "acc" in monitor else float("inf")

    def on_epoch_end(self, epoch, logs=None):
        current_value = logs.get(self.monitor)
        if current_value is None:
            return

        # Check if current model is better
        if self.save_best_only and (
            (self.monitor.endswith("acc") and current_value > self.best_value)
            or (not self.monitor.endswith("acc") and current_value < self.best_value)
        ):
            self.best_value = current_value
            student_model = self.model.student  # Extract student from Distiller
            student_model.save(self.filepath)


class CSPBlock(keras.layers.Layer):
    """A simple CSP block: Conv -> BN -> Swish -> Conv -> BN -> Swish + shortcut."""

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = keras.layers.Conv2D(filters, 1, 1, padding="same", use_bias=False)
        self.bn1 = keras.layers.BatchNormalization()
        self.act1 = keras.layers.Activation("swish")
        self.conv2 = keras.layers.Conv2D(filters, 3, 1, padding="same", use_bias=False)
        self.bn2 = keras.layers.BatchNormalization()
        self.act2 = keras.layers.Activation("swish")
        self.add = keras.layers.Add()

    def call(self, inputs, training=False):
        shortcut = inputs
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        x = self.add([x, shortcut])
        return x


class YOLOV8Backbone(keras.Model):
    """YOLOv8 Backbone implementation (CSPDarkNet-like) for Keras 3+."""

    def __init__(
        self,
        input_shape=(640, 640, 3),
        depth_mult=1.0,
        width_mult=1.0,
        include_rescaling=True,
        pretrained=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_shape_ = input_shape
        self.depth_mult = depth_mult
        self.width_mult = width_mult
        self.include_rescaling = include_rescaling
        self.pretrained = pretrained

        # Stem
        self.rescale = (
            keras.layers.Rescaling(1.0 / 255.0)
            if include_rescaling
            else keras.layers.Lambda(lambda x: x)
        )
        self.stem_conv = keras.layers.Conv2D(
            self._make_divisible(32 * width_mult), 3, 2, padding="same", use_bias=False
        )
        self.stem_bn = keras.layers.BatchNormalization()
        self.stem_act = keras.layers.Activation("swish")

        # Stages (C2, C3, C4, C5)
        self.stage_configs = [
            # (num_blocks, out_channels, stride)
            (1, 64, 2),
            (2, 128, 2),
            (3, 256, 2),
            (1, 512, 2),
        ]
        self.stages = []
        for i, (n, c, s) in enumerate(self.stage_configs):
            blocks = []
            out_channels = self._make_divisible(c * width_mult)
            num_blocks = max(round(n * depth_mult), 1)
            blocks.append(
                keras.layers.Conv2D(out_channels, 3, s, padding="same", use_bias=False)
            )
            blocks.append(keras.layers.BatchNormalization())
            blocks.append(keras.layers.Activation("swish"))
            for _ in range(num_blocks):
                blocks.append(CSPBlock(out_channels))
            self.stages.append(keras.Sequential(blocks, name=f"stage_{i+2}"))

    def call(self, inputs, training=False):
        x = self.rescale(inputs)
        x = self.stem_conv(x)
        x = self.stem_bn(x, training=training)
        x = self.stem_act(x)
        for stage in self.stages:
            x = stage(x, training=training)
        return x

    def _make_divisible(self, v, divisor=8):
        return int((v + divisor / 2) // divisor * divisor)