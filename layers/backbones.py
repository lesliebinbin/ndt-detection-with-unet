import keras


@keras.saving.register_keras_serializable()
class UpScalingLayer(keras.Model):
    def __init__(self, filters, kernel_size=3, strides=2, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.conv2dt_01 = keras.layers.Conv2DTranspose(
            filters=filters, strides=strides, kernel_size=kernel_size, padding="same"
        )
        self.bn01 = keras.layers.BatchNormalization()
        self.ac01 = keras.layers.Activation("relu")

    @property
    def ds_block(self):
        return self._ds_block

    @ds_block.setter
    def ds_block(self, value):
        self._ds_block = value

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        x = self.conv2dt_01(inputs)
        if self.ds_block is not None:
            x = keras.layers.concatenate([x, self.ds_block])
        x = self.bn01(x)
        x = self.ac01(x)
        return x


@keras.saving.register_keras_serializable()
class DownScalingLayer(keras.Model):
    def __init__(self, filters, kernel_size=3, strides=2, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.conv01 = keras.layers.SeparableConv2D(
            filters=filters, kernel_size=kernel_size, padding="same"
        )
        self.bn01 = keras.layers.BatchNormalization()
        self.ac01 = keras.layers.Activation("relu")
        self.conv02 = keras.layers.SeparableConv2D(
            filters=filters, kernel_size=kernel_size, padding="same", strides=strides
        )
        self.bn02 = keras.layers.BatchNormalization()
        self.ac02 = keras.layers.Activation("relu")

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        x = self.conv01(inputs)
        x = self.bn01(x)
        x = self.ac01(x)
        x = self.conv02(x)
        x = self.bn02(x)
        x = self.ac02(x)
        return x


@keras.saving.register_keras_serializable()
class UnetBackbone(keras.Model):
    def __init__(self, general_filters=[16, 32, 64], dropout=0.5, **kwargs):
        super().__init__(**kwargs)
        self.general_filters = general_filters
        self.downscaling_layers = [
            DownScalingLayer(conv_filter) for conv_filter in self.general_filters
        ]
        self.upscaling_layers = [
            UpScalingLayer(conv_filter) for conv_filter in self.general_filters
        ]
        self.dropout = dropout
        self.upscaling_dropout = keras.layers.Dropout(self.dropout)
        self.downscaling_dropout = keras.layers.Dropout(self.dropout)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "general_filters": self.general_filters,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        x = inputs
        for downscaling_layer, up_layer in zip(
            self.downscaling_layers, self.upscaling_layers
        ):
            up_layer.ds_block = x
            x = downscaling_layer(x)
        x = self.downscaling_dropout(x)
        for upscaling_layer in self.upscaling_layers[::-1]:
            x = upscaling_layer(x)
        x = self.upscaling_dropout(x)
        return x