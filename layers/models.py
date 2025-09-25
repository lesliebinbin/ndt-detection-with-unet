import keras


def encoder_block(inputs, num_filters, use_batch_norm=False):
    x = keras.layers.Conv2D(num_filters, kernel_size=3, padding="same")(inputs)
    if use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    x = keras.layers.Conv2D(num_filters, kernel_size=3, padding="same", strides=2)(x)
    if use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    return x


def decoder_block(inputs, skip_fatures, num_filters, use_batch_norm=False):
    x = keras.layers.Conv2DTranspose(
        num_filters, kernel_size=(2, 2), strides=2, padding="same"
    )(inputs)
    x = keras.layers.Concatenate()([x, skip_fatures])
    if use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    return x


def unet_model(
    input_shape=(960, 960, 1),
    depth=4,
    initial_filter=64,
    output_channels=1,
    use_batch_norm=False,
    dropout_rate=0.5,
    final_activation="sigmoid",
):

    inputs = keras.layers.Input(shape=input_shape)
    general_filters = [initial_filter * (2**d) for d in range(depth)]

    downscaling_blocks = [inputs]
    for encoder_filter in general_filters:
        block = encoder_block(
            downscaling_blocks[-1], encoder_filter, use_batch_norm=use_batch_norm
        )
        downscaling_blocks.append(block)

    downscaling_blocks[-1] = keras.layers.Dropout(rate=dropout_rate)(
        downscaling_blocks[-1]
    )

    bottlenet_block = keras.layers.Dropout(dropout_rate)(downscaling_blocks[-1])

    reverse_downscaling_blocks = downscaling_blocks[1:-1][::-1]

    reverse_filters = general_filters[::-1][:-1]

    upscaling_blocks = [bottlenet_block]
    for filter, downscaling_block in zip(reverse_filters, reverse_downscaling_blocks):
        block = decoder_block(
            upscaling_blocks[-1],
            downscaling_block,
            filter,
            use_batch_norm=use_batch_norm,
        )
        upscaling_blocks.append(block)

    x = keras.layers.Dropout(rate=dropout_rate)(upscaling_blocks[-1])
    x = keras.layers.UpSampling2D(size=(2, 2))(x)

    x = keras.layers.Conv2D(
        filters=output_channels,
        kernel_size=3,
        padding="same",
        activation=final_activation,
    )(x)

    # s1 = encoder_block(inputs, 64, use_batch_norm=use_batch_norm)
    # s2 = encoder_block(s1, 128, use_batch_norm=use_batch_norm)
    # s3 = encoder_block(s2, 256, use_batch_norm=use_batch_norm)

    # bottlenet = encoder_block(s3, 512, use_batch_norm=use_batch_norm)

    # bottlenet = keras.layers.Dropout(dropout_rate)(bottlenet)

    # d1 = decoder_block(bottlenet, s3, 512, use_batch_norm=use_batch_norm)
    # d2 = decoder_block(d1, s2, 256, use_batch_norm=use_batch_norm)
    # d3 = decoder_block(d2, s1, 128, use_batch_norm=use_batch_norm)
    # d4 = keras.layers.UpSampling2D(size=(2, 2))(d3)

    # d4 = keras.layers.Dropout(dropout_rate)(d4)

    # x = keras.layers.Conv2D(
    #     filters=output_channels, kernel_size=3, padding="same", activation="sigmoid"
    # )(d4)

    return keras.Model(inputs=inputs, outputs=x)
