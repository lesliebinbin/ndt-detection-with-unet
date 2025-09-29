import keras


def __encoder_block(inputs, num_filters, use_batch_norm=False, kernel_size=3):
    x = keras.layers.Conv2D(num_filters, kernel_size=kernel_size, padding="same")(
        inputs
    )
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(num_filters, kernel_size=kernel_size, padding="same")(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    if use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    return x


def __decoder_block(
    inputs, skip_fatures, num_filters, use_batch_norm=False, kernel_size=3
):
    x = keras.layers.Conv2DTranspose(
        num_filters, kernel_size=2, strides=2, padding="same"
    )(inputs)
    x = keras.layers.Concatenate()([x, skip_fatures])
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(num_filters, kernel_size=kernel_size, padding="same")(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(num_filters, kernel_size=kernel_size, padding="same")(x)
    x = keras.layers.Activation("relu")(x)
    if use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    return x


def __unet_model(
    input_shape=(960, 960, 1),
    depth=4,
    initial_filter=64,
    encoder_kernel_size=3,
    decoder_kenel_size=3,
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
            downscaling_blocks[-1],
            encoder_filter,
            use_batch_norm=use_batch_norm,
            kernel_size=encoder_kernel_size,
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
            kernel_size=decoder_kenel_size,
        )
        upscaling_blocks.append(block)

    x = keras.layers.Dropout(rate=dropout_rate)(upscaling_blocks[-1])
    x = keras.layers.UpSampling2D(size=(2, 2))(x)
    x = keras.layers.Activation("relu")(x)

    x = keras.layers.Conv2D(
        filters=output_channels,
        kernel_size=3,
        padding="same",
        activation=final_activation,
    )(x)
    return keras.Model(inputs=inputs, outputs=x)


# -----------------------------
# Building blocks
# -----------------------------


def encoder_block(x, num_filters, use_batch_norm=False, k=3):
    """
    Encoder block with pre-pool skip (as you suggested).
    Returns:
        x: downsampled tensor
        skip: pre-pool feature map for skip connection
    """
    x = keras.layers.Conv2D(
        num_filters,
        k,
        padding="same",
        use_bias=not use_batch_norm,
        kernel_initializer="he_normal",
    )(x)
    if use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    x = keras.layers.Conv2D(
        num_filters,
        k,
        padding="same",
        use_bias=not use_batch_norm,
        kernel_initializer="he_normal",
    )(x)
    if use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    skip = x  # pre-pool skip
    x = keras.layers.MaxPooling2D(pool_size=2)(x)
    return x, skip


def bottleneck_block(x, num_filters, use_batch_norm=False, k=3):
    """Two convs at the bottleneck (no pooling)."""
    x = keras.layers.Conv2D(
        num_filters,
        k,
        padding="same",
        use_bias=not use_batch_norm,
        kernel_initializer="he_normal",
    )(x)
    if use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    x = keras.layers.Conv2D(
        num_filters,
        k,
        padding="same",
        use_bias=not use_batch_norm,
        kernel_initializer="he_normal",
    )(x)
    if use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    return x


def decoder_block(
    x, skip, num_filters, use_batch_norm=False, k=3, upsample="transpose"
):
    """
    Decoder block: upsample -> concat skip -> 2x conv.
    upsample: "transpose" (Conv2DTranspose) or "bilinear" (UpSampling2D + 1x1 Conv)
    """
    if upsample == "transpose":
        x = keras.layers.Conv2DTranspose(
            num_filters,
            kernel_size=2,
            strides=2,
            padding="same",
            kernel_initializer="he_normal",
            use_bias=True,  # bias fine here
        )(x)
    else:
        x = keras.layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        # project channels to num_filters after upsample
        x = keras.layers.Conv2D(
            num_filters,
            kernel_size=1,
            padding="same",
            use_bias=not use_batch_norm,
            kernel_initializer="he_normal",
        )(x)
        if use_batch_norm:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)

    # Concatenate with the corresponding skip feature
    x = keras.layers.Concatenate()([x, skip])

    # Two convs
    x = keras.layers.Conv2D(
        num_filters,
        k,
        padding="same",
        use_bias=not use_batch_norm,
        kernel_initializer="he_normal",
    )(x)
    if use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    x = keras.layers.Conv2D(
        num_filters,
        k,
        padding="same",
        use_bias=not use_batch_norm,
        kernel_initializer="he_normal",
    )(x)
    if use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    return x


# -----------------------------
# U-Net Model
# -----------------------------


def unet_model(
    input_shape=(960, 960, 1),
    depth=4,
    initial_filter=64,
    encoder_kernel_size=3,
    decoder_kernel_size=3,
    output_channels=1,
    use_batch_norm=False,
    dropout_rate=0.0,  # set >0 to enable dropout at bottleneck
    spatial_dropout=True,  # SpatialDropout2D vs Dropout
    upsample_mode="transpose",  # "transpose" or "bilinear"
    final_activation="sigmoid",
):
    """
    U-Net with pre-pool skips, mirrored decoder, and a single (Spatial)Dropout at bottleneck.

    Notes:
    - Make sure input H and W are divisible by 2**depth.
    - width/capacity controlled by `initial_filter`; context controlled by `depth`.
    """
    assert upsample_mode in ("transpose", "bilinear")

    inputs = keras.layers.Input(shape=input_shape)

    # Compute filter progression for encoder levels
    # e.g., depth=4, initial_filter=64 -> [64, 128, 256, 512]
    enc_filters = [initial_filter * (2**d) for d in range(depth)]
    bottleneck_filters = initial_filter * (2**depth)

    # ----- Encoder -----
    x = inputs
    skips = []
    for nf in enc_filters:
        x, skip = encoder_block(
            x, num_filters=nf, use_batch_norm=use_batch_norm, k=encoder_kernel_size
        )
        skips.append(skip)

    # ----- Bottleneck -----
    x = bottleneck_block(
        x,
        num_filters=bottleneck_filters,
        use_batch_norm=use_batch_norm,
        k=encoder_kernel_size,
    )
    if dropout_rate and dropout_rate > 0.0:
        if spatial_dropout:
            x = keras.layers.SpatialDropout2D(rate=dropout_rate)(x)
        else:
            x = keras.layers.Dropout(rate=dropout_rate)(x)

    # ----- Decoder -----
    # Reverse iterate over encoder filters and corresponding skips
    for nf, skip in zip(enc_filters[::-1], skips[::-1]):
        x = decoder_block(
            x,
            skip=skip,
            num_filters=nf,
            use_batch_norm=use_batch_norm,
            k=decoder_kernel_size,
            upsample=upsample_mode,
        )

    # ----- Output head -----
    outputs = keras.layers.Conv2D(
        filters=output_channels,
        kernel_size=1,
        padding="same",
        activation=final_activation,
        kernel_initializer="glorot_uniform",
        name="segmentation_head",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="UNet_PrePoolSkips")
    return model
