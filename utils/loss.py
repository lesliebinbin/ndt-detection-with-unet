import keras


@keras.saving.register_keras_serializable()
def dice_coef(y_true, y_pred, eps=1e-6):
    y_true = keras.ops.cast(y_true, "float32")
    y_pred = keras.ops.cast(y_pred, "float32")
    intersection = keras.ops.sum(y_true * y_pred)
    union = keras.ops.sum(y_true) + keras.ops.sum(y_pred)
    dice = (2.0 * intersection + eps) / (union + eps)
    return keras.ops.mean(dice)


@keras.saving.register_keras_serializable()
def iou_coef(y_true, y_pred, eps=1e-6):
    y_true = keras.ops.cast(y_true, "float32")
    y_pred = keras.ops.cast(y_pred, "float32")
    intersection = keras.ops.sum(y_true * y_pred)
    union = keras.ops.sum(y_true + y_pred - y_true * y_pred)
    iou = (intersection + eps) / (union + eps)
    return keras.ops.mean(iou)


@keras.saving.register_keras_serializable()
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


@keras.saving.register_keras_serializable()
def bce_dice_loss(y_true, y_pred, bce_weight=0.5):
    bce_map = keras.losses.binary_crossentropy(y_true, y_pred)
    bce = keras.ops.mean(bce_map)
    return bce_weight * bce + (1 - bce_weight) * dice_loss(y_true, y_pred)


@keras.saving.register_keras_serializable()
def bfce_dice_loss(y_true, y_pred, bfce_weight=0.5):
    bfce_map = keras.losses.binary_focal_crossentropy(y_true, y_pred)
    bfce = keras.ops.mean(bfce_map)
    return bfce_weight * bfce + (1 - bfce_weight) * dice_loss(y_true, y_pred)
