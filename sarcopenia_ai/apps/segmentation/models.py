import keras.backend as K
from keras import regularizers, optimizers
from keras.layers import *
from keras.models import Model

from sarcopenia_ai.core import BaseModelWrapper
from sarcopenia_ai.core.layers import GroupNormalization, InstanceNormalization


def get_available_models():
    return {'UNet2D': UNet2D,
            'UNetInception2D': UNetInception2D}


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def conv2d_unit(inp, num_filters=64, kernel_size=3, momentum=0.9, padding="same", norm='batch', dilation_rate=(1, 1)):
    conv = Conv2D(num_filters, kernel_size, padding=padding, dilation_rate=dilation_rate)(inp)
    if norm == 'group':
        conv = GroupNormalization(groups=8, axis=-1, epsilon=0.1)(conv)
    elif norm == 'instance':
        conv = InstanceNormalization()(conv)
    elif norm == 'batch':
        conv = BatchNormalization(momentum=momentum)(conv)
    conv = Activation('relu')(conv)

    return conv


def conv_block(inp, num_filters=64, kernel_size=3, momentum=0.9, padding="same", pool_size=2, norm='batch'):
    conv = conv2d_unit(inp, num_filters, kernel_size, momentum, padding, norm)
    conv = conv2d_unit(conv, num_filters, kernel_size, momentum, padding, norm)

    if pool_size > 1:
        pool = MaxPooling2D(pool_size=pool_size)(conv)
        return conv, pool
    else:
        return conv


def up_conv_block(inp, inp2, num_filters=64, kernel_size=3, momentum=0.9, padding="same", up_size=2):
    def up_conv_unit(inp, num_filters=64, kernel_size=3, momentum=0.9, padding="same"):
        conv = Conv2D(num_filters, kernel_size, padding=padding)(inp)
        conv = BatchNormalization(momentum=momentum)(conv)
        conv = Activation('relu')(conv)
        return conv

    upcov = UpSampling2D(size=up_size, interpolation='bilinear')(inp)
    upcov = concatenate([upcov, inp2], axis=3)
    upcov = up_conv_unit(upcov, num_filters, kernel_size, momentum, padding)
    upcov = up_conv_unit(upcov, num_filters, kernel_size, momentum, padding)

    return upcov


def spatial_pyramid_pooling(inp_layer, num_layers=4,
                            num_filters=64, out_filters=64, kernel_size=3, momentum=0.9, padding="same", norm='group'):
    pool = [inp_layer]
    for l in range(num_layers):
        r = 2 * (l + 1)
        x = conv2d_unit(inp_layer, num_filters, kernel_size, momentum, padding, norm, dilation_rate=(r, r))
        x = conv2d_unit(x, out_filters, kernel_size=1)
        pool += [x]
    return concatenate(pool, axis=3)


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """

    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        kernel_regularizer=regularizers.l2(0.00004),
        name=conv_name)(x)
    x = BatchNormalization(scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def InceptionMixedUnit(x, depth_multiplier=0.5, prefix_name='', idx=0):
    depth = lambda d: max(int(d * depth_multiplier), 16)

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    branch1x1 = conv2d_bn(x, depth(64), 1, 1)

    branch5x5 = conv2d_bn(x, depth(48), 1, 1)
    branch5x5 = conv2d_bn(branch5x5, depth(64), 5, 5)

    branch3x3dbl = conv2d_bn(x, depth(64), 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, depth(96), 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, depth(96), 3, 3)

    branch_pool = AveragePooling2D((3, 3),
                                   strides=(1, 1),
                                   padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, depth(32), 1, 1)

    x = concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name=prefix_name + 'mixed' + str(idx))

    return x


def squeeze(input_layer, ratio=16):
    ch = input_layer._shape_as_list()[-1]
    x = GlobalAveragePooling2D()(input_layer)
    x = Dense(ch // ratio, activation='relu')(x)
    x = Dense(ch, activation='sigmoid')(x)
    x = Multiply()([input_layer, x])

    return x


class UNetInception2D(BaseModelWrapper):

    def __init__(self, model_dir, name='UNet', data_loader=None, config=None, is_multi_gpu=False):
        super(UNetInception2D, self).__init__(model_dir, name, data_loader, config, is_multi_gpu)
        self.custom_objects = {'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss,
                               'GroupNormalization': GroupNormalization}
        self.compile_args['loss'] = dice_coef_loss
        self.compile_args['metrics'] = [dice_coef]
        # self.compile_args['optimizer'] = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True, decay=1e-4)#
        self.depth_multiplier = 0.75
        try:
            self.num_labels = config.num_labels
        except:
            pass
        self.setup_model()

    def define_model(self):
        def conv_unit(inp, num_filters=64, kernel_size=3, momentum=0.9, padding="same"):
            conv = Conv2D(num_filters, kernel_size, padding=padding, kernel_regularizer=regularizers.l2(0.00004))(inp)
            conv = BatchNormalization(momentum=momentum)(conv)
            # conv = GroupNormalization(groups=16, axis=-1, epsilon=0.01)(conv)
            conv = Activation('relu')(conv)
            return conv

        img_input = Input(self.input_shape)

        depth_multiplier = self.depth_multiplier
        depth = lambda d: max(int(d * depth_multiplier), 16)

        img_input_conv = conv2d_bn(img_input, depth(32), 3, 3, strides=(1, 1), padding='same')
        x = conv2d_bn(img_input, depth(32), 3, 3, strides=(2, 2), padding='same')
        x = conv2d_bn(x, depth(32), 3, 3, padding='same')
        conv1 = conv2d_bn(x, depth(64), 3, 3)
        pool1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(conv1)

        x = pool1
        x = conv2d_bn(x, depth(80), 1, 1, padding='same')
        conv2 = conv2d_bn(x, depth(192), 3, 3, padding='same')
        pool2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(conv2)

        mixed1 = InceptionMixedUnit(pool2, depth_multiplier=depth_multiplier, prefix_name='', idx=0)
        mixed2 = InceptionMixedUnit(mixed1, depth_multiplier=depth_multiplier, prefix_name='', idx=1)

        # mixed 3
        x = mixed2
        branch3x3 = conv2d_bn(x, depth(384), 3, 3, strides=(2, 2), padding='same')

        branch3x3dbl = conv2d_bn(x, depth(64), 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, depth(96), 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, depth(96), 3, 3, strides=(2, 2), padding='same')

        branch_pool = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        mixed3 = concatenate(
            [branch3x3, branch3x3dbl, branch_pool],
            axis=3, name='mixed3')

        # mixed 4: 17 x 17 x 768
        x = mixed3
        branch1x1 = conv2d_bn(x, depth(192), 1, 1)

        branch7x7 = conv2d_bn(x, depth(128), 1, 1)
        branch7x7 = conv2d_bn(branch7x7, depth(128), 1, 7)
        branch7x7 = conv2d_bn(branch7x7, depth(192), 7, 1)

        branch7x7dbl = conv2d_bn(x, depth(128), 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, depth(128), 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, depth(128), 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, depth(128), 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, depth(192), 1, 7)

        branch_pool = AveragePooling2D((3, 3),
                                       strides=(1, 1),
                                       padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, depth(192), 1, 1)
        x = concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=3,
            name='mixed4')

        # mixed 5, 6: 17 x 17 x 768
        for i in range(2):
            branch1x1 = conv2d_bn(x, depth(192), 1, 1)

            branch7x7 = conv2d_bn(x, depth(160), 1, 1)
            branch7x7 = conv2d_bn(branch7x7, depth(160), 1, 7)
            branch7x7 = conv2d_bn(branch7x7, depth(192), 7, 1)

            branch7x7dbl = conv2d_bn(x, depth(160), 1, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, depth(160), 7, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, depth(160), 1, 7)
            branch7x7dbl = conv2d_bn(branch7x7dbl, depth(160), 7, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, depth(192), 1, 7)

            branch_pool = AveragePooling2D(
                (3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = conv2d_bn(branch_pool, depth(192), 1, 1)
            x = concatenate(
                [branch1x1, branch7x7, branch7x7dbl, branch_pool],
                axis=3,
                name='mixed' + str(5 + i))

        # mixed 7: 17 x 17 x 768
        branch1x1 = conv2d_bn(x, depth(192), 1, 1)

        branch7x7 = conv2d_bn(x, depth(192), 1, 1)
        branch7x7 = conv2d_bn(branch7x7, depth(192), 1, 7)
        branch7x7 = conv2d_bn(branch7x7, depth(192), 7, 1)

        branch7x7dbl = conv2d_bn(x, depth(192), 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, depth(192), 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, depth(192), 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, depth(192), 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, depth(192), 1, 7)

        branch_pool = AveragePooling2D((3, 3),
                                       strides=(1, 1),
                                       padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, depth(192), 1, 1)
        x = concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=3,
            name='mixed7')

        upconv1 = UpSampling2D(size=2, interpolation='bilinear')(x)
        upconv1 = concatenate([upconv1, mixed2], axis=3)
        upconv1 = conv2d_bn(upconv1, depth(256), 3, 3, padding="same")
        upconv1 = conv2d_bn(upconv1, depth(256), 3, 3, padding="same")
        # upconv1 = conv_unit(upconv1, num_filters=depth(256))

        upconv2 = UpSampling2D(size=2, interpolation='bilinear')(upconv1)
        upconv2 = concatenate([upconv2, conv2], axis=3)
        upconv2 = conv2d_bn(upconv2, depth(128), 3, 3, padding="same")
        upconv2 = conv2d_bn(upconv2, depth(128), 3, 3, padding="same")
        # upconv2 = conv_unit(upconv2, num_filters=depth(128))

        upconv3 = UpSampling2D(size=2, interpolation='bilinear')(upconv2)
        upconv3 = concatenate([upconv3, conv1], axis=3)
        upconv3 = conv2d_bn(upconv3, depth(64), 3, 3, padding="same")
        upconv3 = conv2d_bn(upconv3, depth(64), 3, 3, padding="same")
        # upconv3 = conv_unit(upconv3, num_filters=depth(64))

        upconv4 = UpSampling2D(size=2, interpolation='bilinear')(upconv3)
        upconv4 = concatenate([upconv4, img_input_conv], axis=3)
        upconv4 = conv2d_bn(upconv4, depth(32), 3, 3, padding="same")
        upconv4 = conv2d_bn(upconv4, depth(32), 3, 3, padding="same")
        # upconv4 = conv_unit(upconv4, num_filters=depth(32))

        outs = []
        for i in range(self.num_labels):
            out = Conv2D(1, (1, 1), activation='sigmoid', name='last_conv_' + str(i), padding='same')(upconv4)
            outs.append(out)

        return Model(inputs=[img_input], outputs=outs, name='UNetInception2D')


class UNet2D(BaseModelWrapper):

    def __init__(self, model_dir, name='UNet', data_loader=None, config=None, is_multi_gpu=False):
        super(UNet2D, self).__init__(model_dir, name, data_loader, config, is_multi_gpu)
        self.custom_objects = {'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss,
                               'GroupNormalization': GroupNormalization,
                               'InstanceNormalization': InstanceNormalization}
        self.compile_args['loss'] = dice_coef_loss
        self.compile_args['metrics'] = [dice_coef]
        self.compile_args['optimizer'] = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True, decay=1e-4)
        self.depth_multiplier = 0.5
        try:
            self.num_labels = config.num_labels
        except:
            pass
        self.setup_model()

    def define_model(self):
        inputs = Input(self.input_shape)

        depth = lambda d: max(int(d * self.depth_multiplier), 8)

        conv2, pool2 = conv_block(inputs, num_filters=depth(64), kernel_size=3)
        conv3, pool3 = conv_block(pool2, num_filters=depth(128), kernel_size=3)
        conv4, pool4 = conv_block(pool3, num_filters=depth(256), kernel_size=3)
        conv5, pool5 = conv_block(pool4, num_filters=depth(512), kernel_size=3)

        conv_mid, _ = conv_block(pool5, num_filters=depth(512), kernel_size=3)

        conv5a = up_conv_block(conv_mid, conv5, num_filters=depth(512), kernel_size=3)
        conv6 = up_conv_block(conv5a, conv4, num_filters=depth(256), kernel_size=3)
        conv7 = up_conv_block(conv6, conv3, num_filters=depth(128), kernel_size=3)
        conv8 = up_conv_block(conv7, conv2, num_filters=depth(64), kernel_size=3)

        outs = []
        for i in range(self.num_labels):
            out = Conv2D(1, (1, 1), activation='sigmoid', name='last_conv_' + str(i), padding='same')(conv8)
            outs.append(out)

        return Model(inputs=[inputs], outputs=outs, name='UNet2D')
