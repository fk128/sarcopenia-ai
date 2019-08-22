import keras.backend as K
from keras import regularizers
from keras.applications.vgg16 import VGG16
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam

from sarcopenia_ai.core import BaseModelWrapper


def conv_block(inp, num_filters=64, kernel_size=3, momentum=0.8, padding="same", pool_size=2, num_blocks=1,
               dilation_rate=(1, 1), separable=False):
    def conv_unit(inp, num_filters=64, kernel_size=3, momentum=0.8, padding="same", do_act=True, dilation_rate=(1, 1),
                  separable=False):

        if separable:
            ConvFn = SeparableConv2D
        else:
            ConvFn = Conv2D

        conv = ConvFn(num_filters, kernel_size, padding=padding, dilation_rate=dilation_rate)(inp)
        conv = BatchNormalization(momentum=momentum)(conv)
        if do_act:
            conv = Activation('relu')(conv)

        return conv

    conv = inp
    for i in range(num_blocks):
        conv = conv_unit(conv, num_filters, kernel_size, momentum, padding, True, dilation_rate=dilation_rate,
                         separable=separable)

    if pool_size is not None:
        pool = MaxPooling2D(pool_size=pool_size)(conv)

        return conv, pool
    else:
        return conv


def conv_block1D(inp, num_filters=64, kernel_size=3, momentum=0.8, padding="same", pool_size=2, num_blocks=1,
                 dilation_rate=1):
    def conv_unit(inp, num_filters=64, kernel_size=3, momentum=0.8, padding="same", do_act=True, dilation_rate=1):
        conv = Conv1D(num_filters, kernel_size, padding=padding, dilation_rate=dilation_rate)(inp)
        conv = BatchNormalization(momentum=momentum)(conv)
        if do_act:
            conv = Activation('relu')(conv)

        return conv

    conv = inp
    for i in range(num_blocks):
        conv = conv_unit(conv, num_filters, kernel_size, momentum, padding, True, dilation_rate=dilation_rate)

    if pool_size is not None:
        pool = MaxPooling1D(pool_size=pool_size)(conv)

        return conv, pool
    else:
        return conv


def up_conv_block(inp, num_filters=64, kernel_size=3, momentum=0.8, padding="same", up_size=2, is_residual=False):
    def up_conv_unit(inp, num_filters=64, kernel_size=3, momentum=0.8, padding="same"):
        conv = Conv2D(num_filters, kernel_size, padding=padding)(inp)
        conv = BatchNormalization(momentum=momentum)(conv)
        conv = Activation('relu')(conv)
        return conv

    upcov = UpSampling2D(size=up_size)(inp)

    upcov = up_conv_unit(upcov, num_filters, kernel_size, momentum, padding)
    upcov = up_conv_unit(upcov, num_filters, kernel_size, momentum, padding)
    upcov = up_conv_unit(upcov, num_filters, 1, momentum, padding)
    if is_residual:
        upcov = add([upcov, inp])
        upcov = Activation('relu')(upcov)
    return upcov


def up_conv_block_add(inp, inp2, num_filters=64, kernel_size=3, momentum=0.8, padding="same", up_size=2, num_blocks=3,
                      is_residual=False):
    def up_conv_unit(inp, num_filters=64, kernel_size=3, momentum=0.8, padding="same", do_act=True):
        conv = Conv2D(num_filters, kernel_size, padding=padding)(inp)
        conv = BatchNormalization(momentum=momentum)(conv)
        if do_act:
            conv = Activation('relu')(conv)
        return conv

    if is_residual:
        inp = Conv2D(num_filters, 1, padding=padding)(inp)

    inp = UpSampling2D(size=up_size)(inp)
    upcov = inp
    upcov = concatenate([upcov, inp2], axis=3)

    upcov = SpatialDropout2D(0.25)(upcov)
    for i in range(num_blocks):
        upcov = up_conv_unit(upcov, num_filters, kernel_size, momentum, padding, True)
    # if num_blocks == 1:
    upcov = up_conv_unit(upcov, num_filters, 1, momentum, padding, False)

    if is_residual:
        upcov = add([upcov, inp])
    upcov = Activation('relu')(upcov)

    return upcov


class UNet(BaseModelWrapper):

    def __init__(self, name, config, data_loader, input_shape=(None, None, 1)):
        super(UNet, self).__init__(config, data_loader, name)
        self.model = self.build_model(input_shape)

    def build_model(self, input_shape=(None, None, 1)):
        inputs = Input(input_shape)

        conv2, pool2 = conv_block(inputs, num_filters=64, kernel_size=3, num_blocks=2)
        conv3, pool3 = conv_block(pool2, num_filters=128, kernel_size=3, num_blocks=2)
        conv4, pool4 = conv_block(pool3, num_filters=256, kernel_size=3, num_blocks=2)
        conv5, pool5 = conv_block(pool4, num_filters=256, kernel_size=5, num_blocks=1, pool_size=4)
        #     conv6, pool5 = conv_block(conv5, num_filters=512, kernel_size=1)
        pool5 = SpatialDropout2D(0.25)(pool5)
        conv_mid = conv_block(pool5, num_filters=512, kernel_size=3, num_blocks=2, pool_size=1)

        conv6 = up_conv_block_add(conv_mid, conv5, num_filters=256, kernel_size=5, num_blocks=1, up_size=4)
        conv7 = up_conv_block_add(conv6, conv4, num_filters=128, kernel_size=3, num_blocks=2)
        conv8 = up_conv_block_add(conv7, conv3, num_filters=128, kernel_size=3, num_blocks=4)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid', name='last_conv', padding='same')(conv8)

        model = Model(inputs=[inputs], outputs=[conv10], name=self.name)

        model.compile(optimizer='adam', loss='mse', metrics=['mae'], loss_weights=[1000])

        return model


class UNetFull(BaseModelWrapper):

    def __init__(self, model_dir, name='model', data_loader=None, config=None, is_multi_gpu=False):
        super(UNetFull, self).__init__(model_dir, name, data_loader, config, is_multi_gpu)
        self.compile_args['loss'] = 'mse'
        self.compile_args['metrics'] = ['mae']
        self.setup_model()

    def define_model(self):
        inputs = Input(self.model_input_shape)
        conv2, pool2 = conv_block(inputs, num_filters=32, kernel_size=3, num_blocks=2)
        conv3, pool3 = conv_block(pool2, num_filters=64, kernel_size=3, num_blocks=2)
        conv4, pool4 = conv_block(pool3, num_filters=128, kernel_size=3, num_blocks=2)
        conv5, pool5 = conv_block(pool4, num_filters=256, kernel_size=3, num_blocks=2, pool_size=4)

        conv_mid = conv_block(pool5, num_filters=512, kernel_size=3, num_blocks=2, pool_size=None)

        conv6 = up_conv_block_add(conv_mid, conv5, num_filters=256, kernel_size=3, num_blocks=2, up_size=4)
        conv7 = up_conv_block_add(conv6, conv4, num_filters=128, kernel_size=3, num_blocks=2)
        conv8 = up_conv_block_add(conv7, conv3, num_filters=128, kernel_size=3, num_blocks=2)
        conv9 = up_conv_block_add(conv8, conv2, num_filters=128, kernel_size=3, num_blocks=2)
        conv10 = Conv2D(1, (1, 1), activation='sigmoid', name='last_conv', padding='same')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10], name=self.name)

        return model


class VGG16RegDual(BaseModelWrapper):
    def __init__(self, name, config, data_loader, input_shape=(None, None, 3)):
        super(VGG16RegDual, self).__init__(config, data_loader, name)
        self.model = self.build_model(input_shape)

    def build_model(self, input_shape=(None, None, 3)):
        vgg = VGG16(include_top=False, input_shape=input_shape)

        fc = Flatten()(vgg.output)
        output1 = Dense(1, name='output1')(fc)
        output2 = Dense(1, activation='sigmoid', name='output2')(fc)
        model = Model(vgg.input, [output1, output2], name=self.name)

        model.compile(optimizer=Adam(lr=0.00001), loss={'output1': 'mse', 'output2': 'binary_crossentropy'},
                      metrics={'output1': 'mae', 'output2': 'accuracy'}, loss_weights={'output1': 1, 'output2': 100})

        return model


class VGG16Reg(BaseModelWrapper):
    def __init__(self, name, config, data_loader, input_shape=(None, None, 3)):
        super(VGG16Reg, self).__init__(config, data_loader, name)
        self.model = self.build_model(input_shape)

    def build_model(self, input_shape=(None, None, 3)):
        vgg = VGG16(include_top=False, input_shape=input_shape)

        fc = Flatten()(vgg.output)
        output1 = Dense(1, name='prediction', kernel_regularizer=regularizers.l2(1e-3))(fc)

        model = Model(vgg.input, output1, name=self.name)

        model.compile(optimizer=Adam(lr=0.00001), loss='mse',
                      metrics=['mae'])

        return model


class CNN4Reg(BaseModelWrapper):
    def __init__(self, name, config, data_loader, input_shape=(None, None, 3)):
        super(CNN4Reg, self).__init__(config, data_loader, name)
        self.model = self.build_model(input_shape)

    def build_model(self, input_shape=(128, 512, 1)):
        inputs = Input((128, 512, 1))

        conv2, pool2 = conv_block(inputs, num_filters=16, kernel_size=5, num_blocks=1, pool_size=(1, 2))
        conv3, pool3 = conv_block(pool2, num_filters=32, kernel_size=7, num_blocks=1, pool_size=(1, 2))
        conv4, pool4 = conv_block(pool3, num_filters=32, kernel_size=9, num_blocks=1, pool_size=(1, 2))
        conv5, pool5 = conv_block(pool4, num_filters=32, kernel_size=3, num_blocks=1, pool_size=(1, 2))

        fc = Flatten()(pool5)
        output1 = Dense(1, name='prediction', kernel_regularizer=regularizers.l2(1e-3))(fc)

        model = Model(inputs, output1, name=self.name)
        model.compile(optimizer='adam', loss='mse',
                      metrics=['mae'])

        return model


class _GlobalHorizontalPooling2D(Layer):
    """Abstract class for different global pooling 2D layers.
    """

    def __init__(self, data_format=None, **kwargs):
        super(_GlobalHorizontalPooling2D, self).__init__(**kwargs)
        self.data_format = data_format
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        # if self.data_format == 'channels_last':
        #     return (input_shape[0], input_shape[1], input_shape[2])
        # else:
        return (input_shape[0], input_shape[1], input_shape[3])

    def call(self, inputs):
        raise NotImplementedError

    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(_GlobalHorizontalPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GlobalMaxHorizontalPooling2D(_GlobalHorizontalPooling2D):
    """Global max pooling operation for spatial data.
    # Arguments
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    # Input shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, rows, cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, rows, cols)`
    # Output shape
        2D tensor with shape:
        `(batch_size, channels)`
    """

    def call(self, inputs):
        # if self.data_format == 'channels_last':
        return K.max(inputs, axis=[2])
        # else:
        #     return K.max(inputs, axis=[3])


def up_conv_block_add_1D(inp, inp2, num_filters=64, kernel_size=3, momentum=0.8, padding="same", up_size=2,
                         num_blocks=3, is_residual=False):
    def up_conv_unit(inp, num_filters=64, kernel_size=3, momentum=0.8, padding="same", do_act=True):
        conv = Conv1D(num_filters, kernel_size, padding=padding)(inp)
        conv = BatchNormalization(momentum=momentum)(conv)
        if do_act:
            conv = Activation('relu')(conv)
        return conv

    if is_residual:
        inp = Conv1D(num_filters, 1, padding=padding)(inp)

    inp = UpSampling1D(size=up_size)(inp)
    upcov = inp

    inp2 = GlobalMaxHorizontalPooling2D()(inp2)

    upcov = concatenate([upcov, inp2], axis=2)

    upcov = Dropout(0.25)(upcov)
    for i in range(num_blocks):
        upcov = up_conv_unit(upcov, num_filters, kernel_size, momentum, padding, True)

    upcov = up_conv_unit(upcov, num_filters, 1, momentum, padding, False)

    if is_residual:
        upcov = add([upcov, inp])
    upcov = Activation('relu')(upcov)

    return upcov


class CNNLine(BaseModelWrapper):
    def __init__(self, model_dir, name='CNNLine', data_loader=None, config=None, is_multi_gpu=False):
        super(CNNLine, self).__init__(model_dir, name, data_loader, config, is_multi_gpu)
        self.compile_args['loss'] = 'binary_crossentropy'
        self.compile_args['metrics'] = ['mae', 'accuracy']
        self.compile_args['loss_weights'] = [10000]
        self.custom_objects = {'GlobalMaxHorizontalPooling2D': GlobalMaxHorizontalPooling2D}
        self.setup_model()

    def define_model(self):
        inputs = Input(self.input_shape)

        conv2, pool2 = conv_block(inputs, num_filters=32, kernel_size=3, num_blocks=2)
        conv3, pool3 = conv_block(pool2, num_filters=64, kernel_size=3, num_blocks=2)
        conv4, pool4 = conv_block(pool3, num_filters=128, kernel_size=3, num_blocks=2)
        conv5, pool5 = conv_block(pool4, num_filters=256, kernel_size=5, num_blocks=1, pool_size=4)

        conv_mid = conv_block(pool5, num_filters=512, kernel_size=3, num_blocks=2, pool_size=None)
        conv_mid = GlobalMaxHorizontalPooling2D()(conv_mid)
        conv6 = up_conv_block_add_1D(conv_mid, conv5, num_filters=256, kernel_size=5, num_blocks=1, up_size=4)
        conv7 = up_conv_block_add_1D(conv6, conv4, num_filters=128, kernel_size=3, num_blocks=2)
        conv8 = up_conv_block_add_1D(conv7, conv3, num_filters=128, kernel_size=3, num_blocks=2)
        conv9 = up_conv_block_add_1D(conv8, conv2, num_filters=128, kernel_size=3, num_blocks=2)

        conv10 = Conv1D(1, 1, activation='sigmoid', name='last_conv', padding='same')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10], name=self.name)

        return model


class CNNLineSep(BaseModelWrapper):
    def __init__(self, model_dir, name='CNNLine', data_loader=None, config=None, is_multi_gpu=False):
        super(CNNLineSep, self).__init__(model_dir, name, data_loader, config, is_multi_gpu)
        self.compile_args['loss'] = 'binary_crossentropy'
        self.compile_args['metrics'] = ['mae', 'accuracy']
        self.compile_args['loss_weights'] = [10000]
        self.custom_objects = {'GlobalMaxHorizontalPooling2D': GlobalMaxHorizontalPooling2D}
        self.setup_model()

    def define_model(self):
        depth = lambda d: max(int(d * self.depth_multiplier), 16)

        inputs = Input(self.input_shape)

        conv2, pool2 = conv_block(inputs, num_filters=depth(32), kernel_size=3, num_blocks=2, separable=True)
        conv3, pool3 = conv_block(pool2, num_filters=depth(64), kernel_size=3, num_blocks=2, separable=True)
        conv4, pool4 = conv_block(pool3, num_filters=depth(128), kernel_size=3, num_blocks=2, separable=True)
        conv5, pool5 = conv_block(pool4, num_filters=depth(256), kernel_size=5, num_blocks=1, pool_size=4,
                                  separable=True)

        pool5 = SpatialDropout2D(0.25)(pool5)
        conv_mid = conv_block(pool5, num_filters=depth(512), kernel_size=3, num_blocks=2, pool_size=None,
                              separable=True)
        conv_mid = GlobalMaxHorizontalPooling2D()(conv_mid)
        conv6 = up_conv_block_add_1D(conv_mid, conv5, num_filters=depth(256), kernel_size=5, num_blocks=1, up_size=4)
        conv7 = up_conv_block_add_1D(conv6, conv4, num_filters=depth(128), kernel_size=3, num_blocks=2)
        conv8 = up_conv_block_add_1D(conv7, conv3, num_filters=depth(128), kernel_size=3, num_blocks=2)
        conv9 = up_conv_block_add_1D(conv8, conv2, num_filters=depth(128), kernel_size=3, num_blocks=2)

        conv10 = Conv1D(1, 1, activation='sigmoid', name='last_conv', padding='same')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10], name=self.name)

        return model


class CNNLineDilate(BaseModelWrapper):
    def __init__(self, name, config, data_loader, input_shape=(None, None, 3)):
        super(CNNLineDilate, self).__init__(config, data_loader, name)
        self.model = self.build_model(input_shape)

    def build_model(self, input_shape=(None, None, 1)):
        inputs = Input(input_shape)

        conv2, pool2 = conv_block(inputs, num_filters=32, kernel_size=3, num_blocks=2)
        conv3, pool3 = conv_block(pool2, num_filters=64, kernel_size=3, num_blocks=2)
        conv4, pool4 = conv_block(pool3, num_filters=128, kernel_size=3, num_blocks=2)
        conv5, pool5 = conv_block(pool4, num_filters=256, kernel_size=5, num_blocks=1, pool_size=4)
        #     conv6, pool5 = conv_block(conv5, num_filters=512, kernel_size=1)
        # pool5 = SpatialDropout2D(0.25)(pool5)
        conv_mid = conv_block(pool5, num_filters=512, kernel_size=3, num_blocks=2, pool_size=None)
        conv_mid = conv_block(conv_mid, num_filters=256, kernel_size=(1, 7), num_blocks=1, pool_size=None)
        conv_mid = GlobalMaxHorizontalPooling2D()(conv_mid)
        conv_mid1 = conv_block1D(conv_mid, num_filters=256, kernel_size=3, num_blocks=1, pool_size=None,
                                 dilation_rate=1)
        conv_mid2 = conv_block1D(conv_mid, num_filters=256, kernel_size=3, num_blocks=1, pool_size=None,
                                 dilation_rate=2)
        conv_mid4 = conv_block1D(conv_mid, num_filters=256, kernel_size=3, num_blocks=1, pool_size=None,
                                 dilation_rate=4)
        conv_mid8 = conv_block1D(conv_mid, num_filters=256, kernel_size=3, num_blocks=1, pool_size=None,
                                 dilation_rate=8)
        conv_mid = concatenate([conv_mid1, conv_mid2, conv_mid4, conv_mid8])

        conv6 = up_conv_block_add_1D(conv_mid, conv5, num_filters=256, kernel_size=5, num_blocks=1, up_size=4)
        conv7 = up_conv_block_add_1D(conv6, conv4, num_filters=128, kernel_size=3, num_blocks=2)
        conv8 = up_conv_block_add_1D(conv7, conv3, num_filters=128, kernel_size=3, num_blocks=2)
        conv9 = up_conv_block_add_1D(conv8, conv2, num_filters=128, kernel_size=3, num_blocks=2)

        conv10 = Conv1D(1, 1, activation='sigmoid', name='last_conv', padding='same')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10], name=self.name)

        model.compile(optimizer='adam', loss='mse', metrics=['mae'], loss_weights=[1000])
        return model


class CNNLineDual(BaseModelWrapper):
    def __init__(self, name, config, data_loader, input_shape=(None, None, 3)):
        super(CNNLineDual, self).__init__(config, data_loader, name)
        self.model = self.build_model(input_shape)

    def build_model(self, input_shape=(None, None, 1)):
        def get_shared_model(input_shape):
            inputs = Input(input_shape)
            conv2, pool2 = conv_block(inputs, num_filters=32, kernel_size=3, num_blocks=2)
            conv3, pool3 = conv_block(pool2, num_filters=64, kernel_size=3, num_blocks=2)
            conv4, pool4 = conv_block(pool3, num_filters=128, kernel_size=3, num_blocks=2)
            conv5, pool5 = conv_block(pool4, num_filters=256, kernel_size=5, num_blocks=1, pool_size=4)
            #     conv6, pool5 = conv_block(conv5, num_filters=512, kernel_size=1)
            # pool5 = SpatialDropout2D(0.25)(pool5)
            conv_mid = conv_block(pool5, num_filters=512, kernel_size=3, num_blocks=2, pool_size=None)
            conv_mid = conv_block(conv_mid, num_filters=256, kernel_size=(1, 7), num_blocks=1, pool_size=None)
            conv_mid = GlobalMaxHorizontalPooling2D()(conv_mid)
            conv5 = GlobalMaxHorizontalPooling2D()(conv5)
            conv4 = GlobalMaxHorizontalPooling2D()(conv4)
            conv3 = GlobalMaxHorizontalPooling2D()(conv3)
            conv2 = GlobalMaxHorizontalPooling2D()(conv2)
            model = Model(inputs=[inputs], outputs=[conv_mid, conv5, conv4, conv3, conv2], name='shared_model')
            return model

        inputs1 = Input(input_shape)
        inputs2 = Input(input_shape)

        shared_model = get_shared_model(input_shape)

        output1, conv5_1, conv4_1, conv3_1, conv2_1 = shared_model(inputs1)
        output2, conv5_2, conv4_2, conv3_2, conv2_2 = shared_model(inputs2)
        conv_mid = concatenate([output1, output2], axis=2)
        conv5 = concatenate([conv5_1, conv5_2], axis=2)
        conv4 = concatenate([conv4_1, conv4_2], axis=2)
        conv3 = concatenate([conv3_1, conv3_2], axis=2)
        conv2 = concatenate([conv2_1, conv2_2], axis=2)

        conv_mid = conv_block1D(conv_mid, num_filters=256, kernel_size=3, num_blocks=1, pool_size=None, dilation_rate=1)
        # conv_mid2 = conv_block1D(conv_mid, num_filters=256, kernel_size=3, num_blocks=1, pool_size=None,
        #                          dilation_rate=2)
        # conv_mid4 = conv_block1D(conv_mid, num_filters=256, kernel_size=3, num_blocks=1, pool_size=None,
        #                          dilation_rate=4)
        # conv_mid8 = conv_block1D(conv_mid, num_filters=256, kernel_size=3, num_blocks=1, pool_size=None,
        #                          dilation_rate=8)
        # conv_mid = concatenate([conv_mid1, conv_mid2, conv_mid4, conv_mid8])

        conv6 = up_conv_block_add_1D(conv_mid, conv5, num_filters=256, kernel_size=5, num_blocks=1, up_size=4)
        conv7 = up_conv_block_add_1D(conv6, conv4, num_filters=128, kernel_size=3, num_blocks=2)
        conv8 = up_conv_block_add_1D(conv7, conv3, num_filters=128, kernel_size=3, num_blocks=2)
        conv9 = up_conv_block_add_1D(conv8, conv2, num_filters=128, kernel_size=3, num_blocks=2)

        conv10 = Conv1D(1, 1, activation='sigmoid', name='last_conv', padding='same')(conv9)

        model = Model(inputs=[inputs1, inputs2], outputs=[conv10], name=self.name)

        model.compile(optimizer='adam', loss='mse', metrics=['mae'], loss_weights=[1000])
        return model


available_models = {'UNetFR': UNetFull,
                    'UNetDS': UNet,
                    'VGG16Reg': VGG16Reg,
                    'VGG16Dual': VGG16RegDual,
                    'CNN4Reg': CNN4Reg,
                    'CNNLine': CNNLine,
                    'CNNLineSep': CNNLineSep,
                    }


def get_model(name):
    if name in available_models.keys():
        return available_models[name]
    else:
        print('invalid model name from {}'.format(available_models.keys()))
        exit(0)
        return None


def get_available_models():
    return available_models.keys()
