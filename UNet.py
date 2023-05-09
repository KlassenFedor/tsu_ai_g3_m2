import tensorflow as tf


class UNet:
    def __init__(self):
        self.SAMPLE_SIZE = (256, 256)
        self.OUTPUT_SIZE = (1280, 1028)
        self.CLASSES = 1
        self.unet = None
        self.train_dataset = None
        self.test_dataset = None

    def set_data(self, dataset):
        pass

    def load_images(self, image, mask):
        image = tf.io.read_file(image)
        image = tf.io.decode_jpeg(image)
        image = tf.image.resize(image, self.OUTPUT_SIZE)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = image / 255.0

        mask = tf.io.read_file(mask)
        mask = tf.io.decode_png(mask)
        mask = tf.image.resize(mask, self.OUTPUT_SIZE)
        mask = tf.image.convert_image_dtype(mask, tf.float32)

        masks = []

        for i in range(self.CLASSES):
            masks.append(tf.where(tf.equal(mask, float(i)), 1.0, 0.0))

        masks = tf.stack(masks, axis=2)
        masks = tf.reshape(masks, self.OUTPUT_SIZE + (self.CLASSES,))

        image = tf.image.resize(image, self.SAMPLE_SIZE)
        masks = tf.image.resize(masks, self.SAMPLE_SIZE)

        return image, masks

    def augmentate_images(self, image, masks):
        random_crop = tf.random.uniform((), 0.3, 1)
        image = tf.image.central_crop(image, random_crop)
        masks = tf.image.central_crop(masks, random_crop)

        random_flip = tf.random.uniform((), 0, 1)
        if random_flip >= 0.5:
            image = tf.image.flip_left_right(image)
            masks = tf.image.flip_left_right(masks)

        image = tf.image.resize(image, self.SAMPLE_SIZE)
        masks = tf.image.resize(masks, self.SAMPLE_SIZE)

        return image, masks

    def input_layer(self):
        return tf.keras.layers.Input(shape=self.SAMPLE_SIZE + (3,))

    def output_layer(self, size):
        return tf.keras.layers.Conv2DTranspose(
            self.CLASSES,
            size,
            strides=2,
            padding='same',
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            activation='sigmoid'
        )

    def downsample_block(self, filters, size, batch_norm=True):
        result = tf.keras.Sequential()

        result.add(
            tf.keras.layers.Conv2D(
                filters,
                size,
                strides=2,
                padding='same',
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                use_bias=False
            )
        )

        if batch_norm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result

    def upsample_block(self, filters, size, dropout=False):
        result = tf.keras.Sequential()

        result.add(
            tf.keras.layers.Conv2DTranspose(
                filters,
                size,
                strides=2,
                padding='same',
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                use_bias=False
            )
        )

        result.add(tf.keras.layers.BatchNormalization())

        if dropout:
            result.add(tf.keras.layers.Dropout(0.25))

        result.add(tf.keras.layers.ReLU())

        return result

    def create(self):
        input_layer = self.input_layer()

        downsample_stack= [
            self.downsample_block(64, 4, batch_norm=False),
            self.downsample_block(128, 4),
            self.downsample_block(256, 4),
            self.downsample_block(512, 4),
            self.downsample_block(512, 4),
            self.downsample_block(512, 4),
            self.downsample_block(512, 4),
        ]

        upsample_stack = [
            self.upsample_block(512, 4, dropout=True),
            self.upsample_block(512, 4, dropout=True),
            self.upsample_block(512, 4, dropout=True),
            self.upsample_block(256, 4),
            self.upsample_block(128, 4),
            self.upsample_block(64, 4)
        ]

        out_layer = self.output_layer(4)

        x = input_layer

        downsample_skips = []

        for block in downsample_stack:
            x = block(x)
            downsample_skips.append(x)

        downsample_skips = reversed(downsample_skips[:-1])

        for up_block, down_block in zip(upsample_stack, downsample_skips):
            x = up_block(x)
            x = tf.keras.layers.Concatenate()([x, down_block])

        output_layer = out_layer(x)

        self.unet = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    def compile(self):
        self.unet.compile(optimizer='adam', loss=['binary_crossentropy'], metrics=[])

    def fit(self, epochs_number):
        self.unet.fit(self.train_dataset, validation_data=self.test_dataset, epochs=epochs_number, initial_epoch=0)
        self.unet.save_weights('SemanticSegmentationLesson/networks/unet_like_aug')
