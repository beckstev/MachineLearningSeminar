from keras.preprocessing.image import ImageDataGenerator


def resize_training(df_train, df_val, batch_size):
    """function which resizes the images in the training and the validation
    dataframe. in order to do this, the minimum of both dataframes is
    calculated and used as the target_size of both resizing processes.
    Additionally, the step sizes for both generators are calculated.

    :param df_train: dataframe with the training images
    :param df_val: dataframe with the validation images
    :param batch_size: desired batch size
    """

    # calculate minimum from height and width for botch dataframes and compare
    min_height_train = min(df_train['height'])
    min_width_train = min(df_train['width'])

    min_height_val = min(df_val['height'])
    min_width_val = min(df_val['width'])

    min_height = min(min_height_train, min_height_val)
    min_width = min(min_width_train, min_width_val)

    # write result in target_size
    target_size = (min_height, min_width)

    # create object of type ImageDataGenerator
    datagen = ImageDataGenerator()
    train_generator = datagen.flow_from_dataframe(dataframe=df_train,
                                                  directory='.',
                                                  x_col='path_to_image',
                                                  y_col='race_label',
                                                  target_size=target_size,
                                                  batch_size=batch_size
                                                  )
    val_generator = datagen.flow_from_dataframe(dataframe=df_val,
                                                directory='.',
                                                x_col='path_to_image',
                                                y_col='race_label',
                                                target_size=target_size,
                                                batch_size=batch_size
                                                )

    # calculate stepsize for both dataframes.
    # simply number of pictures/batch_size

    STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID = val_generator.n//val_generator.batch_size

    return train_generator, val_generator, STEP_SIZE_TRAIN, STEP_SIZE_VALID
