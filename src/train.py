import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAvgPool2D

def get_train_generator(df, image_dir, x_col, y_cols, shuffle=True,
                        batch_size=8, seed=1, target_w=320, target_h=320):
    print("getting train generator...")
    image_generator = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True
    )
    generator = image_generator.flow_from_dataframe(
        dataframe=df,
        directory=image_dir,
        x_col=x_col,
        y_col=y_cols,
        class_mode="raw",
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        target_size=(target_w,target_h)
    )
    return generator

def get_valid_generator(valid_df, train_df, image_dir, x_col, y_cols,
                        sample_size=100, batch_size=8, seed=1,
                        target_w=320, target_h=320):
    print("getting train and valid generators...")
    raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=train_df,
        directory=IMAGE_DIR,
        x_col="Image",
        y_col=labels,
        class_mode="raw",
        batch_size=sample_size,
        shuffle=True,
        target_size=(target_w,target_h))

    batch = raw_train_generator.next()
    data_sample = batch[0]

    image_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True)

    image_generator.fit(data_sample)

    valid_generator = image_generator.flow_from_dataframe(
        dataframe=valid_df,
        directory=image_dir,
        x_col=x_col,
        y_col=y_cols,
        class_mode="raw",
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        target_size=(target_w,target_h))

    return valid_generator

def run(fold):
    df = pd.read_csv("../input/train_folds.csv")
    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)
    IMAGE_DIR = "../input/jpeg/train/"
    labels = ['target']
    train_generator = get_train_generator(
        train_df,
        IMAGE_DIR,
        "image_name",
        labels
    )
    valid_generator = get_valid_generator(
        valid_df,
        train_df,
        IMAGE_DIR,
        "Image",
        labels
    )

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)
