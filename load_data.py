import os
from typing import Optional, Tuple, List
from loguru import logger
import pandas as pd
import tensorflow as tf


class GenreException(Exception):
    def __init__(self, genre_input):
        Exception.__init__(self, f"Selected genre is invalid: {genre_input}")


def load_csv(filename: str, genre: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """
    Load CSV data.  Genre ID for Sci-fi & fantasy is 24.
    Parameters
    ----------
    filename: str
        Filename of CSV file to be loaded.  Do not include full filepath, just name and extension.
    genre: int
        Integer indicating the desired genre.  Genre_dict contains this info but is currently not easily accessed.

    Returns
    -------
    Tuple[List[str], List[str]]
        Tuple of lists of cover filepaths and book titles.
    """
    # DataFrame column names and data types
    columns = {
        "id": str,
        "filename": str,
        "url": str,
        "title": str,
        "author": str,
        "genre id": int,
        "genre name": str
    }
    column_names = list(columns.keys())

    data_fp = os.path.join(os.getcwd(), "data")
    fp = os.path.join(data_fp, filename)
    df = pd.read_csv(fp, names=column_names, dtype=columns, encoding="utf-8", encoding_errors="ignore")

    if genre is not None:
        genre_dict = df[["genre id", "genre name"]].drop_duplicates(keep="first").sort_values(by="genre id"). \
            set_index("genre id").squeeze().to_dict()
        if genre in genre_dict:
            logger.info(f"Genre selected: {genre}, {genre_dict[genre]}")
        else:
            raise GenreException(genre)
        df = df.loc[df["genre id"] == genre]  # Overwrite df for simplicity's sake

    image_fp = (data_fp + os.sep + "covers" + os.sep + df["filename"]).to_list()
    titles = df["title"].to_list()
    return image_fp, titles


def read_image(image_path: str, size: Tuple[int, int]) -> tf.Tensor:
    """
    Ensure images are sized correctly and rescale pixels to in range [-1, 1].
    Book cover images should already be 224x224, but I have not checked.

    Parameters
    ----------
    image_path: str
        Image filepath.
    size: Tuple[int, int]
        Desired image dimensions.

    Returns
    -------
    tf.Tensor
        Image scaled between -1 and 1.
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, size)
    img = (img / 255. - 0.5) / 0.5  # rescale between -1 and 1
    return img


@tf.autograph.experimental.do_not_convert
def make_dataset(image_fp: List[str], titles: List[str], im_size: Tuple[int, int], batch_size: int) -> tf.data.Dataset:
    """

    Parameters
    ----------
    image_fp: List[str]
        List of image filepaths.
    titles: List[str]
        List of book titles, matching the order of the image_fp.
    im_size: Tuple[int, int]
        Desired image size.
    batch_size: int
        Batch size.

    Returns
    -------
    tf.data.Dataset
        TensorFlow dataset.
    """
    buffer_size = 1000
    cache_fn = os.path.join(os.getcwd(), "data", "cache")

    dataset = tf.data.Dataset.from_tensor_slices((titles, image_fp))
    dataset = dataset.map(lambda title, fp: (title, read_image(fp, im_size)), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache().shuffle(buffer_size).batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def load_book30(batch_size: int = 64, image_size: Tuple[int, int] = (224, 224), genre: Optional[int] = None)\
        -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Loads Book30 train and test datasets.  Desired values for the Book30 dataset are the default values.
    Parameters
    ----------
    batch_size: int = 32
        Batch size of resulting tf.data.Datasets
    image_size: Tuple[int, int]
        Desired image height and width, in that order.
    genre: Optional[int] = None
        Integer indicating the desired genre.  Genre_dict contains this info but is currently not easily accessed.

    Returns
    -------
    Tuple[tf.data.Dataset, tf.data.Dataset]
        Book30 train and test datasets
    """
    train_fp, train_titles = load_csv("book30-listing-train.csv", genre=genre)
    test_fp, test_titles = load_csv("book30-listing-test.csv", genre=genre)

    train = make_dataset(train_fp, train_titles, image_size, batch_size)
    test = make_dataset(test_fp, test_titles, image_size, batch_size)

    return train, test
