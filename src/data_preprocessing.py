import pandas as pd
import tensorflow as tf
import os
import numpy as np

from tensorflow.keras.layers import TextVectorization

df = pd.read_csv('/Users/rafaeloliveira/code/Github Repos/mlops_comment_toxicity/data/jigsaw-toxic-comment-classification-challenge/train.csv.zip')

X = df['comment_text']
y = df[df.columns[2:]].values

MAX_WORDS = 200_000

vectorizer = TextVectorization(max_tokens=MAX_WORDS,
                               output_sequence_length=1800,
                               output_mode='int')

vectorizer.adapt(X.values)

vectorized_text = vectorizer(X.values)

dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
dataset = dataset.cache()
dataset = dataset.shuffle(160_000)
dataset = dataset.batch(16)
dataset = dataset.prefetch(8)

train = dataset.take(int(len(dataset)*.7))
val = dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
test = dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))
