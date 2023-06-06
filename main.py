# Import the necessary libraries
import pandas as pd
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from transformers import BertTokenizerFast, TFBertForSequenceClassification

# Load the training data
df_train = pd.read_csv('train.csv')

# Split the training data into train and validation sets
train_data, val_data = train_test_split(df_train, test_size=0.1, random_state=42)

# Initialize the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

# Tokenize the text data and create attention masks
max_length = 128
train_encodings = tokenizer(train_data['text'].tolist(), truncation=True, padding='max_length', max_length=max_length)
val_encodings = tokenizer(val_data['text'].tolist(), truncation=True, padding='max_length', max_length=max_length)

# Convert the tokenized data into a tensorflow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_data['target'].tolist()
))
val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    val_data['target'].tolist()
))

# Initialize the BERT model
model = TFBertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)

# Compile the model
model.compile(optimizer=Adam(learning_rate=5e-5),
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=[SparseCategoricalAccuracy()])

# Initialize the early stopping callback
early_stopping = EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=1, restore_best_weights=True)


# Train the model
batch_size = 64
model.fit(train_dataset.shuffle(100).batch(batch_size),
          epochs=5,
          batch_size=batch_size,
          validation_data=val_dataset.batch(batch_size), # validation data needs to be batched too
          callbacks=[early_stopping])

# Load the test data
df_test = pd.read_csv('test.csv')

# Tokenize the test data and create attention masks
test_encodings = tokenizer(df_test['text'].tolist(), truncation=True, padding='max_length', max_length=max_length)

# Convert the test data into a tensorflow dataset
test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
))

# Make predictions on the test data
predictions = model.predict(test_dataset.batch(batch_size))

# Convert the predictions into 0s and 1s
predicted_classes = tf.argmax(predictions.logits, axis=1).numpy()

# Now, `predicted_classes` is a numpy array containing the predicted class for each sample in the test set.
# Let's convert this numpy array into a pandas dataframe and save it as a csv file
df_submission = pd.DataFrame({'id': df_test['id'].tolist(), 'target': predicted_classes.tolist()})
df_submission.to_csv('submission.csv', index=False)
