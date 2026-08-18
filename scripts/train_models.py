import os
# Force Colab to use Keras 2 to maintain compatibility with tensorflow_hub
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
print(tf.__version__)

!git clone https://github.com/Franck-Dernoncourt/pubmed-rct
!ls pubmed-rct

#check files in PubMed_20k_RCT and PubMed_20k_RCT_numbers_replaced_with_at_sign
!ls pubmed-rct/PubMed_20k_RCT/
!ls pubmed-rct/PubMed_20k_RCT_numbers_replaced_with_at_sign/


#assigning directory
data_dir="/content/pubmed-rct/PubMed_20k_RCT_numbers_replaced_with_at_sign"

#check files in target directory
import os
for filename in os.listdir(data_dir):
  print([data_dir+filename])

#creating a function to read the lines of the doc
def get_lines(filename):
  """
  read the lines of the doc
  ARGS:
  filename: a string containing the path to the file
  RETURN:
  a list of strings with one string per line from target file path
  """
  with open(filename,'r') as f:
    lines=f.readlines()
  return lines

#reading training file lines
train_lines=get_lines(data_dir+'/train.txt')
train_lines[:25]

#converting it to data frame
import pandas as pd
train_df=pd.DataFrame(train_lines)
train_df[:10]

input_lines=get_lines(data_dir+'/train.txt')
input_lines[:27]


#creating function to convert dataset into this list of dictionary
def preprocess_data_list(filename):
  """
  returns a list of dictionary that converts file into proper format
  """
  input_lines=get_lines(filename)#getting each line from file
  abstract_lines=""#create an empty abstract
  abstract_samples=[]#create an empty list of abstracts

  for line in input_lines:
    if line.startswith("###"):#ID line check
      abstract_id=line
      abstract_lines=""#resetting abstract string if it the line is an id line
    elif line.isspace():#new line check
      abstract_line_split=abstract_lines.splitlines()#split abstracts into separate lines

      #iterating through each line
      for abstract_line_number,abstract_line in enumerate(abstract_line_split):
        line_data={}#initialising empty string
        target_text_split=abstract_line.split("\t")#split target and text
        line_data["target"]=target_text_split[0]#assign target
        line_data["text"]=" ".join(target_text_split[1:])#assign text
        line_data["line_number"]=abstract_line_number
        line_data["total_lines"]=len(abstract_line_split)-1
        abstract_samples.append(line_data)#append to list


    else:#if above conditions aren't fullfilled
      abstract_lines+=line#add line to abstract


  return abstract_samples







#get samples
train_samples=preprocess_data_list(data_dir+'/train.txt')
val_samples=preprocess_data_list(data_dir+'/dev.txt')
test_samples=preprocess_data_list(data_dir+'/test.txt')



train_samples[:10]

#creating required dataframes

train_df=pd.DataFrame(train_samples)
val_df=pd.DataFrame(val_samples)
test_df=pd.DataFrame(test_samples)
train_df.head()

#distribution of targets
train_df['target'].value_counts()

#length of diffrent lines
train_df['total_lines'].plot.hist()

#convert the abstract text lines into lists
train_lines=train_df['text'].to_list()
val_lines=val_df['text'].to_list()
test_lines=test_df['text'].to_list()

train_lines[:2]

#transformings targets into labels
#one hot labels
from sklearn.preprocessing import OneHotEncoder
OHE=OneHotEncoder()
train_labels_OHE=OHE.fit_transform(train_df[['target']])
val_labels_OHE=OHE.transform(val_df[['target']])
test_labels_OHE=OHE.transform(test_df[['target']])

train_labels_OHE

import tensorflow as tf
train_labels_OHE=tf.constant(train_labels_OHE.toarray(), dtype=tf.float32)
val_labels_OHE=tf.constant(val_labels_OHE.toarray(), dtype=tf.float32)
test_labels_OHE=tf.constant(test_labels_OHE.toarray(), dtype=tf.float32)

train_labels_OHE

#label encoding
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
train_labels=LE.fit_transform(train_df['target'])
val_labels=LE.transform(val_df['target'])
test_labels=LE.transform(test_df['target'])

train_labels

#get class names and number of classes from label encoder
num_classes=len(LE.classes_)
class_names=LE.classes_
print(class_names)
print(num_classes)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

#create a pipeline
model_0=Pipeline([
    ("tfidf",TfidfVectorizer()),
    ("clf",MultinomialNB())
])

#fit the pipeline
model_0.fit(train_lines,train_labels)

#evaluation
baseline_score=model_0.score(val_lines,val_labels)
print(f"Our baseline model achieves an accuracy of: {baseline_score*100:.2f}%")

#make preditions use baseline
baseline_preds=model_0.predict(val_lines)
baseline_preds[:20]

#importing helper funtions from github
!wget https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/helper_functions.py
from helper_functions import unzip_data,create_tensorboard_callback,plot_loss_curves,calculate_results

#calculate baseline results
baseline_results=calculate_results(y_true=val_labels,
                                   y_pred=baseline_preds)
baseline_results

#preparing  our data for sequence models
#finding average length of senences

sent_lens=[len(sentence.split()) for sentence in train_lines]

avg_sen_length=sum(sent_lens)/len(sent_lens)
avg_sen_length

#visualizing sent_lens
import matplotlib.pyplot as plt
plt.hist(sent_lens,bins=10,range=[0,50]);

#how long of sentences length contains 95percent of examples
import numpy as np

out_seq_length=np.percentile(sent_lens,95)
out_seq_length

#max seq length
max_seq_length=max(sent_lens)
max_seq_length

#finding total number of unique words in our vocab
from collections import Counter
word_counts=Counter()
for sent in train_lines:
  word_counts.update(sent.lower().split())


max_tokens=len(word_counts)

max_tokens

#create text vectorizer layer
from tensorflow.keras.layers import TextVectorization
text_vectorizer=TextVectorization(max_tokens=max_tokens,
                                  output_sequence_length=int(out_seq_length),
                                  pad_to_max_tokens=True)



#adapting text vec to training sentences
text_vectorizer.adapt(train_lines)


#test out text vectorizer on random sentences
import random
target_line=random.choice(train_lines)
print(f"Text:\n{target_line}")
print(f"\nLength of text:{len(target_line)}")
print(f"\nVector:{text_vectorizer([target_line])}")

text_vectorizer.get_config()

#create token embedding layer
from tensorflow.keras import layers
embedding=layers.Embedding(input_dim=max_tokens,
                           output_dim=128,
                           input_length=int(out_seq_length),
                           mask_zero=True)

#turning datasets into datasets
train_dataset=tf.data.Dataset.from_tensor_slices((train_lines,train_labels_OHE))
val_dataset=tf.data.Dataset.from_tensor_slices((val_lines,val_labels_OHE))
test_dataset=tf.data.Dataset.from_tensor_slices((test_lines,test_labels_OHE))

#creating batches
train_dataset=train_dataset.batch(32).prefetch(1)
val_dataset=val_dataset.batch(32).prefetch(1)
test_dataset=test_dataset.batch(32).prefetch(1)

#Conv1D with tokens embeddings
from tensorflow.keras import layers
inputs=layers.Input(shape=(1,),dtype=tf.string)
x=text_vectorizer(inputs)
x=embedding(x)
x=layers.Conv1D(filters=64,kernel_size=5,activation="relu")(x)
x=layers.GlobalAveragePooling1D()(x)
outputs=layers.Dense(num_classes,activation="softmax")(x)
model_1=tf.keras.Model(inputs,outputs,name="model_1_Conv1D")
model_1.summary()


#compile the model
model_1.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])


#creating callbacks
from helper_functions import create_tensorboard_callback
SAVE_DIR="model_logs"
tensorboard_callback=create_tensorboard_callback(dir_name=SAVE_DIR,
                                                 experiment_name="Conv1D")

from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint=ModelCheckpoint(filepath='bestmodel.keras',
                           monitor='accuracy',
                           mode='max',
                           save_best_only=True)



#fit the model
history_1=model_1.fit(train_dataset,
                      steps_per_epoch=int(0.1*len(train_dataset)),
                      epochs=20,
                      validation_data=val_dataset,
                      callbacks=[tensorboard_callback,checkpoint])

model_1.evaluate(test_dataset)

#making predictions
model_1_preds=model_1.predict(test_dataset)
model_1_preds[:10],model_1_preds.shape

#convert into pred probs
model_1_pred_probs=tf.squeeze(tf.round(model_1_preds))
model_1_pred_probs=(model_1_pred_probs)
model_1_pred=tf.argmax(model_1_preds,axis=1).numpy()
model_1_pred[:10]

#calculate model_1 results
model_1_results=calculate_results(y_true=test_labels,

                                  y_pred=model_1_pred)
model_1_results,baseline_results

from sentence_transformers import SentenceTransformer
import numpy as np
import tensorflow as tf

# Download the  hugging face MiniLM model
print("Downloading MiniLM...")
hf_model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert sentences to embeddings
print("Converting training sentences...")
train_embeds = hf_model.encode(train_lines, batch_size=32, show_progress_bar=True)
print("Converting validation sentences...")
val_embeds = hf_model.encode(val_lines, batch_size=32, show_progress_bar=True)

# Create the new fast datasets
train_dataset_hf = tf.data.Dataset.from_tensor_slices((train_embeds, train_labels_OHE)).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset_hf = tf.data.Dataset.from_tensor_slices((val_embeds, val_labels_OHE)).batch(32).prefetch(tf.data.AUTOTUNE)


import random
# testing embeddings with MiniLM
random_train_sen = random.choice(train_lines)
print(f"Text:\n{random_train_sen}")

# Using the hf_model to encode the random sentence
random_sen_embd = hf_model.encode([random_train_sen])

print(f"\nsentence embedding shape: {random_sen_embd.shape}")
print(f"sentence embedding (first 20 values):\n{random_sen_embd[0][:20]}")
print(f"\nLength of embedding: {len(random_sen_embd[0])}")

import tensorflow as tf
from tensorflow.keras import layers

# Build Model 2 using pre-computed MiniLM embeddings (384 dimensions)
model_2_input = layers.Input(shape=(384,), dtype=tf.float32, name="embedding_input")
x = layers.Dense(128, activation="relu")(model_2_input)
model_2_output = layers.Dense(num_classes, activation="softmax")(x)

model_2 = tf.keras.Model(inputs=model_2_input, outputs=model_2_output, name="model_2_minilm_feature_extractor")

# Compile the model
model_2.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_2.summary()

#compile model
model_2.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])
#summary
model_2.summary()

# Train Model 2 using the Hugging Face embedding datasets
history_2 = model_2.fit(train_dataset_hf,
                        steps_per_epoch=int(0.1 * len(train_dataset_hf)),
                        epochs=10,
                        validation_data=val_dataset_hf,
                        validation_steps=int(0.1 * len(val_dataset_hf)),
                        callbacks=[tensorboard_callback])

# 1. Create the test embeddings using the HF model
print("Converting test sentences to MiniLM embeddings...")
test_embeds = hf_model.encode(test_lines, batch_size=32, show_progress_bar=True)

# 2. Create the test dataset compatible with model_2_hf
test_dataset_hf = tf.data.Dataset.from_tensor_slices((test_embeds, test_labels_OHE)).batch(32).prefetch(tf.data.AUTOTUNE)

# 3. Evaluate the model using the correct dataset
model_2_hf.evaluate(test_dataset_hf)

# model 2 hf prediction, pred probs and pred labels
model_2_preds = model_2_hf.predict(test_dataset_hf)
model_2_preds[:10], model_2_preds.shape

model_2_pred_probs = tf.squeeze(tf.round(model_2_preds))
model_2_pred_labels = tf.argmax(model_2_preds, axis=1).numpy()
model_2_pred_labels[:10]

# result model_2 using MiniLM
model_2_results = calculate_results(y_true=test_labels,
                                  y_pred=model_2_pred_labels)
model_2_results

#make function to split sentences into chars
def split_chars(text):
  return " ".join(list(text))


#text splitting non-char-level seq into vchars
split_chars(random_train_sen)

#split seq-level datasplits itno character level data splits
train_chars=[split_chars(line) for line in train_lines]
val_chars=[split_chars(line) for line in val_lines]
test_chars=[split_chars(line) for line in test_lines]

from functools import total_ordering
#avg-char length
avg_char_length=np.mean([len(line) for line in train_lines])
avg_char_length

#check the distribution of chars
import matplotlib.pyplot as plt
plt.figure(figsize=(20,15))
plt.hist([len(line) for line in train_lines],bins=20,range=[0,300]);

#find what char length covers 95% seq
max_char_length=np.percentile([len(line) for line in train_lines],95)
max_char_length

#get all keyboard chars
import string
chars=string.ascii_lowercase+string.digits+string.punctuation
chars

#create char level tokenisation
num_char_tokens=len(chars)+2#for space and ['unk']
char_vectorizer=TextVectorization(max_tokens=num_char_tokens,
                                  output_sequence_length=int(max_char_length),
                                  name="char_vectorizer")



#adapt it to train sentences
char_vectorizer.adapt(train_chars)

#check character vocab stats
char_vocab=char_vectorizer.get_vocabulary()
print(f"number fo diff characters :{len(char_vocab)}")
print(f"first 10 characters in vocab:{char_vocab[:10]}")
print(f"last 10 characters in vocab:{char_vocab[-10:]}")


#creating a character embedding
from tensorflow.keras import layers
char_embedding=layers.Embedding(input_dim=len(char_vocab),
                                output_dim=32,
                                mask_zero=True,
                                name="char_embedding")

#test our embedding layer
import random
target_char=random.choice(train_chars)
print(f"Text:\n{target_char}")
print(f"\nLength of text:{len(target_char)}")
print(f"\nVector:{char_vectorizer([target_char])}")
print(f"\nEmbedding:{char_embedding(char_vectorizer([target_char]))}")


#creating model
from tensorflow.keras import layers
inputs=layers.Input(shape=(1,),dtype=tf.string)
x=char_vectorizer(inputs)
x=char_embedding(x)
x=layers.Conv1D(64,
                kernel_size=5,
                padding="same",
                activation="relu")(x)
x=layers.Conv1D(64,
                kernel_size=5,
                padding="same",
                activation="relu")(x)
x=layers.GlobalAveragePooling1D()(x)
x=layers.Dense(128,activation="relu")(x)
outputs=layers.Dense(num_classes,activation="softmax")(x)

model_3=tf.keras.Model(inputs,outputs,name="model_3_char_level_conv")
model_3.summary()

#compile model
model_3.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])


#creating dataset and batching
train_dataset=tf.data.Dataset.from_tensor_slices((train_chars,train_labels_OHE))
val_dataset=tf.data.Dataset.from_tensor_slices((val_chars,val_labels_OHE))
test_dataset=tf.data.Dataset.from_tensor_slices((test_chars,test_labels_OHE))

#creating prefething and batching
train_dataset=train_dataset.batch(32).prefetch(1)
val_dataset=val_dataset.batch(32).prefetch(1)
test_dataset=test_dataset.batch(32).prefetch(1)

#fittting
history_model_3=model_3.fit(train_dataset,
                          steps_per_epoch=int(0.1*len(train_dataset)),
                          epochs=10,
                          validation_data=val_dataset,
                          callbacks=[tensorboard_callback])

#calculating model_3results
model_3_pred=model_3.predict(test_dataset)


#model_3_pred_probs and pred_lables
model_3_pred_probs=tf.squeeze(tf.round(model_3_pred))
model_3_pred_labels=tf.argmax(model_3_pred,axis=1).numpy()
model_3_pred_labels[:10]

#results
model_3_results=calculate_results(y_true=test_labels,
                                  y_pred=model_3_pred_labels)
model_3_results

# combining custom token embd (Model 1) + char embd (Hybrid model)

# 1. Setup token inputs/model using layers from Model 1
token_inputs = layers.Input(shape=(1,), dtype=tf.string, name="token_input_layer")
token_vectors = text_vectorizer(token_inputs)
token_embeddings = embedding(token_vectors)
x_token = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(token_embeddings)
x_token = layers.GlobalAveragePooling1D()(x_token)
token_outputs = layers.Dense(128, activation="relu", name="token_dense_layer")(x_token)
token_model = tf.keras.Model(token_inputs, token_outputs, name="token_model")

# 2. Setup char inputs/model
char_inputs = layers.Input(shape=(1,), dtype=tf.string, name="char_input_layer")
x_char = char_vectorizer(char_inputs)
x_char = char_embedding(x_char)
char_outputs = layers.Bidirectional(layers.LSTM(32), name="bidirectional_layer")(x_char)
char_model = tf.keras.Model(char_inputs, char_outputs, name="char_model")

# 3. Concatenate and add complexity with deeper Dense layers
combined = layers.Concatenate(name="token_char_hybrid")([token_model.output, char_model.output])

#stacking dense layers for better feature extraction
z = layers.Dense(256, activation="relu")(combined)
z = layers.Dropout(0.5)(z)
z = layers.Dense(128, activation="relu")(z)
z = layers.Dropout(0.3)(z)

outputs = layers.Dense(num_classes, activation="softmax", name="output_layer")(z)

model_4 = tf.keras.Model(inputs=[token_model.input, char_model.input],
                         outputs=outputs,
                         name="model_4_hybrid_custom_tokens")

model_4.summary()

#plot hybrid token and character model
import tensorflow as tf
from keras.utils import plot_model

plot_model(model_4,dpi=60,show_shapes=True)

#compile token char model
model_4.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

#creating dataset and prefetched batches
train_dataset_data=tf.data.Dataset.from_tensor_slices((train_lines,train_chars))
val_dataset_data=tf.data.Dataset.from_tensor_slices((val_lines,val_chars))
test_dataset_data=tf.data.Dataset.from_tensor_slices((test_lines,test_chars))

train_dataset_labels=tf.data.Dataset.from_tensor_slices((train_labels_OHE))
val_dataset_labels=tf.data.Dataset.from_tensor_slices((val_labels_OHE))
test_dataset_labels=tf.data.Dataset.from_tensor_slices((test_labels_OHE))

train_dataset=tf.data.Dataset.zip((train_dataset_data,train_dataset_labels))
val_dataset=tf.data.Dataset.zip((val_dataset_data,val_dataset_labels))
test_dataset=tf.data.Dataset.zip((test_dataset_data,test_dataset_labels))

train_dataset=train_dataset.batch(32).prefetch(1)
val_dataset=val_dataset.batch(32).prefetch(1)
test_dataset=test_dataset.batch(32).prefetch(1)

#fitting the model

history_model_4=model_4.fit(train_dataset,
                            steps_per_epoch=int(0.1*len(train_dataset)),
                            epochs=10,
                            validation_data=val_dataset,
                            callbacks=[tensorboard_callback])

#predicting using model 4
model_4_preds=model_4.predict(test_dataset)
model_4_preds[:10],model_4_preds.shape

#pred_probs and pred_labels
model_4_pred_probs=tf.squeeze(tf.round(model_4_preds))
model_4_pred_labels=tf.argmax(model_4_preds,axis=1).numpy()
model_4_pred_labels[:10]

#genrate result dict
model_4_results=calculate_results(y_true=test_labels,
                                  y_pred=model_4_pred_labels)
model_4_results

#checking ditribution of line_number
import matplotlib.pyplot as plt
plt.hist(train_df["line_number"],bins=30,range=[0,30]);
plt.show()

#upto how many lines cover 95 percent of data
np.percentile(train_df["line_number"],95)

#using tensorlow for OHE in line_number
train_line_num_OHE=tf.one_hot(train_df["line_number"].to_numpy(),depth=15)
val_line_num_OHE=tf.one_hot(val_df["line_number"].to_numpy(),depth=15)
test_line_num_OHE=tf.one_hot(test_df["line_number"].to_numpy(),depth=15)

train_line_num_OHE[:10],train_line_num_OHE.shape




#checking distribution of total_lines
plt.hist(train_df["total_lines"],bins=30,range=[0,30]);
plt.show()

#how many lines cover 95 percent of lines
np.percentile(train_df["total_lines"],95)

#OHE total lines
train_total_lines_OHE=tf.one_hot(train_df["total_lines"].to_numpy(),depth=20)
val_total_lines_OHE=tf.one_hot(val_df["total_lines"].to_numpy(),depth=20)
test_total_lines_OHE=tf.one_hot(test_df["total_lines"].to_numpy(),depth=20)

train_total_lines_OHE[:10],train_total_lines_OHE.shape

#making embedding layer

from tensorflow.keras import layers

# 1. Token Branch (Using custom token embeddings from Model 1)
token_inputs = layers.Input(shape=[], dtype=tf.string, name="token_inputs")
token_vectors = text_vectorizer(token_inputs)
token_embeddings = embedding(token_vectors)
x_token = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(token_embeddings)
x_token = layers.GlobalAveragePooling1D()(x_token)
token_outputs = layers.Dense(64, activation="relu", name="token_output_layer")(x_token)
token_model = tf.keras.Model(inputs=token_inputs, outputs=token_outputs, name="token_model")

# 2. Character Branch (Using custom char embeddings from Model 3)
char_inputs = layers.Input(shape=(1,), dtype=tf.string, name="char_input_layer")
x_char = char_vectorizer(char_inputs)
x_char = char_embedding(x_char)
x_char = layers.Bidirectional(layers.LSTM(32), name="bidirectional_layer")(x_char)
char_model = tf.keras.Model(char_inputs, x_char, name="char_model")

# 3. Line Number Branch
line_num_inputs = layers.Input(shape=(15,), dtype=tf.float32, name="line_num_input_layer")
line_num_outputs = layers.Dense(64, activation="relu", name="line_num_output_layer")(line_num_inputs)
line_num_model = tf.keras.Model(line_num_inputs, line_num_outputs, name="line_num_model")

# 4. Total Lines Branch
total_lines_inputs = layers.Input(shape=(20,), dtype=tf.float32, name="total_lines_input_layer")
total_lines_outputs = layers.Dense(64, activation="relu", name="total_lines_output_layer")(total_lines_inputs)
total_lines_model = tf.keras.Model(total_lines_inputs, total_lines_outputs, name="total_lines_model")

# 5. Concatenate and Final Layers
concat_layer = layers.Concatenate(axis=1, name="concat_layer")([token_model.output,
                                                              char_model.output,
                                                              line_num_model.output,
                                                              total_lines_model.output])

# added dense to catch meanings
x = layers.Dropout(0.2)(concat_layer)
x = layers.Dense(512, activation="relu", name="dense_layer_1")(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(256, activation="relu", name="dense_layer_2")(x)
x = layers.Dropout(0.2)(x)
output_layer = layers.Dense(num_classes, activation="softmax", name="output_layer")(x)

model_5 = tf.keras.Model(inputs=[token_model.input,
                                 char_model.input,
                                 line_num_model.input,
                                 total_lines_model.input],
                         outputs=output_layer,
                         name="model_5_quadra_model")

model_5.summary()

#get a summary
model_5.summary()

#plot Quadra Model
import tensorflow as tf
from keras.utils import plot_model

plot_model(model_5,dpi=200,show_shapes=True,show_layer_names=True,show_dtype=False,to_file='Quadra_Model.png',show_layer_activations=True,show_trainable=True)

#compile Quadra Model
model_5.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2 ),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

import tensorflow as tf

# Prepare datasets using dictionary mapping to ensure names match the Input layers exactly
train_dataset = tf.data.Dataset.from_tensor_slices((
    {
        "token_inputs": train_lines,
        "char_input_layer": train_chars,
        "line_num_input_layer": train_line_num_OHE,
        "total_lines_input_layer": train_total_lines_OHE
    },
    train_labels_OHE
)).batch(32).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((
    {
        "token_inputs": val_lines,
        "char_input_layer": val_chars,
        "line_num_input_layer": val_line_num_OHE,
        "total_lines_input_layer": val_total_lines_OHE
    },
    val_labels_OHE
)).batch(32).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((
    {
        "token_inputs": test_lines,
        "char_input_layer": test_chars,
        "line_num_input_layer": test_line_num_OHE,
        "total_lines_input_layer": test_total_lines_OHE
    },
    test_labels_OHE
)).batch(32).prefetch(tf.data.AUTOTUNE)

print("Datasets created with explicit dictionary mapping.")

#using import callbacks
from tensorflow.keras.callbacks import EarlyStopping

early_stopping=EarlyStopping(monitor="val_accuracy",
                             patience=4,
                             mode='max',
                             restore_best_weights=True)

# Re-compiling WITHOUT run_eagerly for maximum speed
model_5.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"],
                run_eagerly=False)

# Re-fitting the model - this will now run much faster on the GPU
history_model_5 = model_5.fit(train_dataset,
                            epochs=10,
                            steps_per_epoch=int(0.1*len(train_dataset)),
                            validation_data=val_dataset,
                            callbacks=[tensorboard_callback, checkpoint, early_stopping])

model_5_test_results = model_5.evaluate(test_dataset)
print(f"Model 5 Test Loss: {model_5_test_results[0]}")
print(f"Model 5 Test Accuracy: {model_5_test_results[1]*100:.2f}%")

# Make predictions to calculate F1-score and other metrics
model_5_test_probs = model_5.predict(test_dataset)
model_5_test_preds = tf.argmax(model_5_test_probs, axis=1)

# Using the helper function to get full results
model_5_final_metrics = calculate_results(y_true=test_labels, y_pred=model_5_test_preds)
display(model_5_final_metrics)

#predictions model_5
model_5_preds=model_5.predict(test_dataset)
model_5_preds[:10],model_5_preds.shape


#pred_probs and labels
model_5_pred_probs=tf.squeeze(tf.round(model_5_preds))
model_5_pred_labels=tf.argmax(model_5_preds,axis=1).numpy()
model_5_pred_labels[:10]

#result dictionary for model 5
model_5_results=calculate_results(y_true=test_labels,
                                  y_pred=model_5_pred_labels)
model_5_results

# dataframe for all model results including the updated MiniLM model
import pandas as pd
results = pd.DataFrame({"Naive_model": baseline_results,
                     "Custom_made_Feature_Ext_model": model_1_results,
                     "MiniLM_Feature_Ext_Model": model_2_results,
                     "Characterised_Token_Model": model_3_results,
                     "Multimodal_Model": model_4_results,
                     "Quadra_Model": model_5_results})

results

import os
import pickle
from google.colab import files

# Create target directory
os.makedirs('saved_models', exist_ok=True)

# 1. Save Model 0 (Naive Bayes pipeline)
with open('saved_models/model_0.pkl', 'wb') as f:
    pickle.dump(model_0, f)
print("Saved Model 0 to saved_models/model_0.pkl")

# 2. Save Model 1 (Conv1D Token-level)
model_1.save('saved_models/model_1.keras')
print("Saved Model 1 to saved_models/model_1.keras")

# 3. Save Model 2 (USE Feature Extractor)
model_2.save('saved_models/model_2.keras')
print("Saved Model 2 to saved_models/model_2.keras")

# 4. Save Model 3 (Char-level Conv1D)
model_3.save('saved_models/model_3.keras')
print("Saved Model 3 to saved_models/model_3.keras")

# 5. Save Model 4 (Token + Char Hybrid)
model_4.save('saved_models/model_4.keras')
print("Saved Model 4 to saved_models/model_4.keras")

# 6. Save Model 5 (Quadra Model - Multimodal)
model_5.save('saved_models/model_5.keras')
print("Saved Model 5 to saved_models/model_5.keras")

# Zip the saved_models directory to download
!zip -r saved_models.zip saved_models/

# Download to your local machine
files.download('saved_models.zip')


!pip install sentence-transformers


