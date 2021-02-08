# import required libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.applications import ResNet50,VGG16
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
import os
os.environ["TF_XLA_FLAGS"]="--tf_xla_enable_xla_devices"


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
 
   print(e)
   
   
   
# read the dataset 
df = pd.read_csv("data_mask.csv")

# drop the unwanted column
df = df.drop(columns=["patient_id","mask_path"], axis=1)

df['mask'] = df['mask'].apply(lambda x : str(x))

# split the dataset into train and test set
train_df, test_df = train_test_split(df,test_size=0.2,stratify=df['mask'])




# Augmentation
data_gen_with_aug = ImageDataGenerator(rescale=1./255.,
                              validation_split=0.15,
                              horizontal_flip=True,
                              vertical_flip=True,
                              fill_mode="nearest")


train_generator = data_gen_with_aug.flow_from_dataframe(
    dataframe=train_df,
    directory= './',
    x_col='image_path',
    y_col='mask',
    subset="training",
    batch_size=16,
    shuffle=True,
    class_mode="categorical",
    target_size=(224,224))


valid_generator=data_gen_with_aug.flow_from_dataframe(
    dataframe=train_df,
    directory= './',
    x_col='image_path',
    y_col='mask',
    subset="validation",
    batch_size=16,
    shuffle=True,
    class_mode="categorical",
    target_size=(224,224))


test_datagen_with_aug=ImageDataGenerator(rescale=1./255.)

test_generator_with_aug=test_datagen_with_aug.flow_from_dataframe(
    dataframe=test_df,
    directory= './',
    x_col='image_path',
    y_col='mask',
    batch_size=16,
    shuffle=False,
    class_mode='categorical',
    target_size=(224,224))



# VGG16 - Pretrained Model
basemodel_vgg = VGG16(include_top=False,input_shape=(224,224,3))

basemodel_vgg.trainable = False

vgg_model = basemodel_vgg.output
vgg_model = tf.keras.layers.Flatten(name="flatten")(vgg_model)
vgg_model = tf.keras.layers.Dense(256, activation = "relu")(vgg_model)
vgg_model = tf.keras.layers.Dropout(0.5)(vgg_model)
vgg_model = tf.keras.layers.Dense(256, activation = "relu")(vgg_model)
vgg_model = tf.keras.layers.Dropout(0.5)(vgg_model)
vgg_model = tf.keras.layers.Dense(2, activation = "softmax")(vgg_model)

vgg_model =tf.keras.models.Model(inputs = basemodel_vgg.input, outputs = vgg_model)

vgg_model.load_weights("classifier-vgg-weights.hdf5")

earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="classifier-vgg-weights.hdf5", verbose=1, save_best_only=True)


optim = tf.keras.optimizers.Adam(lr=1e-3)

vgg_model.compile(loss="categorical_crossentropy",optimizer=optim,metrics=["accuracy"])

vgg_history = vgg_model.fit_generator(train_generator,steps_per_epoch= train_generator.n // 16,
                              epochs = 100, validation_data= valid_generator,
                              validation_steps= valid_generator.n // 16,
                               callbacks=[earlystopping,checkpointer]
                              )


# save the model
vggmodel_json = vgg_model.to_json()
with open("classifier-vgg-model.json","w") as json_file:
  json_file.write(vggmodel_json)

# load the trained model
with open("classifier-vgg-model.json",'r') as json_file:
    json_savedModel = json_file.read()
    # load the model
    model = tf.keras.models.model_from_json(json_savedModel)
    model.load_weights('classifier-vgg-weights.hdf5')
    model.compile(loss = 'categorical_crossentropy', optimizer=optim, metrics= ["accuracy"])


# predict the test set
test_predict =  model.predict(test_generator_with_aug,steps = test_generator_with_aug.n // 16, verbose =1)

predict = []

for i in test_predict:
    predict.append(str(np.argmax(i)))

predict = np.asarray(predict)

original = np.asarray(test_df['mask'])[:len(predict)]

# accuracy score
accuracy = accuracy_score(original,predict)


# confusion matrix
cm = confusion_matrix(original,predict)  

plt.figure(figsize=(7,7))
    
sns.heatmap(cm,annot=True)

# classification report
report = classification_report(original,predict,labels=[0,1])

print(report)




