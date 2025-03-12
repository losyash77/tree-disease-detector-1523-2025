import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.applications import EfficientNetV2L
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, GlobalAveragePooling2D
from keras._tf_keras.keras.optimizers import Adam

# 1. Путь к папке с изображениями
dataset_dir = 'Result/split_1'

train_datagen = ImageDataGenerator(
    validation_split=0.15         
)

test_datagen = ImageDataGenerator()

batch_size = 12

train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),  
    batch_size=batch_size,
    class_mode='categorical', 
    subset='training'  
)

validation_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',  
    subset='validation'  
)


# 2. Инициализация предобученной модели
base_model = EfficientNetV2L(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# 3. Добавление новых слоев для классификации
model = Sequential()
model.add(base_model)  
model.add(GlobalAveragePooling2D())  
model.add(Dense(1024, activation='relu')) 
model.add(Dense(train_generator.num_classes, activation='softmax')) 

base_model.trainable = False

model.compile(optimizer=Adam(learning_rate=0.02), loss='categorical_crossentropy', metrics=['accuracy'])

    
# 4. Обучение модели
epochs = 10
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs
)


# 5. Оценка модели на тестовой выборке
test_loss, test_acc = model.evaluate(validation_generator, steps=validation_generator.samples // batch_size)
print(f'Точность на тестовой выборке: {test_acc * 100:.2f}%')

model.save('part_1_model.h5')