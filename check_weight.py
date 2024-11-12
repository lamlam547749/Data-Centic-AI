import tensorflow as tf
from tensorflow import keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import sys
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
directory = "Data-centric/"
user_data = directory + "clean/Combined Handwritten Roman Numerals Dataset"
test_data = directory + "label_book"                # this can be the label book, or any other test set you create

### DO NOT MODIFY BELOW THIS LINE, THIS IS THE FIXED MODEL ###
batch_size = 8
tf.random.set_seed(123)


if __name__ == "__main__":

    # train = tf.keras.preprocessing.image_dataset_from_directory(
    #     user_data + '/new82_train',
    #     labels="inferred",
    #     label_mode="categorical",
    #     class_names=["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"],
    #     shuffle=True,
    #     seed=123,
    #     batch_size=batch_size,
    #     image_size=(32, 32),
    # )    
    # valid = tf.keras.preprocessing.image_dataset_from_directory(
    #     user_data + '/new82_val',
    #     labels="inferred",  
    #     label_mode="categorical",
    #     class_names=["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"],
    #     shuffle=True,
    #     seed=123,
    #     batch_size=batch_size,
    #     image_size=(32, 32),
    # )
    test = tf.keras.preprocessing.image_dataset_from_directory(
        test_data,
        labels="inferred",
        label_mode="categorical",
        class_names=["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"],
        shuffle=False,
        seed=123,
        batch_size=batch_size,
        image_size=(32, 32),
    )
    # Define the model architecture (must match the architecture when saving weights)
    base_model = tf.keras.applications.ResNet50(
        input_shape=(32, 32, 3),
        include_top=False,
        weights=None,
    )
    base_model = tf.keras.Model(
        base_model.inputs, outputs=[base_model.get_layer("conv2_block3_out").output]
    )

    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.applications.resnet.preprocess_input(inputs)
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(10)(x)
    model = tf.keras.Model(inputs, x)

    # Compile the model (this step is required before loading weights)
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(lr=0.0001),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # Load the weights from the checkpoint
    model.load_weights("best_model")                    ######################### THAY ĐƯỜNG DẪN CHECK WEIGHT
    # train_loss, train_acc = model.evaluate(train)
    # print(f"Loaded model - train_loss: {train_loss}, train_acc: {train_acc}")
    # val_loss, val_acc = model.evaluate(valid)
    # print(f"Loaded model - val_loss: {val_loss}, val_acc: {val_acc}")
    # # Now the model is ready to use, you can evaluate or predict using this model
    loss, acc = model.evaluate(test)  # Assuming `valid` is your validation dataset
    print(f"Loaded model - loss: {loss}, accuracy: {acc}")


    # Đường dẫn đến ảnh bạn muốn dự đoán
    img_path = 'Data-centric/label_book/x/label_book_x_4.png'  # Thay đổi đường dẫn ảnh tại đây

    # Lấy nhãn thực từ cấu trúc thư mục
    true_label = img_path.split('/')[-2]  # Lấy tên thư mục cha của ảnh làm nhãn thực

    # Load và tiền xử lý ảnh
    img = image.load_img(img_path, target_size=(32, 32))  # Điều chỉnh kích thước theo yêu cầu mô hình
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet.preprocess_input(img_array)

    # Dự đoán nhãn của ảnh
    predictions = model.predict(img_array)
    predicted_label = np.argmax(predictions, axis=1)[0]
    class_names = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]  # Danh sách nhãn đã sử dụng
    predicted_label_name = class_names[predicted_label]

    # Hiển thị ảnh và nhãn
    plt.imshow(image.load_img(img_path))  # Hiển thị ảnh gốc
    plt.title(f"True Label: {true_label}, Predicted Label: {predicted_label_name}")
    plt.axis('off')
    plt.show()