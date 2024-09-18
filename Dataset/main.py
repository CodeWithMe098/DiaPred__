import pandas as pd
import numpy as np
import joblib
from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn import preprocessing

def train_model():
    # Load the dataset
    dataset = pd.read_csv('C:\Users\Liaqat\Documents\FYP\diabetes\Dataset\diabetes.csv')
    label_encoder = preprocessing.LabelEncoder()

    # Encode categorical columns
    for column in dataset.columns[1:]:
        dataset[column] = label_encoder.fit_transform(dataset[column])

    # Split data into features and target
    X = dataset.drop('class', axis=1)  # Features
    y = dataset['class']  # Target variable

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Standardize features using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build the ANN model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=6, activation='relu', input_shape=(16,)),
        tf.keras.layers.Dense(units=6, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32)

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    # Save the model
    model.save('C:\Users\Liaqat\Documents\FYP\diabetes\Dataset\diabetes.h5')

def predict_diabetes(request):
    if request.method == 'POST':
        # Get the form data
        Age = request.POST.get('Age')
        Gender = request.POST.get('Gender')
        Polyuria = request.POST.get('Polyuria')
        Polydipsia = request.POST.get('Polydipsia')
        sudden_weight_loss = request.POST.get('sudden_weight_loss')
        weakness = request.POST.get('weakness')
        Polyphagia = request.POST.get('Polyphagia')
        Genital_thrush = request.POST.get('Genital_thrush')
        visual_blurring = request.POST.get('visual_blurring')
        Itching = request.POST.get('Itching')
        Irritability = request.POST.get('Irritability')
        delayed_healing = request.POST.get('delayed_healing')
        partial_paresis = request.POST.get('partial_paresis')
        Alopecia = request.POST.get('Alopecia')
        muscle_stiffness = request.POST.get('muscle_stiffness')
        Obesity = request.POST.get('Obesity')

        # Validate form fields
        if (
            Age is None or Gender is None or Polyuria is None or Polydipsia is None or
            sudden_weight_loss is None or weakness is None or Polyphagia is None or
            Genital_thrush is None or visual_blurring is None or Itching is None or
            Irritability is None or delayed_healing is None or partial_paresis is None or
            Alopecia is None or muscle_stiffness is None or Obesity is None
        ):
            error_message = "Please fill in all the required fields."
            return render(request, 'base.html', {'error_message': error_message})

        # Convert form fields to integers
        try:
            Age = int(Age)
            Gender = int(Gender)
            Polyuria = int(Polyuria)
            Polydipsia = int(Polydipsia)
            sudden_weight_loss = int(sudden_weight_loss)
            weakness = int(weakness)
            Polyphagia = int(Polyphagia)
            Genital_thrush = int(Genital_thrush)
            visual_blurring = int(visual_blurring)
            Itching = int(Itching)
            Irritability = int(Irritability)
            delayed_healing = int(delayed_healing)
            partial_paresis = int(partial_paresis)
            Alopecia = int(Alopecia)
            muscle_stiffness = int(muscle_stiffness)
            Obesity = int(Obesity)
        except ValueError:
            error_message = "Invalid input. Please enter valid integer values."
            return render(request, 'base.html', {'error_message': error_message})

        input_data = [
            Age, Gender, Polyuria, Polydipsia, sudden_weight_loss, weakness, Polyphagia,
            Genital_thrush, visual_blurring, Itching, Irritability, delayed_healing,
            partial_paresis, muscle_stiffness, Alopecia, Obesity
        ]

        # Load the trained model
        model = tf.keras.models.load_model('C:\Users\Liaqat\Documents\FYP\diabetes\Dataset\diabetes.h5')

        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        prediction = model.predict(input_data_reshaped)

        result = ""
        if prediction[0][0] < 0.5:
            result = "This person is non-diabetic."
        else:
            result = "This person is diabetic."

        return render(request, 'base.html', {'result': result})

    return render(request, 'base.html')
