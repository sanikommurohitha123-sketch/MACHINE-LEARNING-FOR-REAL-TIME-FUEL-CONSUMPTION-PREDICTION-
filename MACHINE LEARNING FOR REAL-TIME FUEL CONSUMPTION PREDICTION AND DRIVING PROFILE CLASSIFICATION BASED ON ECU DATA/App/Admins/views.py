from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import messages
from User.models import PredictionLog
from django.core.paginator import Paginator
from django.http import HttpResponse
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint


def adminhome(request):
    users = User.objects.filter(is_staff=False, is_superuser=False) 
    return render(request, "Admin/adminhome.html", {"users": users})

def admin_update_userstatus(request, user_id):
    try:
        user = User.objects.get(id=user_id)
        
        # Toggle the is_active status
        user.is_active = not user.is_active
        user.save()

        # Display message based on the action
        if user.is_active:
            messages.success(request, f"User {user.username} has been activated.")
        else:
            messages.success(request, f"User {user.username} has been deactivated.")
        
        return redirect('adminhome')  # Redirect back to the admin home page
    except User.DoesNotExist:
        messages.error(request, "User not found.")
        return redirect('adminhome')
    
def adminaccuracy(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['dataset']
        dataset_path = f'media/{uploaded_file.name}'
        
        # Save the uploaded file
        with open(dataset_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        # Load dataset
        try:
            data = pd.read_csv(dataset_path, encoding='latin1')

            # Define features and target
            features = ['Year', 'Make', 'Model', 'Vehicle Class', 'Engine Size (L)', 'Cylinders',
                        'Transmission', 'Fuel Type', 'Hwy (L/100 km)', 'Comb (L/100 km)']
            target = 'Fuel Consumption (L/100Km)'

            # Encode categorical features
            categorical_features = ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']
            label_encoders = {}
            for col in categorical_features:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                label_encoders[col] = le

            # Split data
            X = data[features]
            y = data[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train models
            models = {
                'Ridge Regression': Ridge(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Support Vector Regression': SVR()
            }
            results = []
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                predictions = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                results.append({'Model': name, 'MSE': round(mse, 4), 'R2 Score': round(r2, 4)})

            # LSTM Model Training
            X_train_lstm = np.expand_dims(X_train_scaled, axis=2)  # Convert to 3D
            X_test_lstm = np.expand_dims(X_test_scaled, axis=2)

            lstm_model = Sequential([
                LSTM(50, activation='relu', input_shape=(X_train_lstm.shape[1], 1)),
                Dense(1)
            ])
            lstm_model.compile(optimizer='adam', loss='mse')

            checkpoint = ModelCheckpoint('media/best_lstm_model.h5', save_best_only=True, monitor='val_loss', mode='min')
            lstm_model.fit(X_train_lstm, y_train, validation_data=(X_test_lstm, y_test), epochs=50, callbacks=[checkpoint], verbose=0)

            # Load the best LSTM model and evaluate
            lstm_model.load_weights('media/best_lstm_model.h5')
            lstm_preds = lstm_model.predict(X_test_lstm).flatten()
            lstm_mse = mean_squared_error(y_test, lstm_preds)
            lstm_r2 = r2_score(y_test, lstm_preds)
            results.append({'Model': 'LSTM', 'MSE': round(lstm_mse, 4), 'R2 Score': round(lstm_r2, 4)})

            # Remove the uploaded file after processing
            os.remove(dataset_path)

            return render(request, 'Admin/adminaccuracy.html', {'results': results})

        except Exception as e:
            return HttpResponse(f"Error processing the dataset: {str(e)}")
    
    return render(request, 'Admin/adminaccuracy.html')

def adminuserpredictions(request):
    predictions = PredictionLog.objects.all().order_by('-created_at')
    paginator = Paginator(predictions, 10) 
    page_number = request.GET.get('page')  
    page_obj = paginator.get_page(page_number) 
    return render(request, 'Admin/adminuserpredictions.html', {'page_obj': page_obj})