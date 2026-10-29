from django.shortcuts import render
import joblib
from .models import PredictionLog

# Load models
loaded_ridge = joblib.load('model/ridge_regressor_model.pkl')
loaded_scaler = joblib.load('model/scaler.pkl')

# Create your views here.
def userhome(request):
    user = request.user
    return render(request, 'User/userhome.html', {'user':user})

def userpredict(request):
    if request.method == 'POST':
        try:
            # Get inputs from the form
            year = int(request.POST.get('year'))
            make = int(request.POST.get('make'))
            model = int(request.POST.get('model'))
            vehicle_class = int(request.POST.get('vehicle_class'))
            engine_size = float(request.POST.get('engine_size'))
            cylinders = int(request.POST.get('cylinders'))
            transmission = int(request.POST.get('transmission'))
            fuel_type = int(request.POST.get('fuel_type'))
            hwy = float(request.POST.get('hwy'))
            comb = float(request.POST.get('comb'))

            # Prepare input for the model
            future_input = [[year, make, model, vehicle_class, engine_size, cylinders, transmission, fuel_type, hwy, comb]]
            future_input_scaled = loaded_scaler.transform(future_input)
            future_prediction = loaded_ridge.predict(future_input_scaled)
            
            # Round prediction to two decimal places
            predicted_value = round(future_prediction[0], 2)

            # Save data to the database
            PredictionLog.objects.create(
                year=year,
                make=make,
                model=model,
                vehicle_class=vehicle_class,
                engine_size=engine_size,
                cylinders=cylinders,
                transmission=transmission,
                fuel_type=fuel_type,
                hwy=hwy,
                comb=comb,
                predicted_value=predicted_value
            )

            return render(request, 'User/userpredict.html', {'predicted_value': predicted_value})
        
        except Exception as e:
            return render(request, 'User/userpredict.html', {'error': str(e)})

    return render(request, 'User/userpredict.html')