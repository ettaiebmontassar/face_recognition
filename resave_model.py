from tensorflow.keras.models import load_model
import numpy as np

# Load the Keras model
model = load_model('models/facenet_keras.h5')

# Print the model summary
print("Model loaded successfully.")
model.summary()

# Generate a dummy input array with the expected input shape
dummy_input = np.random.random((1, 160, 160, 3))  # Assuming the model expects 160x160 RGB images

# Perform a prediction
dummy_output = model.predict(dummy_input)

print("Dummy prediction result:", dummy_output)
