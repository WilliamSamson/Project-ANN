from tensorflow.keras.models import load_model

# Load the model
model = load_model('best_model.h5')

# View the model summary
model.summary()
