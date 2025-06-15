from tensorflow.keras.models import load_model

# Old model path
old_model_path = "keras_cifar10_trained_model.h5"

# New model path (overwrite or change name)
new_model_path = "keras_cifar10_resaved_tf212.h5"

# Load with old config
model = load_model(old_model_path, compile=False)

# Re-save the model with new format
model.save(new_model_path)

print("Model re-saved successfully with TensorFlow 2.12.0 compatibility.")
