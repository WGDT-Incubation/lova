import os

# Save trained model
directory = os.path.join(os.getcwd(), 'saved_models')
if not os.path.isdir(directory):
    os.makedirs(directory)

model_path = os.path.join(directory, 'keras_cifar10_trained_model.h5')
cnn_model.save(model_path)
print(f"✅ Model saved to: {model_path}")
