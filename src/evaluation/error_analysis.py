import tensorflow as tf
import os

def main():
    # Correct model path
    model_path = r"D:\Visual_quality_system\models\trained\vision_spec_qc.keras"
    
    # Check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Continue with your analysis...
    print("Model loaded successfully!")

if __name__ == "__main__":
    main()
