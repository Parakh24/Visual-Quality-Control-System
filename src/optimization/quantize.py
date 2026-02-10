"""
TensorFlow Lite Model Conversion with Quantization

Week 4 Task: Convert the trained model to TensorFlow Lite and apply quantization

This script converts a trained Keras model to TensorFlow Lite format with
different quantization options for edge deployment and real-time inference.

Author: Vision QC Team - Week 4
"""

import os
import sys
import tensorflow as tf
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import config


class ModelQuantizer:
    """Handles model conversion to TFLite with various quantization options"""
    
    def __init__(self, model_path, output_dir=None):
        """
        Initialize the quantizer
        
        Args:
            model_path: Path to trained Keras model (.h5 or .keras)
            output_dir: Directory to save TFLite models (default: models/optimized/)
        """
        self.model_path = model_path
        self.output_dir = output_dir or os.path.join(config.MODELS_DIR, "optimized")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load the trained model
        print(f"Loading model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded successfully")
        print(f"   Input shape: {self.model.input_shape}")
        print(f"   Output shape: {self.model.output_shape}\n")
    
    def convert_float32(self, output_name="model_float32.tflite"):
        """
        Convert to TFLite without quantization (float32)
        
        Args:
            output_name: Name of output TFLite file
            
        Returns:
            str: Path to saved TFLite model
        """
        print("="*60)
        print("Converting to Float32 TFLite (No Quantization)")
        print("="*60)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # No quantization - full precision
        tflite_model = converter.convert()
        
        # Save model
        output_path = os.path.join(self.output_dir, output_name)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ Float32 model saved: {output_path}")
        print(f"   Size: {size_mb:.2f} MB\n")
        
        return output_path
    
    def convert_float16(self, output_name="model_float16.tflite"):
        """
        Convert to TFLite with float16 quantization
        Reduces model size by ~50% with minimal accuracy loss
        
        Args:
            output_name: Name of output TFLite file
            
        Returns:
            str: Path to saved TFLite model
        """
        print("="*60)
        print("Converting to Float16 TFLite (Float16 Quantization)")
        print("="*60)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Apply float16 quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        # Save model
        output_path = os.path.join(self.output_dir, output_name)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ Float16 model saved: {output_path}")
        print(f"   Size: {size_mb:.2f} MB")
        print(f"   Compression: ~50% size reduction\n")
        
        return output_path
    
    def convert_dynamic_range(self, output_name="model_dynamic_quant.tflite"):
        """
        Convert to TFLite with dynamic range quantization
        Quantizes weights to int8, activations remain float
        Good balance between size and accuracy
        
        Args:
            output_name: Name of output TFLite file
            
        Returns:
            str: Path to saved TFLite model
        """
        print("="*60)
        print("Converting to Dynamic Range Quantized TFLite")
        print("="*60)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Apply dynamic range quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        # Save model
        output_path = os.path.join(self.output_dir, output_name)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ Dynamic range quantized model saved: {output_path}")
        print(f"   Size: {size_mb:.2f} MB")
        print(f"   Weights: int8, Activations: float32")
        print(f"   Compression: ~75% size reduction\n")
        
        return output_path
    
    def _representative_dataset_generator(self):
        """
        Generate representative dataset for full integer quantization
        Uses test data to calibrate quantization
        
        Yields:
            np.ndarray: Batch of sample images
        """
        # Load sample images from test set
        test_dir = config.TEST_DIR
        
        # Get all image files
        image_paths = []
        for class_name in os.listdir(test_dir):
            class_dir = os.path.join(test_dir, class_name)
            if os.path.isdir(class_dir):
                for img_file in os.listdir(class_dir)[:50]:  # Use 50 samples per class
                    if img_file.endswith(('.jpg', '.jpeg', '.png')):
                        image_paths.append(os.path.join(class_dir, img_file))
        
        print(f"   Using {len(image_paths)} representative samples for calibration")
        
        # Generate batches
        for img_path in image_paths:
            # Load and preprocess image
            img = tf.keras.preprocessing.image.load_img(
                img_path, 
                target_size=config.IMG_SIZE
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0  # Normalize to [0, 1]
            img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
            
            yield [img_array]
    
    def convert_full_integer(self, output_name="model_int8_quant.tflite"):
        """
        Convert to TFLite with full integer quantization
        Both weights and activations are int8
        Maximum compression and speed, some accuracy loss
        
        Args:
            output_name: Name of output TFLite file
            
        Returns:
            str: Path to saved TFLite model
        """
        print("="*60)
        print("Converting to Full Integer Quantized TFLite (INT8)")
        print("="*60)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Apply full integer quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = self._representative_dataset_generator
        
        # Ensure all ops are quantized to int8
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        try:
            tflite_model = converter.convert()
        except Exception as e:
            print(f"⚠️ Full int8 quantization failed: {e}")
            print("Falling back to hybrid quantization...")
            
            # Fallback to hybrid (int8 weights, float32 activations)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            converter.inference_input_type = tf.float32
            converter.inference_output_type = tf.float32
            tflite_model = converter.convert()
        
        # Save model
        output_path = os.path.join(self.output_dir, output_name)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ Integer quantized model saved: {output_path}")
        print(f"   Size: {size_mb:.2f} MB")
        print(f"   Weights: int8, Activations: int8 (or float32 if fallback)")
        print(f"   Compression: ~75-80% size reduction\n")
        
        return output_path
    
    def convert_all(self, base_name="vision_spec_qc"):
        """
        Convert model with all quantization options
        
        Args:
            base_name: Base name for output files
            
        Returns:
            dict: Paths to all generated models
        """
        print("\n" + "#"*60)
        print("# CONVERTING MODEL WITH ALL QUANTIZATION OPTIONS")
        print("#"*60 + "\n")
        
        models = {}
        
        # 1. Float32 (no quantization)
        models['float32'] = self.convert_float32(f"{base_name}_float32.tflite")
        
        # 2. Float16 (half precision)
        models['float16'] = self.convert_float16(f"{base_name}_float16.tflite")
        
        # 3. Dynamic range (int8 weights, float32 activations)
        models['dynamic'] = self.convert_dynamic_range(f"{base_name}_dynamic_quant.tflite")
        
        # 4. Full integer (int8 weights and activations)
        models['int8'] = self.convert_full_integer(f"{base_name}_int8_quant.tflite")
        
        # Summary
        print("="*60)
        print("CONVERSION SUMMARY")
        print("="*60)
        
        for quant_type, path in models.items():
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"✓ {quant_type.upper():12s}: {size_mb:6.2f} MB - {path}")
        
        print("\n" + "="*60)
        print("✅ All quantization options completed!")
        print("="*60 + "\n")
        
        return models
    
    def test_converted_model(self, tflite_path, num_test_samples=10):
        """
        Test the converted TFLite model on sample images
        
        Args:
            tflite_path: Path to TFLite model
            num_test_samples: Number of samples to test
            
        Returns:
            dict: Test results
        """
        print(f"\nTesting TFLite model: {os.path.basename(tflite_path)}")
        print("-"*60)
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Input type: {input_details[0]['dtype']}")
        print(f"Output shape: {output_details[0]['shape']}")
        print(f"Output type: {output_details[0]['dtype']}")
        
        # Test on sample images
        test_dir = config.TEST_DIR
        sample_count = 0
        
        for class_name in os.listdir(test_dir):
            class_dir = os.path.join(test_dir, class_name)
            if os.path.isdir(class_dir) and sample_count < num_test_samples:
                for img_file in os.listdir(class_dir):
                    if img_file.endswith(('.jpg', '.jpeg', '.png')) and sample_count < num_test_samples:
                        img_path = os.path.join(class_dir, img_file)
                        
                        # Load and preprocess
                        img = tf.keras.preprocessing.image.load_img(img_path, target_size=config.IMG_SIZE)
                        img_array = tf.keras.preprocessing.image.img_to_array(img)
                        img_array = img_array / 255.0
                        img_array = np.expand_dims(img_array, axis=0).astype(input_details[0]['dtype'])
                        
                        # Run inference
                        interpreter.set_tensor(input_details[0]['index'], img_array)
                        interpreter.invoke()
                        output = interpreter.get_tensor(output_details[0]['index'])
                        
                        sample_count += 1
        
        print(f"✅ Tested successfully on {sample_count} samples\n")
        
        return {"tested_samples": sample_count, "status": "success"}


def main():
    """Main execution function"""
    
    # Configuration
    model_path = os.path.join(config.TRAINED_MODELS_DIR, "vision_spec_qc.keras")
    
    # Check if model exists
    if not os.path.exists(model_path):
        # Try .h5 extension
        model_path = os.path.join(config.TRAINED_MODELS_DIR, "vision_spec_qc.h5")
        if not os.path.exists(model_path):
            # Try to find any model in the directory
            trained_dir = config.TRAINED_MODELS_DIR
            if os.path.exists(trained_dir):
                models = [f for f in os.listdir(trained_dir) if f.endswith(('.h5', '.keras'))]
                if models:
                    model_path = os.path.join(trained_dir, models[0])
                    print(f"Using found model: {model_path}")
                else:
                    print(f"❌ No trained model found in {trained_dir}")
                    print("Please ensure your trained model is in the models/trained/ directory.")
                    return
            else:
                print(f"❌ Model directory not found: {trained_dir}")
                return
    
    # Create quantizer
    quantizer = ModelQuantizer(model_path)
    
    # Convert with all quantization options
    converted_models = quantizer.convert_all(base_name="vision_spec_qc")
    
    # Test the dynamic quantized model (recommended for deployment)
    quantizer.test_converted_model(converted_models['dynamic'], num_test_samples=10)
    
    print("\n" + "#"*60)
    print("# QUANTIZATION COMPLETE!")
    print("#"*60)
    print("\n📋 Recommendation:")
    print("   - Use 'dynamic_quant' for best balance of size/speed/accuracy")
    print("   - Use 'int8_quant' for maximum speed on edge devices")
    print("   - Use 'float16' if you need near-original accuracy")
    print("\n🚀 Next Step: Benchmark the models using benchmark.py\n")


if __name__ == "__main__":
    main()
