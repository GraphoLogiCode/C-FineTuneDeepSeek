#!/usr/bin/env python3
"""
Quick inference test for fine-tuned DeepSeek Coder models
Simplified version with better memory management
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import gc
import os

def test_detection_model():
    """Quick test of detection model"""
    print("Testing Detection Model")
    print("-" * 30)
    
    # Model paths
    base_model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
    detection_path = "./models/detection_qlora"
    
    # Check if model exists
    if not os.path.exists(detection_path):
        print(f"Error: Model path {detection_path} not found!")
        return
    
    # Simple quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load and merge model
    try:
        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        print("Loading PEFT adapters...")
        model = PeftModel.from_pretrained(base_model, detection_path)
        model = model.merge_and_unload()
        model.eval()
        
        print("Model loaded successfully!")
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB" if torch.cuda.is_available() else "CPU mode")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Test code snippet
    test_code = """
void Update() {
    GameObject* obj = new GameObject();
    obj->DoSomething();
    delete obj;
}
"""
    
    # Simple system prompt
    system_prompt = """You are a C++ expert. Detect code smells and output JSON: {"smells": [{"type": "smell_name", "severity": 50, "explanation": "description"}]}"""
    
    # Create prompt
    instruction = f"Detect code smells in this C++ code: {test_code}"
    prompt = f"[INST] {system_prompt}\n\n{instruction} [/INST]"
    
    # Generate
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        print("\nGenerating response...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response after [/INST]
        if "[/INST]" in generated_text:
            response = generated_text.split("[/INST]", 1)[1].strip()
        else:
            response = generated_text
            
        print("\nInput code:")
        print(test_code)
        print("\nModel response:")
        print(response)
        
        # Try to parse as JSON
        try:
            if "{" in response:
                json_start = response.find("{")
                json_part = response[json_start:]
                # Find end of JSON
                brace_count = 0
                json_end = 0
                for i, char in enumerate(json_part):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if json_end > 0:
                    json_output = json_part[:json_end]
                    parsed = json.loads(json_output)
                    print("\nParsed JSON:")
                    print(json.dumps(parsed, indent=2))
        except:
            print("\nCould not parse as valid JSON, but model generated a response!")
            
    except Exception as e:
        print(f"Error during generation: {e}")
        return
    
    # Cleanup
    del model, base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("\nDetection test completed!")

def test_fixing_model():
    """Quick test of fixing model"""
    print("\nTesting Fixing Model")
    print("-" * 30)
    
    # Model paths
    base_model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
    fixing_path = "./models/fixing_qlora"
    
    # Check if model exists
    if not os.path.exists(fixing_path):
        print(f"Error: Model path {fixing_path} not found!")
        return
    
    # Simple quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load and merge model
    try:
        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        print("Loading PEFT adapters...")
        model = PeftModel.from_pretrained(base_model, fixing_path)
        model = model.merge_and_unload()
        model.eval()
        
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Test code snippet
    test_code = """
void Update() {
    GameObject* obj = new GameObject();
    obj->DoSomething();
    delete obj;
}
"""
    
    # Simple system prompt
    system_prompt = """You are a C++ expert. Fix code smells and output JSON: {"fixed_code": "corrected_code", "explanations": [{"smell_type": "type", "fix_description": "description"}]}"""
    
    # Create prompt
    instruction = f"Fix this C++ code: {test_code}"
    prompt = f"[INST] {system_prompt}\n\n{instruction} [/INST]"
    
    # Generate
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        print("\nGenerating response...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response after [/INST]
        if "[/INST]" in generated_text:
            response = generated_text.split("[/INST]", 1)[1].strip()
        else:
            response = generated_text
            
        print("\nInput code:")
        print(test_code)
        print("\nModel response:")
        print(response)
        
    except Exception as e:
        print(f"Error during generation: {e}")
        return
    
    # Cleanup
    del model, base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("\nFixing test completed!")

def main():
    """Main function"""
    print("=" * 50)
    print("DeepSeek Fine-tuned Model Quick Test")
    print("=" * 50)
    
    # Clear any existing GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Test detection model
    test_detection_model()
    
    # Small delay and cleanup between models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Test fixing model
    test_fixing_model()
    
    print("\n" + "=" * 50)
    print("All tests completed!")

if __name__ == "__main__":
    main()