#!/usr/bin/env python3
"""
Simple inference test for DeepSeek fine-tuned models
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import gc
import os

def test_detection():
    """Test detection model"""
    print("Testing Detection Model...")
    
    base_model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
    detection_path = "./models/detection_qlora"
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    model = PeftModel.from_pretrained(base_model, detection_path)
    model = model.merge_and_unload()
    model.eval()
    
    # Test code
    test_code = """
void GameUpdate() {
    GameObject* obj = new GameObject();
    obj->Update();
    delete obj;
}
"""
    
    # Create prompt
    system_prompt = "You are a C++ expert. Detect code smells and respond with JSON format."
    instruction = f"Analyze this C++ code for problems: {test_code}"
    prompt = f"[INST] {system_prompt}\n\n{instruction} [/INST]"
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.5,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\nTest Code:")
    print(test_code)
    print("\nDetection Response:")
    if "[/INST]" in response:
        result = response.split("[/INST]")[-1].strip()
        print(result)
    else:
        print(response)
    
    # Cleanup
    del model, base_model
    torch.cuda.empty_cache()
    gc.collect()

def test_fixing():
    """Test fixing model"""
    print("\n" + "="*50)
    print("Testing Fixing Model...")
    
    base_model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
    fixing_path = "./models/fixing_qlora"
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    model = PeftModel.from_pretrained(base_model, fixing_path)
    model = model.merge_and_unload()
    model.eval()
    
    # Test code
    test_code = """
void GameUpdate() {
    GameObject* obj = new GameObject();
    obj->Update();
    delete obj;
}
"""
    
    # Create prompt
    system_prompt = "You are a C++ expert. Fix code problems and respond with improved code."
    instruction = f"Improve this C++ code: {test_code}"
    prompt = f"[INST] {system_prompt}\n\n{instruction} [/INST]"
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.5,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\nTest Code:")
    print(test_code)
    print("\nFixing Response:")
    if "[/INST]" in response:
        result = response.split("[/INST]")[-1].strip()
        print(result)
    else:
        print(response)
    
    # Cleanup
    del model, base_model
    torch.cuda.empty_cache()
    gc.collect()

def main():
    """Main function"""
    print("=" * 50)
    print("DeepSeek Fine-tuned Model Test")
    print("=" * 50)
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Test detection
    test_detection()
    
    # Test fixing
    test_fixing()
    
    print("\n" + "=" * 50)
    print("Test completed!")
    print("=" * 50)

if __name__ == "__main__":
    main()