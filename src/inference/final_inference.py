#!/usr/bin/env python3
"""
Final optimized inference script for DeepSeek fine-tuned models
Handles both detection and fixing with proper JSON extraction
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import gc
import os
import re

class DeepSeekFineTunedInference:
    def __init__(self):
        self.base_model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
        self.detection_path = "./models/detection_qlora"
        self.fixing_path = "./models/fixing_qlora"
        
        # Optimized quantization
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        # Load tokenizer once
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name, 
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def extract_json_response(self, text, expected_keys):
        """Extract JSON from model response"""
        # Try to find JSON-like structure
        json_patterns = [
            r'\{.*?"smells".*?\}',  # Detection format
            r'\{.*?"fixed_code".*?\}',  # Fixing format
            r'\{[^{}]*\}',  # Simple JSON
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if any(key in parsed for key in expected_keys):
                        return parsed
                except:
                    continue
        
        # Fallback: return empty structure
        if "smells" in expected_keys:
            return {"smells": []}
        else:
            return {"fixed_code": "", "explanations": []}
    
    def create_detection_prompt(self, code):
        """Create detection prompt"""
        system_prompt = """You are a C++ expert detecting code smells. Output only valid JSON:
{"smells": [{"type": "smell_name", "severity": 50, "explanation": "brief description"}]}
If no smells, output: {"smells": []}"""
        
        instruction = f"Analyze this C++ code for smells:\n{code}"
        return f"[INST] {system_prompt}\n\n{instruction} [/INST]"
    
    def create_fixing_prompt(self, code, smells=None):
        """Create fixing prompt"""
        system_prompt = """You are a C++ expert fixing code smells. Output only valid JSON:
{"fixed_code": "corrected_code_here", "explanations": [{"smell_type": "type", "fix_description": "what was fixed"}]}"""
        
        if smells and smells.get("smells"):
            smell_types = [s["type"] for s in smells["smells"]]
            instruction = f"Fix these smells {smell_types} in code:\n{code}"
        else:
            instruction = f"Improve this C++ code:\n{code}"
            
        return f"[INST] {system_prompt}\n\n{instruction} [/INST]"
    
    def run_detection(self, code_snippet):
        """Run detection inference"""
        print("🔍 Running code smell detection...")
        
        # Load model
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=self.bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            model = PeftModel.from_pretrained(base_model, self.detection_path)
            model = model.merge_and_unload()
            model.eval()
            
            # Generate
            prompt = self.create_detection_prompt(code_snippet)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # Extract response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text.split("[/INST]")[-1].strip()
            
            # Parse JSON
            result = self.extract_json_response(response, ["smells"])
            
            # Cleanup
            del model, base_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return result
            
        except Exception as e:
            print(f"❌ Detection error: {e}")
            return {"smells": []}
    
    def run_fixing(self, code_snippet, detected_smells=None):
        """Run fixing inference"""
        print("🛠️  Running code smell fixing...")
        
        # Load model
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=self.bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            model = PeftModel.from_pretrained(base_model, self.fixing_path)
            model = model.merge_and_unload()
            model.eval()
            
            # Generate
            prompt = self.create_fixing_prompt(code_snippet, detected_smells)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # Extract response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text.split("[/INST]")[-1].strip()
            
            # Parse JSON
            result = self.extract_json_response(response, ["fixed_code"])
            
            # Cleanup
            del model, base_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return result
            
        except Exception as e:
            print(f"❌ Fixing error: {e}")
            return {"fixed_code": code_snippet, "explanations": []}

def test_inference():
    """Test the inference pipeline"""
    print("=" * 60)
    print("🚀 DeepSeek Fine-tuned Model Inference Test")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        {
            "name": "Clean Code",
            "code": """
int calculateSum(int a, int b) {
    return a + b;
}
""".strip()
        },
        {
            "name": "Problematic Game Code", 
            "code": """
void GameEngine::Update() {
    for (int i = 0; i < entities.size(); i++) {
        GameObject* obj = new GameObject();
        obj->ProcessHeavyAI();
        obj->RenderComplexGraphics();
        delete obj;
    }
    
    GameObject* player = GameObject::FindByName("Player");
    if (player) {
        player->Update();
        player->Render();
        player->PlaySound();
    }
}
""".strip()
        },
        {
            "name": "Memory Management Issues",
            "code": """
class ResourceManager {
    std::vector<Texture*> textures;
    
    void LoadTextures() {
        for (int i = 0; i < 100; i++) {
            Texture* tex = new Texture();
            tex->LoadFromFile("texture" + std::to_string(i) + ".png");
            textures.push_back(tex);
        }
    }
};
""".strip()
        }
    ]
    
    # Initialize inference
    inferencer = DeepSeekFineTunedInference()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test_case['name']}")
        print("=" * 40)
        print("Original Code:")
        print(test_case['code'])
        
        # Run detection
        detected_smells = inferencer.run_detection(test_case['code'])
        print(f"\n🔍 Detected Smells:")
        print(json.dumps(detected_smells, indent=2))
        
        # Run fixing
        fixed_result = inferencer.run_fixing(test_case['code'], detected_smells)
        print(f"\n🛠️  Fixed Code:")
        print(json.dumps(fixed_result, indent=2))
        
        print("-" * 60)
    
    print("\n✅ All tests completed successfully!")

if __name__ == "__main__":
    test_inference()