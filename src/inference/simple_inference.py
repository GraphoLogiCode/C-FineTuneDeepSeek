#!/usr/bin/env python3
"""
Simple inference script for fine-tuned DeepSeek Coder 1.3B models
Tests both detection and fixing capabilities with code smell analysis
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import gc

# System prompts from the training
DETECTION_SYSTEM_PROMPT = """You are a C++ code expert specializing in detecting code smells in game engines. Analyze the provided C++ code snippet for code smells, prioritizing game-specific ones below.
- Game-Specific Smells (prioritize these)= 
  [ 1. Creating components/objects at run-time (design/logic, severity context: 83%)
  2. Weak temporization strategy (design/logic, severity context: 80%)
  3. Lack of separation of concerns (design/logic/maintainability, severity context: 77%)
  4. Bloated asset (design/logic, severity context: 63%)
  5. Poor design of object state management (design/logic, severity context: 63%)
  6. Search by string/ID (design/logic, severity context: 53%)
  7. Dependencies between objects (design/logic, severity context: 48%)
  8. Prefer static classes instead of singleton (design/logic, severity context: 33%)
  9. Static coupling (design/logic, severity context: 32%)
  10. Allocating and destroying GameObjects in updates (performance)
  11. Getting GameObject by name (performance)
  12. Heavyweight Update methods (performance)
  13. Coupling objects through the Inspector (maintainability; adapt to C++ equivalents like direct dependencies)]
- Output ALWAYS in this exact JSON format (no extra text, valid JSON only):
{
  "smells":
  [  // List of smell objects; empty [] if none
    {
      "type": "smell_type1",  // e.g., "Heavyweight Update methods"; could be more
      "severity": int,  // Overall 0-100 (e.g., average or max per smell; use provided % where applicable)
      "explanation": "brief overall description"  // e.g., "No smells detected" if clean
    }
  ]
}"""

FIXING_SYSTEM_PROMPT = """You are a C++ code expert specializing in fixing code smells in game engines. Based on detected smells, refactor the provided C++ code snippet, prioritizing game-specific fixes.
- Game-Specific Smells (prioritize these)= 
  [ 1. Creating components/objects at run-time (design/logic, severity context: 83%)
  2. Weak temporization strategy (design/logic, severity context: 80%)
  3. Lack of separation of concerns (design/logic/maintainability, severity context: 77%)
  4. Bloated asset (design/logic, severity context: 63%)
  5. Poor design of object state management (design/logic, severity context: 63%)
  6. Search by string/ID (design/logic, severity context: 53%)
  7. Dependencies between objects (design/logic, severity context: 48%)
  8. Prefer static classes instead of singleton (design/logic, severity context: 33%)
  9. Static coupling (design/logic, severity context: 32%)
  10. Allocating and destroying GameObjects in updates (performance)
  11. Getting GameObject by name (performance)
  12. Heavyweight Update methods (performance)
  13. Coupling objects through the Inspector (maintainability; adapt to C++ equivalents like direct dependencies)]
- Output ALWAYS in this exact JSON format (no extra text, valid JSON only):
{
  "fixed_code": "corrected_code_snippet",  // The refactored/filled C++ code
  "explanations": [  // List of fix explanations; empty [] if none
    {
      "smell_type": "smell_type1",  // e.g., "Heavyweight Update methods"
      "fix_description": "brief description of fix"  // e.g., "Optimized update loop"
    },
  ]
}"""

class DeepSeekInference:
    def __init__(self):
        self.base_model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
        self.detection_model_path = "./models/detection_qlora"  # QLoRA adapter path
        self.fixing_model_path = "./models/fixing_qlora"       # QLoRA adapter path
        
        # Setup quantization for efficient inference
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("DeepSeek Inference initialized")
        
    def create_prompt(self, system_prompt, instruction):
        """Create formatted prompt for DeepSeek model"""
        bos_token = self.tokenizer.bos_token or ""
        return f"{bos_token}[INST] {system_prompt}\n\n{instruction} [/INST]"
    
    def load_detection_model(self):
        """Load detection model with QLoRA adapters"""
        print("Loading detection model...")
        
        # Load base model with quantization
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load and merge PEFT adapters
        model = PeftModel.from_pretrained(base_model, self.detection_model_path)
        model = model.merge_and_unload()
        model.eval()
        
        print(f"Detection model loaded. GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        return model
    
    def load_fixing_model(self):
        """Load fixing model with QLoRA adapters"""
        print("Loading fixing model...")
        
        # Clear GPU cache first
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Load base model with quantization
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load and merge PEFT adapters
        model = PeftModel.from_pretrained(base_model, self.fixing_model_path)
        model = model.merge_and_unload()
        model.eval()
        
        print(f"Fixing model loaded. GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        return model
    
    def detect_smells(self, model, code_snippet):
        """Run detection inference on code snippet"""
        instruction = f"Detect code smells in this C++ snippet: {code_snippet}"
        prompt = self.create_prompt(DETECTION_SYSTEM_PROMPT, instruction)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and extract JSON
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            json_start = generated_text.find('{"smells"')
            if json_start != -1:
                json_text = generated_text[json_start:]
                # Find the end of JSON
                brace_count = 0
                json_end = json_start
                for i, char in enumerate(json_text):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = json_start + i + 1
                            break
                
                json_output = generated_text[json_start:json_end]
                return json.loads(json_output)
        except:
            pass
        
        return {"smells": []}
    
    def fix_smells(self, model, code_snippet, detected_smells):
        """Run fixing inference on code snippet"""
        smell_types = [smell["type"] for smell in detected_smells.get("smells", [])]
        
        if smell_types:
            instruction = f"Fill and fix this code with smells {smell_types}: {code_snippet}"
        else:
            instruction = f"Fill and fix this code: {code_snippet}"
            
        prompt = self.create_prompt(FIXING_SYSTEM_PROMPT, instruction)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and extract JSON
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            json_start = generated_text.find('{"fixed_code"')
            if json_start != -1:
                json_text = generated_text[json_start:]
                # Find the end of JSON
                brace_count = 0
                json_end = json_start
                for i, char in enumerate(json_text):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = json_start + i + 1
                            break
                
                json_output = generated_text[json_start:json_end]
                return json.loads(json_output)
        except:
            pass
        
        return {"fixed_code": code_snippet, "explanations": []}

def main():
    """Main inference demonstration"""
    print("=" * 50)
    print("DeepSeek Coder 1.3B Fine-tuned Inference Demo")
    print("=" * 50)
    
    # Test code snippets
    test_snippets = [
        # Test 1: Clean code
        """
int main() {
    std::cout << "Hello World" << std::endl;
    return 0;
}
        """.strip(),
        
        # Test 2: Code with potential smells
        """
void GameEngine::Update() {
    for (int i = 0; i < 1000; i++) {
        GameObject* obj = new GameObject();
        obj->DoHeavyCalculation();
        obj->RenderComplexMesh();
        delete obj;
    }
    
    GameObject* player = GameObject::FindByName("Player");
    player->Update();
}
        """.strip(),
        
        # Test 3: Another problematic code
        """
class GameManager {
    void UpdateAll() {
        for (auto& entity : entities) {
            entity->ProcessAI();
            entity->UpdatePhysics();
            entity->RenderGraphics();
            entity->PlayAudio();
            entity->HandleInput();
        }
    }
};
        """.strip()
    ]
    
    # Initialize inference
    inferencer = DeepSeekInference()
    
    for i, code_snippet in enumerate(test_snippets, 1):
        print(f"\n{'='*20} Test {i} {'='*20}")
        print("Code snippet:")
        print(code_snippet)
        
        # Load detection model and detect smells
        detection_model = inferencer.load_detection_model()
        detected_smells = inferencer.detect_smells(detection_model, code_snippet)
        
        print(f"\nDetected smells:")
        print(json.dumps(detected_smells, indent=2))
        
        # Clean up detection model
        del detection_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Load fixing model and fix smells
        fixing_model = inferencer.load_fixing_model()
        fixed_result = inferencer.fix_smells(fixing_model, code_snippet, detected_smells)
        
        print(f"\nFixed code and explanations:")
        print(json.dumps(fixed_result, indent=2))
        
        # Clean up fixing model
        del fixing_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print("-" * 50)

if __name__ == "__main__":
    main()