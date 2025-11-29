import os
import json
import base64
import requests
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
from io import BytesIO


class LLMDriver:
    def __init__(self, model_type: str = "gpt-4-vision-preview", api_key: Optional[str] = None):
        self.model_type = model_type
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY', '')
        if 'gpt' in model_type.lower():
            self.api_base = os.environ.get('OPENAI_API_BASE', 'https://api.openai.com/v1')
        else:
            self.api_base = "http://localhost:8000"
    
    def encode_image(self, image_path: str) -> str:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded
    
    def encode_image_from_pil(self, image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def call_llm(self, 
                 image: Optional[str] = None,
                 text_prompt: str = "",
                 image_pil: Optional[Image.Image] = None,
                 max_tokens: int = 1024,
                 temperature: float = 0) -> str:
        if 'gpt' in self.model_type.lower():
            return self._call_gpt(image, text_prompt, image_pil, max_tokens, temperature)
        else:
            return self._call_local_model(image, text_prompt, image_pil, max_tokens, temperature)
    
    def _call_gpt(self,
                  image: Optional[str],
                  text_prompt: str,
                  image_pil: Optional[Image.Image],
                  max_tokens: int,
                  temperature: float) -> str:
        if image_pil is not None:
            image_b64 = self.encode_image_from_pil(image_pil)
        elif image is not None:
            if os.path.exists(image):
                image_b64 = self.encode_image(image)
            else:
                image_b64 = image
        else:
            image_b64 = None
        content = []
        if image_b64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}"
                }
            })
        content.append({
            "type": "text",
            "text": text_prompt
        })
        payload = {
            "model": self.model_type,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            import time
            api_start = time.time()
            
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            api_elapsed = time.time() - api_start
            
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            return content
        except Exception as e:
            print(f"          → [API] Error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"          → [API] Response: {e.response.text[:200]}")
            raise
    
    def _call_local_model(self,
                         image: Optional[str],
                         text_prompt: str,
                         image_pil: Optional[Image.Image],
                         max_tokens: int,
                         temperature: float) -> str:
        raise NotImplementedError("Local model API not implemented yet")


class ImageProcessor:
    @staticmethod
    def crop_image(image_path: str, bbox: List[float]) -> Image.Image:
        image = Image.open(image_path)
        width, height = image.size
        x1, y1, x2, y2 = bbox
        if max(bbox) <= 1.0:
            x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
        else:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        
        cropped_image = image.crop((x1, y1, x2, y2))
        return cropped_image
    
    @staticmethod
    def crop_image_from_pil(image: Image.Image, bbox: List[float]) -> Image.Image:
        width, height = image.size
        x1, y1, x2, y2 = bbox
        if max(bbox) <= 1.0:
            x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
        else:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        
        return image.crop((x1, y1, x2, y2))


def parse_json_response(response: str) -> Optional[Dict]:
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        import re
        json_pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(json_pattern, response, re.DOTALL)
        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                pass
        
        brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(brace_pattern, response, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        return None


def format_verified_facts(verified_fact_graph: Dict[str, Any]) -> str:
    if not verified_fact_graph:
        return "No verified facts yet."
    
    facts_str = "Verified Facts:\n"
    for fact_id, fact_data in verified_fact_graph.items():
        facts_str += f"- {fact_id}: {fact_data.get('name', 'Unknown')}"
        if 'bbox' in fact_data:
            facts_str += f" (bbox: {fact_data['bbox']})"
        if 'relation' in fact_data:
            facts_str += f" - Relation: {fact_data['relation']}"
        facts_str += "\n"
    
    return facts_str


def save_results(output_path: str, results: Dict[str, Any]):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
