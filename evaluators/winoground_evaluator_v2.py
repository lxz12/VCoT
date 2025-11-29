"""
Winoground Evaluator V2
参考CCoT的评测方法，实现正确的Text/Image/Group三个指标
"""

import os
import json
from typing import Dict, List, Any, Optional
from .base_evaluator import BaseEvaluator


class WinogroundEvaluatorV2(BaseEvaluator):
    """Winoground数据集评估器 - 使用CCoT方法"""
    
    def __init__(self, dataset_path: str, hf_token: Optional[str] = None):
        super().__init__(dataset_name="winoground", dataset_path=dataset_path)
        self.hf_token = hf_token
        self.images_dir = os.path.join(dataset_path, "images")
        self.annotations_file = os.path.join(dataset_path, "annotations.json")
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """加载Winoground数据集"""
        # 方式1: 如果有annotations.json文件（自动生成的格式）
        if os.path.exists(self.annotations_file):
            print(f"Loading from local annotations: {self.annotations_file}")
            with open(self.annotations_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, list) else data.get('data', [])
        
        # 方式2: 如果有examples.jsonl文件（官方下载的格式）
        jsonl_file = os.path.join(self.dataset_path, "examples.jsonl")
        if os.path.exists(jsonl_file):
            print(f"Loading from local JSONL: {jsonl_file}")
            return self._load_from_jsonl(jsonl_file)
        
        raise FileNotFoundError(f"No data found in {self.dataset_path}")
    
    def _load_from_jsonl(self, jsonl_file: str) -> List[Dict[str, Any]]:
        """从examples.jsonl文件加载数据"""
        samples = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    sample_id = item['id']
                    image_0_name = f"{item['image_0']}.png"
                    image_1_name = f"{item['image_1']}.png"
                    
                    samples.append({
                        "id": sample_id,
                        "caption_0": item['caption_0'],
                        "caption_1": item['caption_1'],
                        "image_0_path": os.path.join(self.images_dir, image_0_name),
                        "image_1_path": os.path.join(self.images_dir, image_1_name),
                        "tag": item.get('tag', ''),
                    })
        
        print(f"Successfully loaded {len(samples)} samples from JSONL")
        return samples
    
    def evaluate_sample(self, sample: Dict[str, Any], model_controller) -> Dict[str, Any]:
        """
        评估单个样本 - 使用CCoT方法
        
        步骤：
        1. 为4个图像-文本对生成解释
        2. 使用LLM比较解释，计算Text/Image/Group得分
        """
        sample_id = sample.get("id", "unknown")
        caption_0 = sample["caption_0"]
        caption_1 = sample["caption_1"]
        image_0_path = sample["image_0_path"]
        image_1_path = sample["image_1_path"]
        
        print(f"    [Sample {sample_id}] Generating explanations for 4 pairs...")
        
        # 生成4个解释
        explanations = {}
        
        print(f"    [1/4] (image_0, caption_0)...")
        explanations["c0_i0"] = self._generate_explanation(
            image_0_path, caption_0, model_controller
        )
        
        print(f"    [2/4] (image_0, caption_1)...")
        explanations["c1_i0"] = self._generate_explanation(
            image_0_path, caption_1, model_controller
        )
        
        print(f"    [3/4] (image_1, caption_0)...")
        explanations["c0_i1"] = self._generate_explanation(
            image_1_path, caption_0, model_controller
        )
        
        print(f"    [4/4] (image_1, caption_1)...")
        explanations["c1_i1"] = self._generate_explanation(
            image_1_path, caption_1, model_controller
        )
        
        # 计算Text Score: 对于每个图像，选择更合理的caption解释
        print(f"    Computing Text Score...")
        text_result_i0 = self._compare_explanations_for_text(
            caption_0, caption_1, 
            explanations["c0_i0"], explanations["c1_i0"],
            model_controller
        )
        text_result_i1 = self._compare_explanations_for_text(
            caption_0, caption_1,
            explanations["c0_i1"], explanations["c1_i1"],
            model_controller
        )
        text_score = 1.0 if (text_result_i0 == "A" and text_result_i1 == "B") else 0.0
        
        # 计算Image Score: 对于每个caption，选择更合理的图像解释
        print(f"    Computing Image Score...")
        image_result_c0 = self._compare_explanations_for_image(
            caption_0,
            explanations["c0_i0"], explanations["c0_i1"],
            model_controller
        )
        image_result_c1 = self._compare_explanations_for_image(
            caption_1,
            explanations["c1_i0"], explanations["c1_i1"],
            model_controller
        )
        image_score = 1.0 if (image_result_c0 == "A" and image_result_c1 == "B") else 0.0
        
        # 计算Group Score
        group_score = 1.0 if (text_score == 1.0 and image_score == 1.0) else 0.0
        
        print(f"    Results: Text={text_score}, Image={image_score}, Group={group_score}")
        
        return {
            "sample_id": sample_id,
            "explanations": explanations,
            "text_score": text_score,
            "image_score": image_score,
            "group_score": group_score,
            "success": True
        }
    
    def _generate_explanation(self, image_path: str, caption: str, model_controller) -> str:
        """为图像-文本对生成解释"""
        # 使用CCoT风格的prompt
        question = f"Does the given caption accurately describe the given image? Caption: {caption}\n\nProvide a detailed explanation with reasoning."
        
        try:
            result = model_controller.run(
                image_path=image_path,
                question=question,
                verbose=False
            )
            return result.get("answer", "")
        except Exception as e:
            print(f"      Error: {e}")
            return ""
    
    def _compare_explanations_for_text(self, caption_0: str, caption_1: str, 
                                       exp_0: str, exp_1: str, 
                                       model_controller) -> str:
        """
        Text Score比较：选择更合理的caption解释
        参考CCoT的get_ans函数
        """
        prompt = f"""Caption A: {caption_0}. Explanation A: {exp_0}

Caption B: {caption_1}. Explanation B: {exp_1}

Each explanation tries to justify why an image match with the corresponding caption. Pick the most logical explanation and return only an alphabet letter (A or B)."""
        
        try:
            # 直接调用LLM，不需要V-CoT
            from utils import LLMDriver
            llm = model_controller.llm_driver
            response = llm.call_llm(
                image=None,
                text_prompt=prompt,
                max_tokens=10,
                temperature=0
            )
            # 提取A或B
            if "A" in response.upper():
                return "A"
            elif "B" in response.upper():
                return "B"
            return "A"  # 默认
        except Exception as e:
            print(f"      Error comparing: {e}")
            return "A"
    
    def _compare_explanations_for_image(self, caption: str, 
                                        exp_0: str, exp_1: str,
                                        model_controller) -> str:
        """
        Image Score比较：选择更符合caption的图像解释
        参考CCoT的get_ans2函数
        """
        prompt = f"""Caption: {caption}. Explanation A: {exp_0} Explanation B: {exp_1}

Pick the explanation with information that align with the caption and return only an alphabet letter (A or B)."""
        
        try:
            llm = model_controller.llm_driver
            response = llm.call_llm(
                image=None,
                text_prompt=prompt,
                max_tokens=10,
                temperature=0
            )
            if "A" in response.upper():
                return "A"
            elif "B" in response.upper():
                return "B"
            return "A"
        except Exception as e:
            print(f"      Error comparing: {e}")
            return "A"
    
    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算Winoground的3个指标"""
        if not results:
            return {"text_score": 0.0, "image_score": 0.0, "group_score": 0.0}
        
        successful_results = [r for r in results if r.get("success", False)]
        if not successful_results:
            return {"text_score": 0.0, "image_score": 0.0, "group_score": 0.0}
        
        text_scores = [r["text_score"] for r in successful_results]
        image_scores = [r["image_score"] for r in successful_results]
        group_scores = [r["group_score"] for r in successful_results]
        
        return {
            "text_score": sum(text_scores) / len(text_scores),
            "image_score": sum(image_scores) / len(image_scores),
            "group_score": sum(group_scores) / len(group_scores),
            "num_samples": len(successful_results)
        }
