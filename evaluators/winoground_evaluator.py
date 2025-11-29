"""
Winoground Dataset Evaluator
评估模型在Winoground基准上的表现
"""

import os
import json
from typing import Dict, List, Any
from PIL import Image
from .base_evaluator import BaseEvaluator


class WinogroundEvaluator(BaseEvaluator):
    """
    Winoground数据集评估器
    
    Winoground测试视觉-语言组合理解能力
    每个样本包含2张图片和2个文本描述
    评估指标：
    - WinoText: 文本匹配准确率
    - WinoImage: 图像匹配准确率  
    - WinoGroup: 组合匹配准确率
    """
    
    def __init__(self, dataset_path: str, hf_token: str = None):
        """
        初始化Winoground评估器
        
        Args:
            dataset_path: 数据集路径，应包含images/和annotations.json
            hf_token: Hugging Face访问令牌（可选，用于从HF加载数据集）
        """
        super().__init__("Winoground", dataset_path)
        self.images_dir = os.path.join(dataset_path, "images")
        self.annotations_file = os.path.join(dataset_path, "annotations.json")
        self.hf_token = hf_token or os.environ.get('HF_TOKEN')
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """
        加载Winoground数据集
        
        Returns:
            数据样本列表
        """
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
        
        # 方式3: 否则尝试从Hugging Face数据集加载
        try:
            from datasets import load_dataset
            
            # 使用use_auth_token参数加载数据集
            if self.hf_token:
                print(f"Loading Winoground from Hugging Face with authentication...")
                dataset = load_dataset("facebook/winoground", use_auth_token=self.hf_token, split="test")
            else:
                print(f"Warning: No HF token provided. Attempting to load without authentication...")
                print(f"If loading fails, please set HF_TOKEN environment variable or pass hf_token parameter.")
                dataset = load_dataset("facebook/winoground", split="test")
            
            print(f"Successfully loaded {len(dataset)} samples from Hugging Face")
            
            # 保存图像到本地
            os.makedirs(self.images_dir, exist_ok=True)
            
            samples = []
            for i, item in enumerate(dataset):
                sample_id = item.get('id', i)
                
                # 保存图像
                img_0_path = os.path.join(self.images_dir, f"{sample_id}_img0.png")
                img_1_path = os.path.join(self.images_dir, f"{sample_id}_img1.png")
                
                item['image_0'].save(img_0_path)
                item['image_1'].save(img_1_path)
                
                samples.append({
                    "id": sample_id,
                    "caption_0": item['caption_0'],
                    "caption_1": item['caption_1'],
                    "image_0_path": img_0_path,
                    "image_1_path": img_1_path,
                    "tag": item.get('tag', ''),  # 保存tag信息
                })
                
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i + 1}/{len(dataset)} samples...")
            
            # 保存annotations
            with open(self.annotations_file, 'w', encoding='utf-8') as f:
                json.dump(samples, f, indent=2, ensure_ascii=False)
            
            print(f"Saved dataset to {self.annotations_file}")
            return samples
        except Exception as e:
            raise RuntimeError(f"Failed to load Winoground dataset: {e}")
    
    def _load_from_jsonl(self, jsonl_file: str) -> List[Dict[str, Any]]:
        """
        从examples.jsonl文件加载数据（官方下载的格式）
        
        Args:
            jsonl_file: JSONL文件路径
            
        Returns:
            数据样本列表
        """
        samples = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    # 转换为统一格式
                    sample_id = item['id']
                    # 图像文件名格式: ex_0_img_0.png
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
        评估单个Winoground样本
        
        策略：
        1. 对每个(图像,文本)对，使用模型判断是否匹配
        2. 计算四种配对的得分：
           - (image_0, caption_0)
           - (image_0, caption_1)
           - (image_1, caption_0)
           - (image_1, caption_1)
        
        Args:
            sample: 包含2张图片和2个文本的样本
            model_controller: V-CoT控制器
            
        Returns:
            评估结果
        """
        # lxz，这里是wino数据集单个文件的评估函数
        sample_id = sample.get("id", "unknown")
        caption_0 = sample["caption_0"]
        caption_1 = sample["caption_1"]
        image_0_path = sample["image_0_path"]
        image_1_path = sample["image_1_path"]
        
        # 调试：检查图像路径是否存在
        print(f"    [DEBUG] Sample ID: {sample_id}")
        print(f"    [DEBUG] Caption 0: {caption_0}")
        print(f"    [DEBUG] Caption 1: {caption_1}")
        print(f"    [DEBUG] Image 0 path: {image_0_path}")
        print(f"    [DEBUG] Image 0 exists: {os.path.exists(image_0_path)}")
        print(f"    [DEBUG] Image 1 path: {image_1_path}")
        print(f"    [DEBUG] Image 1 exists: {os.path.exists(image_1_path)}")
        
        # 存储4种配对的得分
        scores = {}
        
        # 评估 (image_0, caption_0)
        print(f"    [1/4] Evaluating (image_0, caption_0)...")
        scores["c0_i0"] = self._evaluate_pair(
            image_0_path, caption_0, model_controller
        )
        print(f"    [1/4] Score: {scores['c0_i0']:.2f}")
        
        # 评估 (image_0, caption_1)
        print(f"    [2/4] Evaluating (image_0, caption_1)...")
        scores["c1_i0"] = self._evaluate_pair(
            image_0_path, caption_1, model_controller
        )
        print(f"    [2/4] Score: {scores['c1_i0']:.2f}")
        
        # 评估 (image_1, caption_0)
        print(f"    [3/4] Evaluating (image_1, caption_0)...")
        scores["c0_i1"] = self._evaluate_pair(
            image_1_path, caption_0, model_controller
        )
        print(f"    [3/4] Score: {scores['c0_i1']:.2f}")
        
        # 评估 (image_1, caption_1)
        print(f"    [4/4] Evaluating (image_1, caption_1)...")
        scores["c1_i1"] = self._evaluate_pair(
            image_1_path, caption_1, model_controller
        )
        print(f"    [4/4] Score: {scores['c1_i1']:.2f}")
        
        # 计算准确性
        # Text score: 对于每个image，选择得分更高的caption
        text_i0_correct = scores["c0_i0"] > scores["c1_i0"]  # image_0应匹配caption_0
        text_i1_correct = scores["c1_i1"] > scores["c0_i1"]  # image_1应匹配caption_1
        text_score = 1.0 if (text_i0_correct and text_i1_correct) else 0.0
        
        # Image score: 对于每个caption，选择得分更高的image
        image_c0_correct = scores["c0_i0"] > scores["c0_i1"]  # caption_0应匹配image_0
        image_c1_correct = scores["c1_i1"] > scores["c1_i0"]  # caption_1应匹配image_1
        image_score = 1.0 if (image_c0_correct and image_c1_correct) else 0.0
        
        # Group score: text和image都正确
        group_score = 1.0 if (text_score == 1.0 and image_score == 1.0) else 0.0
        
        return {
            "sample_id": sample_id,
            "scores": scores,
            "text_score": text_score,
            "image_score": image_score,
            "group_score": group_score,
            "success": True
        }
    
    def _generate_explanation(self, 
                             image_path: str, 
                             caption: str, 
                             model_controller) -> str:
        """
        为图像-文本对生成解释（参考CCoT方法）
        
        Args:
            image_path: 图像路径
            caption: 文本描述
            model_controller: V-CoT控制器
            
        Returns:
            解释文本
        """
        # 使用CCoT风格的prompt
        question = f"Does the given caption accurately describe the given image? Caption: {caption}\n\nProvide a detailed explanation with reasoning."
        
        print(f"      [DEBUG] Generating explanation for: {caption[:50]}...")
        print(f"      [DEBUG] Image: {os.path.basename(image_path)}")
        
        try:
            import time
            start_time = time.time()
            result = model_controller.run(
                image_path=image_path,
                question=question,
                verbose=False
            )
            elapsed = time.time() - start_time
            print(f"      → Explanation generated in {elapsed:.1f}s")
            
            explanation = result.get("answer", "")
            print(f"      → Explanation: {explanation[:100]}...")
            return explanation
                
        except Exception as e:
            print(f"      Error generating explanation: {e}")
            return ""
    
    def _evaluate_pair(self, image_path: str, caption: str, model_controller) -> float:
        """
        评估单个图像-文本对的匹配程度
        
        Args:
            image_path: 图像路径
            caption: 文本描述
            model_controller: V-CoT控制器
            
        Returns:
            匹配得分（0-1之间）
        """
        # 构建判断问题
        question = f"Does this image accurately depict the following description? Description: '{caption}' Answer with Yes or No and provide a confidence level."
        
        # 调试：检查传递给V-CoT的图像路径
        print(f"      [DEBUG] Calling V-CoT with image: {image_path}")
        print(f"      [DEBUG] Image exists: {os.path.exists(image_path)}")
        print(f"      [DEBUG] Question: {question[:100]}...")
        
        try:
            # 使用V-CoT进行推理（这会调用多个API：Planner、Perceiver、Verifier等）
            import time
            start_time = time.time()
            # lxz，这里是调用框架后的结果
            result = model_controller.run(
                image_path=image_path,
                question=question,
                verbose=False
            )
            elapsed = time.time() - start_time
            print(f"      → V-CoT inference completed in {elapsed:.1f}s")
            
            answer = result.get("answer", "").lower()
            
            # 解析答案和置信度
            if "yes" in answer:
                # 尝试提取置信度，如果没有则默认为1.0
                if "high" in answer or "very" in answer or "definitely" in answer:
                    return 1.0
                elif "low" in answer or "uncertain" in answer:
                    return 0.6
                else:
                    return 0.8
            else:
                return 0.0
                
        except Exception as e:
            print(f"Error evaluating pair: {e}")
            return 0.0
    
    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        计算Winoground评估指标
        
        Args:
            results: 所有样本的评估结果
            
        Returns:
            包含WinoText, WinoImage, WinoGroup的指标字典
        """
        # 过滤成功的结果
        valid_results = [r for r in results if r.get("success", False)]
        
        if not valid_results:
            return {
                "WinoText": 0.0,
                "WinoImage": 0.0,
                "WinoGroup": 0.0,
                "total_samples": len(results),
                "valid_samples": 0
            }
        
        # 计算平均得分
        text_scores = [r["text_score"] for r in valid_results]
        image_scores = [r["image_score"] for r in valid_results]
        group_scores = [r["group_score"] for r in valid_results]
        
        metrics = {
            "WinoText": sum(text_scores) / len(text_scores) * 100,  # 转换为百分比
            "WinoImage": sum(image_scores) / len(image_scores) * 100,
            "WinoGroup": sum(group_scores) / len(group_scores) * 100,
            "total_samples": len(results),
            "valid_samples": len(valid_results)
        }
        
        return metrics
