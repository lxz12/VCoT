"""
WHOOPS! Dataset Evaluator
评估模型在WHOOPS!基准上的表现
WHOOPS!是一个关于识别和解释图像中不寻常元素的数据集
"""

import os
import json
from typing import Dict, List, Any
from PIL import Image
from .base_evaluator import BaseEvaluator


class WHOOPSEvaluator(BaseEvaluator):
    """
    WHOOPS!数据集评估器
    
    WHOOPS!测试模型识别和解释图像中不寻常或违反常识元素的能力
    每个样本包含：
    - 一张包含违反常识元素的图像
    - 多个question-answer pairs
    
    评估指标：
    - Accuracy: 回答准确率
    - Average Score: 平均得分
    """
    
    def __init__(self, dataset_path: str, hf_token: str = None):
        """
        初始化WHOOPS!评估器
        
        Args:
            dataset_path: 数据集路径，应包含images/和annotations.json
            hf_token: Hugging Face访问令牌（可选，用于从HF加载数据集）
        """
        super().__init__("WHOOPS", dataset_path)
        self.images_dir = os.path.join(dataset_path, "images")
        self.annotations_file = os.path.join(dataset_path, "annotations.json")
        self.hf_token = hf_token or os.environ.get('HF_TOKEN')
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """
        加载WHOOPS!数据集
        
        Returns:
            数据样本列表
        """
        # 如果有annotations.json文件
        if os.path.exists(self.annotations_file):
            with open(self.annotations_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, list) else data.get('data', [])
        
        # 否则尝试从Hugging Face数据集加载
        try:
            from datasets import load_dataset
            
            # 使用use_auth_token参数加载数据集
            if self.hf_token:
                print(f"Loading WHOOPS! from Hugging Face with authentication...")
                dataset = load_dataset("nlphuji/whoops", use_auth_token=self.hf_token, split="test")
            else:
                print(f"Warning: No HF token provided. Attempting to load without authentication...")
                dataset = load_dataset("nlphuji/whoops", split="test")
            
            print(f"Successfully loaded {len(dataset)} samples from Hugging Face")
            
            # 保存图像到本地
            os.makedirs(self.images_dir, exist_ok=True)
            
            samples = []
            for i, item in enumerate(dataset):
                image_id = item.get('image_id', i)
                
                # 保存图像
                image_path = os.path.join(self.images_dir, f"{image_id}.png")
                item['image'].save(image_path)
                
                samples.append({
                    "image_id": image_id,
                    "image_path": image_path,
                    "question_answering_pairs": item.get('question_answering_pairs', []),
                    "designer_explanation": item.get('designer_explanation', ''),
                    "selected_caption": item.get('selected_caption', '')
                })
                
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i + 1}/{len(dataset)} samples...")
            
            # 保存annotations
            with open(self.annotations_file, 'w', encoding='utf-8') as f:
                json.dump(samples, f, indent=2, ensure_ascii=False)
            
            print(f"Saved dataset to {self.annotations_file}")
            return samples
        except Exception as e:
            raise RuntimeError(f"Failed to load WHOOPS! dataset: {e}")
    
    def evaluate_sample(self, sample: Dict[str, Any], model_controller) -> Dict[str, Any]:
        """
        评估单个WHOOPS!样本
        
        策略：
        1. 对每个question-answer pair，使用模型生成回答
        2. 使用GPT评估回答的准确性和质量
        
        Args:
            sample: 包含图像和question_answering_pairs的样本
            model_controller: V-CoT控制器
            
        Returns:
            评估结果
        """
        image_id = sample.get("image_id", "unknown")
        image_path = sample["image_path"]
        qa_pairs = sample.get("question_answering_pairs", [])
        
        if not qa_pairs:
            return {
                "image_id": image_id,
                "success": False,
                "error": "No question-answer pairs found"
            }
        
        predictions = []
        scores = []
        
        for qa_pair in qa_pairs:
            if len(qa_pair) < 2:
                continue
            
            question = qa_pair[0]
            ground_truth = qa_pair[1]
            
            try:
                # 使用V-CoT进行推理
                result = model_controller.run(
                    image_path=image_path,
                    question=question,
                    verbose=False
                )
                
                prediction = result.get("answer", "")
                predictions.append(prediction)
                
                # 简单的评估：检查关键词匹配
                score = self._evaluate_answer(prediction, ground_truth)
                scores.append(score)
                
            except Exception as e:
                print(f"Error evaluating question for image {image_id}: {e}")
                predictions.append("")
                scores.append(0.0)
        
        # 计算平均得分
        avg_score = sum(scores) / len(scores) if scores else 0.0
        accuracy = sum(1 for s in scores if s >= 0.5) / len(scores) if scores else 0.0
        
        return {
            "image_id": image_id,
            "predictions": predictions,
            "scores": scores,
            "avg_score": avg_score,
            "accuracy": accuracy,
            "num_questions": len(qa_pairs),
            "success": True
        }
    
    def _evaluate_answer(self, prediction: str, ground_truth: str) -> float:
        """
        评估回答质量
        
        简单方法：检查关键词重叠
        更高级的方法可以使用LLM进行语义评估
        """
        pred_lower = prediction.lower()
        gt_lower = ground_truth.lower()
        
        # 关键词匹配
        gt_words = set(gt_lower.split())
        pred_words = set(pred_lower.split())
        
        if len(gt_words) == 0:
            return 0.0
        
        overlap = len(gt_words.intersection(pred_words))
        score = overlap / len(gt_words)
        
        return min(score, 1.0)
    
    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        计算WHOOPS!评估指标
        
        Args:
            results: 所有样本的评估结果
            
        Returns:
            包含Accuracy和Average Score的指标字典
        """
        # 过滤成功的结果
        valid_results = [r for r in results if r.get("success", False)]
        
        if not valid_results:
            return {
                "Accuracy": 0.0,
                "Average_Score": 0.0,
                "total_samples": len(results),
                "valid_samples": 0
            }
        
        # 计算平均指标
        accuracies = [r["accuracy"] for r in valid_results]
        avg_scores = [r["avg_score"] for r in valid_results]
        
        metrics = {
            "Accuracy": sum(accuracies) / len(accuracies) * 100,  # 转换为百分比
            "Average_Score": sum(avg_scores) / len(avg_scores) * 100,
            "total_samples": len(results),
            "valid_samples": len(valid_results),
            "total_questions": sum(r.get("num_questions", 0) for r in valid_results)
        }
        
        return metrics
