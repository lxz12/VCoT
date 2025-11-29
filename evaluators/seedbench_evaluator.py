"""
SEED-Bench Dataset Evaluator
评估模型在SEED-Bench基准上的表现
SEED-Bench是一个全面的多模态理解基准
"""

import os
import json
from typing import Dict, List, Any
from PIL import Image
from .base_evaluator import BaseEvaluator


class SEEDBenchEvaluator(BaseEvaluator):
    """
    SEED-Bench数据集评估器
    
    SEED-Bench (Multimodal Comprehension Benchmark)
    测试多个维度的视觉理解能力
    
    评估指标：
    - Accuracy: 整体准确率
    - Per-Category Accuracy: 各类别准确率
    """
    
    def __init__(self, dataset_path: str, hf_token: str = None):
        """
        初始化SEED-Bench评估器
        
        Args:
            dataset_path: 数据集路径，应包含images/和annotations.jsonl
            hf_token: Hugging Face访问令牌（可选）
        """
        super().__init__("SEED-Bench", dataset_path)
        self.images_dir = os.path.join(dataset_path, "images")
        self.annotations_file = os.path.join(dataset_path, "annotations.jsonl")
        self.hf_token = hf_token or os.environ.get('HF_TOKEN')
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """
        加载SEED-Bench数据集
        
        Returns:
            数据样本列表
        """
        # 如果有annotations.jsonl文件
        if os.path.exists(self.annotations_file):
            samples = []
            with open(self.annotations_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
            return samples
        
        # 否则尝试从本地或HF加载
        # 注意：SEED-Bench通常需要从官方网站下载
        # 这里提供基本的加载逻辑
        try:
            print(f"Loading SEED-Bench dataset from {self.dataset_path}...")
            
            # 如果存在预处理的jsonl文件（如CCoT中的llava-seed-bench-filtered.jsonl）
            ccot_file = os.path.join(self.dataset_path, "llava-seed-bench-filtered.jsonl")
            if os.path.exists(ccot_file):
                print(f"Found filtered SEED-Bench file: {ccot_file}")
                samples = []
                with open(ccot_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            samples.append(json.loads(line))
                
                # 保存为标准格式
                with open(self.annotations_file, 'w', encoding='utf-8') as f:
                    for sample in samples:
                        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                
                print(f"Successfully loaded {len(samples)} samples")
                return samples
            
            raise FileNotFoundError(
                f"SEED-Bench data not found. Please download from official source "
                f"or provide annotations at {self.annotations_file}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load SEED-Bench dataset: {e}")
    
    def evaluate_sample(self, sample: Dict[str, Any], model_controller) -> Dict[str, Any]:
        """
        评估单个SEED-Bench样本
        
        SEED-Bench是多选题格式
        
        Args:
            sample: 包含question, choices和answer的样本
            model_controller: V-CoT控制器
            
        Returns:
            评估结果
        """
        question_id = sample.get("question_id", "unknown")
        question_type = sample.get("question_type", "unknown")
        
        # 构建完整问题
        text = sample.get("text", sample.get("question", ""))
        image_path = sample.get("image", sample.get("image_path", ""))
        
        # 如果image_path是相对路径，添加images_dir前缀
        if image_path and not os.path.isabs(image_path):
            image_path = os.path.join(self.images_dir, image_path)
        
        # 获取选项（如果有）
        choices = sample.get("choices", [])
        answer = sample.get("answer", sample.get("correct_choice", ""))
        
        try:
            # 使用V-CoT进行推理
            result = model_controller.run(
                image_path=image_path,
                question=text,
                verbose=False
            )
            
            prediction = result.get("answer", "").strip()
            
            # 评估回答
            correct = self._check_answer(prediction, answer, choices)
            
            return {
                "question_id": question_id,
                "question_type": question_type,
                "prediction": prediction,
                "ground_truth": answer,
                "correct": correct,
                "success": True
            }
            
        except Exception as e:
            print(f"Error evaluating question {question_id}: {e}")
            return {
                "question_id": question_id,
                "question_type": question_type,
                "error": str(e),
                "correct": False,
                "success": False
            }
    
    def _check_answer(self, prediction: str, ground_truth: str, choices: List[str] = None) -> bool:
        """
        检查回答是否正确
        
        支持多种答案格式：
        - 选项字母 (A, B, C, D)
        - 完整答案文本
        """
        pred_lower = prediction.lower().strip()
        gt_lower = ground_truth.lower().strip()
        
        # 直接匹配
        if pred_lower == gt_lower:
            return True
        
        # 提取选项字母
        if len(gt_lower) == 1 and gt_lower in 'abcd':
            # 从预测中提取第一个选项字母
            for char in pred_lower:
                if char in 'abcd':
                    return char == gt_lower
        
        # 如果有choices，检查语义匹配
        if choices:
            try:
                gt_idx = ord(gt_lower) - ord('a')
                if 0 <= gt_idx < len(choices):
                    gt_text = choices[gt_idx].lower()
                    if gt_text in pred_lower or pred_lower in gt_text:
                        return True
            except:
                pass
        
        # 关键词匹配
        if gt_lower in pred_lower:
            return True
        
        return False
    
    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        计算SEED-Bench评估指标
        
        Args:
            results: 所有样本的评估结果
            
        Returns:
            包含整体和分类准确率的指标字典
        """
        # 过滤成功的结果
        valid_results = [r for r in results if r.get("success", False)]
        
        if not valid_results:
            return {
                "Accuracy": 0.0,
                "total_samples": len(results),
                "valid_samples": 0
            }
        
        # 计算整体准确率
        correct_count = sum(1 for r in valid_results if r.get("correct", False))
        overall_accuracy = correct_count / len(valid_results) * 100
        
        # 按类别计算准确率
        category_stats = {}
        for result in valid_results:
            q_type = result.get("question_type", "unknown")
            if q_type not in category_stats:
                category_stats[q_type] = {"correct": 0, "total": 0}
            
            category_stats[q_type]["total"] += 1
            if result.get("correct", False):
                category_stats[q_type]["correct"] += 1
        
        # 构建指标字典
        metrics = {
            "Accuracy": overall_accuracy,
            "total_samples": len(results),
            "valid_samples": len(valid_results),
            "correct_count": correct_count
        }
        
        # 添加每个类别的准确率
        for q_type, stats in category_stats.items():
            category_acc = stats["correct"] / stats["total"] * 100
            metrics[f"Accuracy_{q_type}"] = category_acc
        
        return metrics
