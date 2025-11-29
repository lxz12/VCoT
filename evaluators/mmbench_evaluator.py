"""
MMBench Dataset Evaluator
评估模型在MMBench基准上的表现
MMBench是一个系统化的多模态基准测试
"""

import os
import json
import pandas as pd
from typing import Dict, List, Any
from PIL import Image
import base64
from io import BytesIO
from .base_evaluator import BaseEvaluator


class MMBenchEvaluator(BaseEvaluator):
    """
    MMBench数据集评估器
    
    MMBench (Multimodal Benchmark)
    测试多个维度的视觉-语言理解能力
    使用选择题格式
    
    评估指标：
    - Accuracy: 整体准确率
    - Per-Category Accuracy: 各类别准确率
    """
    
    def __init__(self, dataset_path: str):
        """
        初始化MMBench评估器
        
        Args:
            dataset_path: 数据集路径，应包含tsv文件
        """
        super().__init__("MMBench", dataset_path)
        self.data_file = os.path.join(dataset_path, "mmbench_test.tsv")
        
        # 如果没找到，尝试其他常见文件名
        if not os.path.exists(self.data_file):
            possible_files = [
                "mmbench_dev_20230712.tsv",
                "mmbench_dev.tsv",
                "MMBench_DEV_EN.tsv",
                "MMBench_TEST_EN.tsv"
            ]
            for fname in possible_files:
                fpath = os.path.join(dataset_path, fname)
                if os.path.exists(fpath):
                    self.data_file = fpath
                    break
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """
        加载MMBench数据集
        
        MMBench使用TSV格式
        
        Returns:
            数据样本列表
        """
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(
                f"MMBench data file not found at {self.data_file}. "
                f"Please download from official source."
            )
        
        try:
            print(f"Loading MMBench from {self.data_file}...")
            df = pd.read_table(self.data_file)
            
            samples = []
            for idx, row in df.iterrows():
                sample = {
                    "index": row.get("index", idx),
                    "question": row.get("question", ""),
                    "image": row.get("image", ""),  # 通常是base64编码
                    "A": row.get("A", ""),
                    "B": row.get("B", ""),
                    "C": row.get("C", None),
                    "D": row.get("D", None),
                    "answer": row.get("answer", ""),
                    "category": row.get("category", ""),
                    "l2-category": row.get("l2-category", "")
                }
                samples.append(sample)
            
            print(f"Successfully loaded {len(samples)} samples")
            return samples
            
        except Exception as e:
            raise RuntimeError(f"Failed to load MMBench dataset: {e}")
    
    def evaluate_sample(self, sample: Dict[str, Any], model_controller) -> Dict[str, Any]:
        """
        评估单个MMBench样本
        
        MMBench是选择题格式，通常是2-4个选项
        
        Args:
            sample: 包含question, options和image的样本
            model_controller: V-CoT控制器
            
        Returns:
            评估结果
        """
        index = sample.get("index", "unknown")
        question = sample.get("question", "")
        category = sample.get("category", "unknown")
        l2_category = sample.get("l2-category", "unknown")
        
        # 获取选项
        options = []
        option_chars = ['A', 'B', 'C', 'D']
        for char in option_chars:
            opt_value = sample.get(char)
            if opt_value is not None and not pd.isna(opt_value) and opt_value != '':
                options.append(opt_value)
            else:
                break
        
        # 构建完整问题
        full_question = question
        for i, (char, opt) in enumerate(zip(option_chars[:len(options)], options)):
            full_question += f"\n{char}. {opt}"
        full_question += "\nAnswer with the option's letter from the given choices directly."
        
        # 处理图像
        image_data = sample.get("image", "")
        
        try:
            # 如果image_data是base64编码
            if isinstance(image_data, str) and image_data.startswith('data:image'):
                # 提取base64部分
                base64_str = image_data.split(',')[1] if ',' in image_data else image_data
                image_bytes = base64.b64decode(base64_str)
                image = Image.open(BytesIO(image_bytes))
                
                # 保存临时图像
                temp_image_path = f"/tmp/mmbench_{index}.png"
                image.save(temp_image_path)
                image_path = temp_image_path
            else:
                # 假设是文件路径
                image_path = os.path.join(self.dataset_path, "images", image_data)
            
            # 使用V-CoT进行推理
            result = model_controller.run(
                image_path=image_path,
                question=full_question,
                verbose=False
            )
            
            prediction = result.get("answer", "").strip()
            ground_truth = sample.get("answer", "")
            
            # 评估回答
            correct = self._check_answer(prediction, ground_truth, options)
            
            return {
                "index": index,
                "category": category,
                "l2_category": l2_category,
                "prediction": prediction,
                "ground_truth": ground_truth,
                "correct": correct,
                "success": True
            }
            
        except Exception as e:
            print(f"Error evaluating question {index}: {e}")
            return {
                "index": index,
                "category": category,
                "l2_category": l2_category,
                "error": str(e),
                "correct": False,
                "success": False
            }
    
    def _check_answer(self, prediction: str, ground_truth: str, options: List[str]) -> bool:
        """
        检查回答是否正确
        
        提取预测中的选项字母并与ground_truth比较
        """
        pred_lower = prediction.lower().strip()
        gt_lower = ground_truth.lower().strip()
        
        # 直接匹配
        if pred_lower == gt_lower:
            return True
        
        # 提取选项字母（从预测中找第一个A/B/C/D）
        for char in pred_lower:
            if char in 'abcd':
                return char == gt_lower.lower()
        
        # 如果预测包含完整选项文本
        if options:
            try:
                gt_idx = ord(gt_lower) - ord('a')
                if 0 <= gt_idx < len(options):
                    gt_text = options[gt_idx].lower()
                    if gt_text in pred_lower:
                        return True
            except:
                pass
        
        return False
    
    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        计算MMBench评估指标
        
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
        
        # 按一级类别计算准确率
        category_stats = {}
        for result in valid_results:
            cat = result.get("category", "unknown")
            if cat not in category_stats:
                category_stats[cat] = {"correct": 0, "total": 0}
            
            category_stats[cat]["total"] += 1
            if result.get("correct", False):
                category_stats[cat]["correct"] += 1
        
        # 按二级类别计算准确率
        l2_category_stats = {}
        for result in valid_results:
            l2_cat = result.get("l2_category", "unknown")
            if l2_cat not in l2_category_stats:
                l2_category_stats[l2_cat] = {"correct": 0, "total": 0}
            
            l2_category_stats[l2_cat]["total"] += 1
            if result.get("correct", False):
                l2_category_stats[l2_cat]["correct"] += 1
        
        # 构建指标字典
        metrics = {
            "Accuracy": overall_accuracy,
            "total_samples": len(results),
            "valid_samples": len(valid_results),
            "correct_count": correct_count
        }
        
        # 添加一级类别准确率
        for cat, stats in category_stats.items():
            cat_acc = stats["correct"] / stats["total"] * 100
            metrics[f"Accuracy_{cat}"] = cat_acc
        
        # 添加二级类别准确率（可选，如果类别太多可以注释掉）
        for l2_cat, stats in l2_category_stats.items():
            l2_acc = stats["correct"] / stats["total"] * 100
            # 使用简短名称避免过长
            short_name = l2_cat.replace(" ", "_")[:20]
            metrics[f"Acc_L2_{short_name}"] = l2_acc
        
        return metrics
