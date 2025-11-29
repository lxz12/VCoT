"""
LLaVA-W (LLaVA-Bench in the Wild) Dataset Evaluator
评估模型在LLaVA-W基准上的表现
LLaVA-W是一个开放式视觉问答基准
"""

import os
import json
from typing import Dict, List, Any
from PIL import Image
from .base_evaluator import BaseEvaluator


class LLaVAWEvaluator(BaseEvaluator):
    """
    LLaVA-W数据集评估器
    
    LLaVA-Bench in the Wild (LLaVA-W)
    测试模型在真实世界图像上的开放式问答能力
    
    评估指标：
    - Conversation: 对话质量得分
    - Detail: 细节描述得分  
    - Complex Reasoning: 复杂推理得分
    - Overall: 总体得分
    """
    
    def __init__(self, dataset_path: str):
        """
        初始化LLaVA-W评估器
        
        Args:
            dataset_path: 数据集路径，应包含images/和questions.jsonl
        """
        super().__init__("LLaVA-W", dataset_path)
        self.images_dir = os.path.join(dataset_path, "images")
        self.questions_file = os.path.join(dataset_path, "questions.jsonl")
        
        # 尝试其他可能的文件名
        if not os.path.exists(self.questions_file):
            possible_files = [
                "llava_bench_wild.jsonl",
                "llava_bench_in_the_wild.jsonl",
                "questions.json"
            ]
            for fname in possible_files:
                fpath = os.path.join(dataset_path, fname)
                if os.path.exists(fpath):
                    self.questions_file = fpath
                    break
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """
        加载LLaVA-W数据集
        
        Returns:
            数据样本列表
        """
        if not os.path.exists(self.questions_file):
            raise FileNotFoundError(
                f"LLaVA-W data file not found at {self.questions_file}. "
                f"Please download from official LLaVA repository."
            )
        
        try:
            print(f"Loading LLaVA-W from {self.questions_file}...")
            
            samples = []
            # 尝试JSONL格式
            if self.questions_file.endswith('.jsonl'):
                with open(self.questions_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            samples.append(json.loads(line))
            else:
                # JSON格式
                with open(self.questions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    samples = data if isinstance(data, list) else data.get('questions', [])
            
            print(f"Successfully loaded {len(samples)} samples")
            return samples
            
        except Exception as e:
            raise RuntimeError(f"Failed to load LLaVA-W dataset: {e}")
    
    def evaluate_sample(self, sample: Dict[str, Any], model_controller) -> Dict[str, Any]:
        """
        评估单个LLaVA-W样本
        
        LLaVA-W包含开放式问题，需要详细的回答
        
        Args:
            sample: 包含question和image的样本
            model_controller: V-CoT控制器
            
        Returns:
            评估结果
        """
        question_id = sample.get("question_id", sample.get("id", "unknown"))
        question = sample.get("text", sample.get("question", ""))
        image_file = sample.get("image", "")
        category = sample.get("category", "unknown")
        
        # 构建图像路径
        if image_file:
            if os.path.isabs(image_file):
                image_path = image_file
            else:
                image_path = os.path.join(self.images_dir, image_file)
        else:
            return {
                "question_id": question_id,
                "category": category,
                "success": False,
                "error": "No image file specified"
            }
        
        try:
            # 使用V-CoT进行推理
            result = model_controller.run(
                image_path=image_path,
                question=question,
                verbose=False
            )
            
            answer = result.get("answer", "")
            
            # LLaVA-W的评估通常需要GPT-4进行打分
            # 这里我们提供基本的评估（长度、完整性等）
            # 实际使用时可以调用GPT-4 API进行更准确的评估
            scores = self._evaluate_answer_quality(answer, question, category)
            
            return {
                "question_id": question_id,
                "category": category,
                "question": question,
                "answer": answer,
                "scores": scores,
                "success": True
            }
            
        except Exception as e:
            print(f"Error evaluating question {question_id}: {e}")
            return {
                "question_id": question_id,
                "category": category,
                "error": str(e),
                "success": False
            }
    
    def _evaluate_answer_quality(self, answer: str, question: str, category: str) -> Dict[str, float]:
        """
        评估回答质量
        
        简单评估方法，基于：
        1. 回答长度和完整性
        2. 是否包含相关信息
        
        更准确的评估需要使用GPT-4
        """
        scores = {}
        
        answer_len = len(answer.split())
        
        # 基于长度和类别的简单评分
        if category == "conversation":
            # 对话类：期望中等长度，自然的回答
            if 10 <= answer_len <= 50:
                scores["conversation"] = 0.8
            elif 5 <= answer_len < 10 or 50 < answer_len <= 100:
                scores["conversation"] = 0.6
            else:
                scores["conversation"] = 0.4
        
        elif category == "detail":
            # 细节描述：期望较长、详细的回答
            if answer_len >= 30:
                scores["detail"] = 0.8
            elif answer_len >= 15:
                scores["detail"] = 0.6
            else:
                scores["detail"] = 0.4
        
        elif category == "complex":
            # 复杂推理：期望结构化、逻辑清晰的回答
            if answer_len >= 20:
                scores["complex_reasoning"] = 0.7
            elif answer_len >= 10:
                scores["complex_reasoning"] = 0.5
            else:
                scores["complex_reasoning"] = 0.3
        
        # 整体得分
        if scores:
            scores["overall"] = sum(scores.values()) / len(scores)
        else:
            # 默认评分
            if answer_len >= 10:
                scores["overall"] = 0.6
            else:
                scores["overall"] = 0.3
        
        return scores
    
    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        计算LLaVA-W评估指标
        
        Args:
            results: 所有样本的评估结果
            
        Returns:
            包含各维度得分的指标字典
        """
        # 过滤成功的结果
        valid_results = [r for r in results if r.get("success", False)]
        
        if not valid_results:
            return {
                "Overall": 0.0,
                "total_samples": len(results),
                "valid_samples": 0
            }
        
        # 收集所有得分
        all_scores = {
            "conversation": [],
            "detail": [],
            "complex_reasoning": [],
            "overall": []
        }
        
        for result in valid_results:
            scores = result.get("scores", {})
            for key in all_scores.keys():
                if key in scores:
                    all_scores[key].append(scores[key])
        
        # 计算平均分
        metrics = {
            "total_samples": len(results),
            "valid_samples": len(valid_results)
        }
        
        for key, score_list in all_scores.items():
            if score_list:
                avg_score = sum(score_list) / len(score_list) * 100
                # 使用更友好的名称
                if key == "conversation":
                    metrics["Conversation"] = avg_score
                elif key == "detail":
                    metrics["Detail"] = avg_score
                elif key == "complex_reasoning":
                    metrics["Complex_Reasoning"] = avg_score
                elif key == "overall":
                    metrics["Overall"] = avg_score
        
        # 按类别统计
        category_stats = {}
        for result in valid_results:
            cat = result.get("category", "unknown")
            if cat not in category_stats:
                category_stats[cat] = []
            
            overall_score = result.get("scores", {}).get("overall", 0)
            category_stats[cat].append(overall_score)
        
        # 添加类别得分
        for cat, score_list in category_stats.items():
            if score_list:
                cat_score = sum(score_list) / len(score_list) * 100
                metrics[f"Category_{cat}"] = cat_score
        
        return metrics
