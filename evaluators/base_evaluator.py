"""
Base Evaluator Class
所有数据集评估器的基类
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import json
import os
from datetime import datetime


class BaseEvaluator(ABC):
    """数据集评估器基类"""
    
    def __init__(self, dataset_name: str, dataset_path: str):
        """
        初始化评估器
        
        Args:
            dataset_name: 数据集名称
            dataset_path: 数据集路径
        """
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.results = []
    
    @abstractmethod
    def load_dataset(self) -> List[Dict[str, Any]]:
        """
        加载数据集
        
        Returns:
            数据样本列表
        """
        pass
    
    @abstractmethod
    def evaluate_sample(self, sample: Dict[str, Any], model_controller) -> Dict[str, Any]:
        """
        评估单个样本
        
        Args:
            sample: 数据样本
            model_controller: 模型控制器（如VCoTController）
            
        Returns:
            评估结果字典
        """
        pass
    
    @abstractmethod
    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            results: 所有样本的评估结果
            
        Returns:
            指标字典
        """
        pass
    
    def run_evaluation(self, model_controller, save_path: Optional[str] = None, 
                      verbose: bool = True, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        运行完整评估流程
        
        Args:
            model_controller: 模型控制器
            save_path: 结果保存路径
            verbose: 是否打印详细信息
            max_samples: 最大评估样本数（用于测试）
            
        Returns:
            包含指标和详细结果的字典
        """
        if verbose:
            print(f"{'='*80}")
            print(f"Starting evaluation on {self.dataset_name}")
            print(f"{'='*80}\n")
        
        # 加载数据集
        if verbose:
            print("Loading dataset...")
        dataset = self.load_dataset()
        
        if max_samples:
            dataset = dataset[:max_samples]
        
        if verbose:
            print(f"Loaded {len(dataset)} samples\n")
        
        # 评估每个样本
        self.results = []
        for i, sample in enumerate(dataset):
            print("当前是第", i+1, "个样本")
            print("样本信息：", sample)
            if verbose:
                print(f"Evaluating sample {i+1}/{len(dataset)}...")
            
            try:
                # lxz，这里是评估单个样本的入口
                result = self.evaluate_sample(sample, model_controller)
                self.results.append(result)
                
                if verbose:
                    print(f"  Sample {i+1} completed\n")
            except Exception as e:
                if verbose:
                    print(f"  Error in sample {i+1}: {e}\n")
                self.results.append({
                    "sample_id": sample.get("id", i),
                    "error": str(e),
                    "success": False
                })
        
        # 计算指标
        if verbose:
            print("Computing metrics...")
        metrics = self.compute_metrics(self.results)
        
        # 构建最终结果
        final_results = {
            "dataset": self.dataset_name,
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(dataset),
            "completed_samples": sum(1 for r in self.results if r.get("success", False)),
            "metrics": metrics,
            "detailed_results": self.results
        }
        
        # 保存结果
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)
            if verbose:
                print(f"\nResults saved to: {save_path}")
        
        # 打印指标
        if verbose:
            print(f"\n{'='*80}")
            print("Evaluation Results")
            print(f"{'='*80}")
            print(f"Dataset: {self.dataset_name}")
            print(f"Total Samples: {final_results['total_samples']}")
            print(f"Completed: {final_results['completed_samples']}")
            print(f"\nMetrics:")
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, float):
                    print(f"  {metric_name}: {metric_value:.4f}")
                else:
                    print(f"  {metric_name}: {metric_value}")
            print(f"{'='*80}\n")
        
        return final_results
