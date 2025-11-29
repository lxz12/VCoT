import os
import argparse
import json
from typing import Optional, Dict, Any
from datetime import datetime

from v_cot_main import VCoTController
from evaluators import (
    WinogroundEvaluator,
    WinogroundEvaluatorV2,
    WinogroundEvaluatorV3,
    WHOOPSEvaluator,
    SEEDBenchEvaluator,
    MMBenchEvaluator,
    LLaVAWEvaluator
)


EVALUATOR_MAP = {
    "winoground": WinogroundEvaluatorV3,  
    "winoground-v2": WinogroundEvaluatorV2,  
    "winoground-v1": WinogroundEvaluator,
    "whoops": WHOOPSEvaluator,
    "seed-bench": SEEDBenchEvaluator,
    "mmbench": MMBenchEvaluator,
    "llava-w": LLaVAWEvaluator,
}


def setup_model(args) -> VCoTController:
    controller = VCoTController(
        model_type=args.model,
        api_key=args.api_key,
        max_retries=args.max_retries,
        max_replan_attempts=args.max_replan
    )
    return controller


def run_evaluation(dataset_name: str,
                   dataset_path: str,
                   model_controller: VCoTController,
                   output_dir: str,
                   max_samples: Optional[int] = None,
                   verbose: bool = True,
                   hf_token: Optional[str] = None) -> Dict[str, Any]:
    
    if dataset_name not in EVALUATOR_MAP:
        raise ValueError(f"Unsupported dataset: {dataset_name}. "
                        f"Supported datasets: {list(EVALUATOR_MAP.keys())}")
    
    evaluator_class = EVALUATOR_MAP[dataset_name]
    
    if dataset_name in ["winoground", "whoops"]:
        evaluator = evaluator_class(dataset_path, hf_token=hf_token)
    else:
        evaluator = evaluator_class(dataset_path)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{dataset_name}_results_{timestamp}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    results = evaluator.run_evaluation(
        model_controller=model_controller,
        save_path=output_path,
        verbose=verbose,
        max_samples=max_samples
    )
    
    return results


def run_multi_dataset_evaluation(dataset_configs: Dict[str, str],
                                 model_controller: VCoTController,
                                 output_dir: str,
                                 max_samples: Optional[int] = None,
                                 verbose: bool = True,
                                 hf_token: Optional[str] = None) -> Dict[str, Any]:
    
    all_results = {}
    
    for dataset_name, dataset_path in dataset_configs.items():
        if verbose:
            print(f"\n{'='*80}")
            print(f"Starting evaluation for: {dataset_name}")
            print(f"{'='*80}\n")
        
        try:
            results = run_evaluation(
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                model_controller=model_controller,
                output_dir=output_dir,
                max_samples=max_samples,
                verbose=verbose,
                hf_token=hf_token
            )
            all_results[dataset_name] = results
        except Exception as e:
            if verbose:
                print(f"Error evaluating {dataset_name}: {e}\n")
            all_results[dataset_name] = {
                "error": str(e),
                "success": False
            }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(output_dir, f"summary_{timestamp}.json")
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "datasets": list(dataset_configs.keys()),
        "results": {}
    }
    
    for dataset_name, results in all_results.items():
        if results.get("success", True) and "metrics" in results:
            summary["results"][dataset_name] = results["metrics"]
        else:
            summary["results"][dataset_name] = {"error": results.get("error", "Unknown error")}
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"\n{'='*80}")
        print("Multi-Dataset Evaluation Summary")
        print(f"{'='*80}")
        for dataset_name, metrics in summary["results"].items():
            print(f"\n{dataset_name}:")
            if "error" not in metrics:
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, float):
                        print(f"  {metric_name}: {metric_value:.4f}")
                    else:
                        print(f"  {metric_name}: {metric_value}")
            else:
                print(f"  Error: {metrics['error']}")
        print(f"\nSummary saved to: {summary_path}")
        print(f"{'='*80}\n")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="V-CoT Benchmark Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
        """
    )
    
    parser.add_argument("--dataset", type=str, 
                       choices=list(EVALUATOR_MAP.keys()),
                       help="Dataset to evaluate")
    parser.add_argument("--dataset-path", type=str,
                       help="Path to the dataset")
    parser.add_argument("--config", type=str,
                       help="Path to config file for multi-dataset evaluation")
    
    parser.add_argument("--model", type=str, default="gpt-4-vision-preview",
                       help="Model type to use")
    parser.add_argument("--api-key", type=str, default=None,
                       help="API key (optional if set in env)")
    parser.add_argument("--max-retries", type=int, default=2,
                       help="Max retries per step")
    parser.add_argument("--max-replan", type=int, default=1,
                       help="Max replanning attempts")
    parser.add_argument("--hf-token", type=str, default=None,
                       help="Hugging Face token for loading datasets (optional if set in HF_TOKEN env)")
    
    parser.add_argument("--output-dir", type=str, default="./eval_results",
                       help="Output directory for results")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to evaluate (for testing)")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress verbose output")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    verbose = not args.quiet
    if verbose:
        print("Initializing V-CoT model...")
    model_controller = setup_model(args)
    
    if args.config:
        if verbose:
            print(f"Loading config from: {args.config}")
        
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        dataset_configs = config.get("datasets", {})
        
        if not dataset_configs:
            print("Error: No datasets found in config file")
            return
        
        run_multi_dataset_evaluation(
            dataset_configs=dataset_configs,
            model_controller=model_controller,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            verbose=verbose,
            hf_token=args.hf_token
        )
    
    elif args.dataset and args.dataset_path:
        run_evaluation(
            dataset_name=args.dataset,
            dataset_path=args.dataset_path,
            model_controller=model_controller,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            verbose=verbose,
            hf_token=args.hf_token
        )
    
    else:
        print("Error: Must specify either --dataset and --dataset-path, or --config")
        parser.print_help()
        return


if __name__ == "__main__":
    main()
