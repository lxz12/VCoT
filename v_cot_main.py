
import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from utils import LLMDriver, save_results
from modules import Planner, Perceiver, Verifier, ReflectorRePlanner, Synthesizer
from color_utils import Colors, print_section, print_step, print_success, print_error, print_warning


class VCoTController:
    def __init__(self,
                 model_type: str = "gpt-4-vision-preview",
                 api_key: Optional[str] = None,
                 max_retries: int = 2,
                 max_replan_attempts: int = 1):
        self.llm_driver = LLMDriver(model_type=model_type, api_key=api_key)
        self.max_retries = max_retries
        self.max_replan_attempts = max_replan_attempts
        self.planner = Planner(self.llm_driver)
        self.perceiver = Perceiver(self.llm_driver)
        self.verifier = Verifier(self.llm_driver)
        self.reflector_replanner = ReflectorRePlanner(self.llm_driver)
        self.synthesizer = Synthesizer(self.llm_driver)
        self.plan: List[str] = []
        self.verified_fact_graph: Dict[str, Any] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.current_step_index: int = 0
        self.replan_count: int = 0
    
    def run(self, image_path: str, question: str, verbose: bool = True) -> Dict[str, Any]:
        if verbose:
            print_section("V-CoT System Starting")
            print(f"{Colors.CYAN}Image:{Colors.RESET} {Colors.WHITE}{image_path}{Colors.RESET}")
            print(f"{Colors.CYAN}Question:{Colors.RESET} {Colors.WHITE}{question}{Colors.RESET}")
        self._initialize()
        if verbose:
            print_section("STEP 1: Planning", "─")
        
        import time
        plan_start = time.time()
        self.plan = self.planner.run(image_path, question)
        plan_elapsed = time.time() - plan_start
        
        if verbose:
            print_success(f"Plan generated in {plan_elapsed:.1f}s with {len(self.plan)} steps")
        
        if verbose:
            print_section("STEP 2: Executing Plan with Verification", "─")
        
        while self.current_step_index < len(self.plan):
            success = self._execute_single_step(
                image_path, 
                question, 
                self.current_step_index,
                verbose
            )
            
            if success:
                self.current_step_index += 1
            else:
                if self.replan_count < self.max_replan_attempts:
                    if verbose:
                        print(f"\n[REFLECTION] Step {self.current_step_index + 1} failed after retries.")
                        print("Initiating reflection and re-planning...")
                    
                    new_plan = self.reflector_replanner.run(
                        question,
                        self.plan,
                        self.execution_history,
                        image_path
                    )
                    
                    if verbose:
                        print(f"New plan generated with {len(new_plan)} steps:")
                        for i, step in enumerate(new_plan, 1):
                            print(f"  {i}. {step}")
                    
                    self.plan = new_plan
                    self.current_step_index = 0
                    self.replan_count += 1
                else:
                    if verbose:
                        print("\n[ERROR] Maximum replanning attempts reached. Cannot answer the question reliably.")
                    
                    return self._create_result(
                        question=question,
                        image_path=image_path,
                        final_answer="Could not reliably answer the question after multiple attempts.",
                        success=False
                    )
        
        if verbose:
            print_section("STEP 3: Synthesizing Final Answer", "─")
        
        synth_start = time.time()
        final_answer = self.synthesizer.run(
            question,
            self.verified_fact_graph,
            image_path
        )
        synth_elapsed = time.time() - synth_start
        
        if verbose:
            print_success(f"Synthesis completed in {synth_elapsed:.1f}s")
            print(f"\n{Colors.BRIGHT_GREEN}{Colors.BOLD}{'='*80}{Colors.RESET}")
            print(f"{Colors.BRIGHT_GREEN}{Colors.BOLD}FINAL ANSWER:{Colors.RESET}")
            print(f"{Colors.WHITE}{final_answer}{Colors.RESET}")
            print(f"{Colors.BRIGHT_GREEN}{Colors.BOLD}{'='*80}{Colors.RESET}\n")
        
        return self._create_result(
            question=question,
            image_path=image_path,
            final_answer=final_answer,
            success=True
        )
    
    def _initialize(self):
        self.plan = []
        self.verified_fact_graph = {}
        self.execution_history = []
        self.current_step_index = 0
        self.replan_count = 0
    
    def _execute_single_step(self,
                            image_path: str,
                            question: str,
                            step_index: int,
                            verbose: bool) -> bool:
        
        current_plan_step = self.plan[step_index]
        
        if verbose:
            print(f"\n{Colors.BRIGHT_CYAN}{'='*80}{Colors.RESET}")
            print_step(step_index + 1, len(self.plan), current_plan_step)
            print(f"{Colors.BRIGHT_CYAN}{'='*80}{Colors.RESET}")
        
        for retry_count in range(self.max_retries):
            if verbose and retry_count > 0:
                print_warning(f"    Retry attempt {retry_count + 1}/{self.max_retries}")
            
            import time
            start_time = time.time()
            candidate_fact = self.perceiver.run(
                image_path,
                current_plan_step,
                self.verified_fact_graph
            )
            elapsed = time.time() - start_time
            
            if candidate_fact is None:
                if verbose:
                    print_error(f"    Perception failed (took {elapsed:.1f}s)")
                
                self._log_execution(
                    step_index=step_index,
                    plan_step=current_plan_step,
                    perception_output=None,
                    verification_result="N/A",
                    status="Failed - Perception Error"
                )
                continue
            
            verify_start = time.time()
            verification_result = self.verifier.run(
                image_path,
                candidate_fact,
                current_plan_step,
                question
            )
            verify_elapsed = time.time() - verify_start
            
            if verification_result == "Yes":
                fact_id = f"fact_{len(self.verified_fact_graph) + 1}"
                self.verified_fact_graph[fact_id] = {
                    **candidate_fact,
                    "status": "verified",
                    "source_step": step_index + 1
                }
                
                self._log_execution(
                    step_index=step_index,
                    plan_step=current_plan_step,
                    perception_output=candidate_fact,
                    verification_result="Yes",
                    status="Success"
                )
                
                if verbose:
                    print_success(f"Step {step_index + 1} completed successfully (total: {elapsed + verify_elapsed:.1f}s)")
                
                return True
            else:
                if verbose:
                    print(f"      Reason: {reason}")
                
                self._log_execution(
                    step_index=step_index,
                    plan_step=current_plan_step,
                    perception_output=candidate_fact,
                    verification_result="No",
                    status="Failed - Verification"
                )
        
        if verbose:
            print(f"    Step {step_index + 1} failed after {self.max_retries + 1} attempts]")
        
        return False
    
    def _log_execution(self,
                      step_index: int,
                      plan_step: str,
                      perception_output: Optional[Dict[str, Any]],
                      verification_result: str,
                      status: str):
        log_entry = {
            "step_index": step_index,
            "plan_step": plan_step,
            "perception_output": perception_output,
            "verification_result": verification_result,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        self.execution_history.append(log_entry)
    
    def _create_result(self,
                      question: str,
                      image_path: str,
                      final_answer: str,
                      success: bool) -> Dict[str, Any]:
        return {
            "question": question,
            "image": image_path,
            "answer": final_answer,
            "success": success,
            "plan": self.plan,
            "verified_facts": self.verified_fact_graph,
            "execution_history": self.execution_history,
            "stats": {
                "total_steps": len(self.plan),
                "completed_steps": self.current_step_index,
                "verified_facts_count": len(self.verified_fact_graph),
                "replan_count": self.replan_count,
                "total_attempts": len(self.execution_history)
            }
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="V-CoT Visual Reasoning System")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--question", type=str, required=True, help="Question about the image")
    parser.add_argument("--model", type=str, default="gpt-4-vision-preview", help="Model type to use")
    parser.add_argument("--api-key", type=str, default=None, help="API key (optional if set in env)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--max-retries", type=int, default=2, help="Max retries per step")
    parser.add_argument("--max-replan", type=int, default=1, help="Max replanning attempts")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    
    args = parser.parse_args()
    
    print(args.image)
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return

    controller = VCoTController(
        model_type=args.model,
        api_key=args.api_key,
        max_retries=args.max_retries,
        max_replan_attempts=args.max_replan
    )
    
    result = controller.run(
        image_path=args.image,
        question=args.question,
        verbose=not args.quiet
    )
    
    if args.output:
        save_results(args.output, result)
        print(f"\nResults saved to: {args.output}")
    
    return result


if __name__ == "__main__":
    main()
