from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
from utils import LLMDriver, ImageProcessor, parse_json_response
from color_utils import Colors, print_warning, print_success, print_error


class TieredErrorRecovery:

    def __init__(self, 
                 llm_driver: LLMDriver,
                 bbox_expansion_factor: float = 1.3):
        self.llm = llm_driver
        self.image_processor = ImageProcessor()
        self.bbox_expansion_factor = bbox_expansion_factor
    
    def tier1_bbox_expansion(self,
                            image_path: str,
                            candidate_fact: Dict[str, Any],
                            plan_step: str,
                            question: str,
                            verifier) -> Tuple[bool, Optional[Dict[str, Any]]]:
        
        print(f"\n{Colors.YELLOW}   [Tier 1 Recovery] Expanding bounding box...{Colors.RESET}")
        
        try:
            original_bbox = candidate_fact.get('bbox', [])
            if not original_bbox or len(original_bbox) != 4:
                print_warning("    Invalid bbox, cannot expand")
                return False, None
            
            expanded_bbox = self._expand_bbox(original_bbox, self.bbox_expansion_factor)
            
            print(f"    {Colors.DIM}Original bbox: {[f'{x:.3f}' for x in original_bbox]}{Colors.RESET}")
            print(f"    {Colors.DIM}Expanded bbox: {[f'{x:.3f}' for x in expanded_bbox]} (α={self.bbox_expansion_factor}){Colors.RESET}")
            
            expanded_fact = candidate_fact.copy()
            expanded_fact['bbox'] = expanded_bbox
            expanded_fact['_recovery_note'] = f"Tier1: bbox expanded by {self.bbox_expansion_factor}x"
            
            print(f"    {Colors.YELLOW}    → Re-verifying with expanded region...{Colors.RESET}")
            verification_result = verifier.run(
                image_path,
                expanded_fact,
                plan_step,
                question
            )
            
            if verification_result == "Yes":
                print_success("    Tier 1 Recovery successful - expanded bbox verified!")
                return True, expanded_fact
            else:
                print_warning("    Tier 1 Recovery failed - even expanded bbox not verified")
                return False, None
                
        except Exception as e:
            print_error(f"    Tier 1 Error: {e}")
            return False, None
    
    def tier2_negative_constrained_perception(self,
                                             image_path: str,
                                             plan_step: str,
                                             verified_facts: Dict[str, Any],
                                             rejected_regions: List[List[float]],
                                             perceiver) -> Optional[Dict[str, Any]]:
        
        print(f"\n{Colors.YELLOW}[Tier 2 Recovery] Negative-constrained perception...{Colors.RESET}")
        print(f"    {Colors.DIM}Excluding {len(rejected_regions)} rejected region(s){Colors.RESET}")
        
        try:
            negative_constraint_prompt = self._create_negative_constraint_prompt(
                plan_step,
                verified_facts,
                rejected_regions
            )
            
            print(f"    {Colors.YELLOW}    → Calling LLM with negative constraints...{Colors.RESET}")
            response = self.llm.call_llm(
                image=image_path,
                text_prompt=negative_constraint_prompt,
                max_tokens=512,
                temperature=0.2 
            )
            
            candidate_fact = parse_json_response(response)
            
            if candidate_fact and candidate_fact.get('bbox'):
                print_success(f"   Found alternative candidate: {candidate_fact.get('name', 'Unknown')}")
                candidate_fact['_recovery_note'] = f"Tier2: Alternative candidate (excluded {len(rejected_regions)} regions)"
                return candidate_fact
            else:
                print_warning("    Tier 2 failed - no alternative candidate found")
                return None
                
        except Exception as e:
            print_error(f"    Tier 2 Error: {e}")
            return None
    
    def _expand_bbox(self, bbox: List[float], factor: float) -> List[float]:
        
        if len(bbox) != 4:
            return bbox
        
        y1, x1, y2, x2 = bbox
        
        center_y = (y1 + y2) / 2
        center_x = (x1 + x2) / 2
        height = y2 - y1
        width = x2 - x1
        
        new_height = height * factor
        new_width = width * factor
        
        new_y1 = max(0.0, center_y - new_height / 2)
        new_x1 = max(0.0, center_x - new_width / 2)
        new_y2 = min(1.0, center_y + new_height / 2)
        new_x2 = min(1.0, center_x + new_width / 2)
        
        return [new_y1, new_x1, new_y2, new_x2]
    
    def _create_negative_constraint_prompt(self,
                                          plan_step: str,
                                          verified_facts: Dict[str, Any],
                                          rejected_regions: List[List[float]]) -> str:
        
        if verified_facts:
            facts_json = "\n".join([
                f"  {fact_id}: {{\"name\": \"{data.get('name', 'Unknown')}\", \"bbox\": {data.get('bbox', [])}}}"
                for fact_id, data in verified_facts.items()
            ])
            facts_context = f"Verified Fact Graph:\n{{{facts_json}\n}}"
        else:
            facts_context = "Verified Fact Graph: {} (empty)"
        
        rejected_str = "Previously REJECTED regions (DO NOT re-detect objects in these areas):\n"
        for i, bbox in enumerate(rejected_regions, 1):
            rejected_str += f"  Rejected #{i}: bbox {bbox}\n"
        
        prompt = f"""# Role
You are a Context-Aware Visual Grounding Expert with NEGATIVE CONSTRAINTS.

# Task
{plan_step}

# Context
{facts_context}

# CRITICAL NEGATIVE CONSTRAINTS
{rejected_str}

**IMPORTANT**: You MUST find an ALTERNATIVE candidate that:
1. Satisfies the task requirements
2. Is located in a DIFFERENT spatial region from the rejected areas
3. Has NOT been previously detected and rejected

# Strategy
- If looking for object type X, find a DIFFERENT instance of X (if multiple exist)
- If the rejected regions are near anchor object A, look further away or on opposite side
- Consider that the previous detection might have been a false positive - look more carefully

# Output Format
Return a complete JSON node:
{{
  "id": "unique_id",
  "name": "object description",
  "bbox": [ymin, xmin, ymax, xmax] (0-1 scale, MUST be different from rejected regions),
  "attributes": ["attribute1", "attribute2"],
  "relationships": {{"target_id": "fact_N", "type": "relation_type"}} (if applicable)
}}

If you truly cannot find ANY alternative candidate:
{{
  "name": "not_found",
  "bbox": null,
  "reason": "explain why no alternative exists"
}}

**Output ONLY valid JSON.**"""
        
        return prompt
    
    def check_bbox_overlap(self, bbox1: List[float], bbox2: List[float], threshold: float = 0.5) -> bool:
    
        if len(bbox1) != 4 or len(bbox2) != 4:
            return False
        
        y1_max = max(bbox1[0], bbox2[0])
        x1_max = max(bbox1[1], bbox2[1])
        y2_min = min(bbox1[2], bbox2[2])
        x2_min = min(bbox1[3], bbox2[3])
        
        if y2_min <= y1_max or x2_min <= x1_max:
            return False  
        
        intersection = (y2_min - y1_max) * (x2_min - x1_max)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        iou = intersection / union if union > 0 else 0
        
        return iou > threshold


class ErrorRecoveryManager:
    
    def __init__(self,
                 llm_driver: LLMDriver,
                 perceiver,
                 verifier,
                 reflector_replanner,
                 bbox_expansion_factor: float = 1.3,
                 max_tier2_attempts: int = 2):
        
        self.recovery = TieredErrorRecovery(llm_driver, bbox_expansion_factor)
        self.perceiver = perceiver
        self.verifier = verifier
        self.reflector_replanner = reflector_replanner
        self.max_tier2_attempts = max_tier2_attempts
    
    def handle_verification_failure(self,
                                   image_path: str,
                                   candidate_fact: Dict[str, Any],
                                   plan_step: str,
                                   question: str,
                                   verified_facts: Dict[str, Any],
                                   rejected_regions: List[List[float]]) -> Tuple[str, Optional[Dict[str, Any]]]:
        
        success, expanded_fact = self.recovery.tier1_bbox_expansion(
            image_path,
            candidate_fact,
            plan_step,
            question,
            self.verifier
        )
        
        if success:
            return ("tier1_bbox_expansion", expanded_fact)
        
        if candidate_fact.get('bbox'):
            rejected_regions.append(candidate_fact['bbox'])
        
        for attempt in range(self.max_tier2_attempts):
            print(f"\n    {Colors.YELLOW}Tier 2 Attempt {attempt + 1}/{self.max_tier2_attempts}{Colors.RESET}")
            
            alternative_fact = self.recovery.tier2_negative_constrained_perception(
                image_path,
                plan_step,
                verified_facts,
                rejected_regions,
                self.perceiver
            )
            
            if alternative_fact is None:
                continue
            
            print(f"    {Colors.YELLOW}    → Verifying alternative candidate...{Colors.RESET}")
            verification_result = self.verifier.run(
                image_path,
                alternative_fact,
                plan_step,
                question
            )
            
            if verification_result == "Yes":
                print_success(f"    Tier 2 Recovery successful on attempt {attempt + 1}!")
                return ("tier2_negative_constraint", alternative_fact)
            else:
                if alternative_fact.get('bbox'):
                    rejected_regions.append(alternative_fact['bbox'])
        
        print(f"\n    {Colors.BRIGHT_RED}Tier 1 & Tier 2 failed - triggering Tier 3 (Reflection){Colors.RESET}")
        return ("failed_need_tier3", None)
