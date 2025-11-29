
import json
from typing import Dict, List, Any, Optional
from PIL import Image
from utils import LLMDriver, ImageProcessor, parse_json_response, format_verified_facts
from color_utils import (
    Colors, print_colored, print_llm_response, print_verification_result,
    print_success, print_error, print_warning, print_fact
)


class Planner:
    def __init__(self, llm_driver: LLMDriver):
        self.llm = llm_driver
    
    def run(self, image_path: str, question: str) -> List[str]:
        prompt = self._create_planning_prompt(question)
        
        try:
            response = self.llm.call_llm(
                image=image_path,
                text_prompt=prompt,
                max_tokens=1024,
                temperature=0
            )
            
            
            plan = self._parse_plan(response)
            
            print_success(f"Plan generated with {len(plan)} steps:")
            for i, step in enumerate(plan, 1):
                print(f"        {Colors.CYAN}{i}.{Colors.RESET} {Colors.WHITE}{step}{Colors.RESET}")
            
            return plan
        except Exception as e:
            print_error(f"Planner Error: {e}")
            return [f"Analyze the image to answer: {question}"]
    
    def _create_planning_prompt(self, question: str) -> str:
        raise NotImplementedError(
            "Core planning algorithm is proprietary. "
            "Contact the author for licensing or implement your own planning strategy. "
            "Hint: Consider spatial dependency and anchor-to-detail decomposition."
        )
    
    def _parse_plan(self, response: str) -> List[str]:
        lines = response.strip().split('\n')
        plan = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line[0].isdigit() or line.lower().startswith('step'):
                import re
                cleaned = re.sub(r'^(\d+\.?\s*|Step\s+\d+:?\s*)', '', line, flags=re.IGNORECASE)
                if cleaned:
                    plan.append(cleaned.strip())
        
        if not plan:
            plan = [response.strip()]
        
        if len(plan) > 3:
            plan = plan[:3]
        
        return plan


class Perceiver:
    def __init__(self, llm_driver: LLMDriver):
        self.llm = llm_driver
    
    def run(self, 
            image_path: str,
            plan_step: str,
            verified_facts: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        prompt = self._create_perception_prompt(plan_step, verified_facts)
        
        try:
            response = self.llm.call_llm(
                image=image_path,
                text_prompt=prompt,
                max_tokens=512,
                temperature=0
            )
            
            
            print(f"{Colors.YELLOW}        → [Perceiver] Parsing perception result...{Colors.RESET}")
            candidate_fact = self._parse_perception_result(response)
            
            if candidate_fact:
                print_success(f"Identified: {candidate_fact.get('name', 'Unknown')}")
                if 'bbox' in candidate_fact:
                    bbox = candidate_fact['bbox']
                    print(f"        {Colors.DIM}BBox: [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]{Colors.RESET}")
                if 'relation' in candidate_fact and candidate_fact['relation']:
                    rel = candidate_fact['relation']
                    print(f"        {Colors.CYAN}Relation: {rel.get('type', 'N/A')} → {rel.get('target', 'N/A')}{Colors.RESET}")
            else:
                print_warning("Perceiver returned no valid result")
            
            return candidate_fact
        except Exception as e:
            print_error(f"Perceiver Error: {e}")
            return None
    
    def _create_perception_prompt(self, plan_step: str, verified_facts: Dict[str, Any]) -> str:
        if verified_facts:
            facts_json = "\n".join([
                f"  {fact_id}: {{\"name\": \"{data.get('name', 'Unknown')}\", \"bbox\": {data.get('bbox', [])}}}"
                for fact_id, data in verified_facts.items()
            ])
            facts_context = f"Verified Facts:\n{{{facts_json}\n}}"
        else:
            facts_context = "Verified Facts: {{}} (empty)"
        
        prompt = f"""You are a visual grounding expert. Locate objects and expand the scene graph.

Current Task: {plan_step}
{facts_context}

Output Format (JSON):
{{
  "id": "<object_type>_<number>",
  "name": "<object_description>",
  "bbox": [ymin, xmin, ymax, xmax],
  "attributes": ["<attr1>", "<attr2>"],
  "relationships": {{"target_id": "<fact_id>", "type": "<relation_type>"}}
}}

If not found: {{"name": "not_found", "bbox": null}}
Output ONLY valid JSON."""
        return prompt
    
    def _parse_perception_result(self, response: str) -> Optional[Dict[str, Any]]:
        result = parse_json_response(response)
        
        if result is None:
            return None
        
        if 'name' not in result:
            return None
        
        if result.get('name') == 'not_found' or result.get('bbox') is None:
            return None
        
        bbox = result.get('bbox')
        if not isinstance(bbox, list) or len(bbox) != 4:
            return None
        
        return result


class Verifier:
    def __init__(self, llm_driver: LLMDriver):
        self.llm = llm_driver
        self.image_processor = ImageProcessor()
    
    def run(self,
            image_path: str,
            candidate_fact: Dict[str, Any],
            plan_step: str,
            question: str) -> str:
        try:
            sub_image = self.image_processor.crop_image(image_path, candidate_fact['bbox'])
            
            prompt = self._create_verification_prompt(candidate_fact, plan_step, question)
            
            response = self.llm.call_llm(
                image_pil=sub_image,
                text_prompt=prompt,
                max_tokens=256,
                temperature=0
            )
            
            print_llm_response("Verifier Response", response, max_length=200)
            
            print(f"{Colors.MAGENTA}        → [Verifier] Parsing verification result...{Colors.RESET}")
            result = self._parse_verification_result(response)
            
            print_verification_result(result, f"Fact: {candidate_fact.get('name', 'Unknown')}")
            
            return result
        except Exception as e:
            print_error(f"Verifier Error: {e}")
            return "No"
    
    def _create_verification_prompt(self,
                                   candidate_fact: Dict[str, Any],
                                   plan_step: str,
                                   question: str) -> str:
        fact_name = candidate_fact.get('name', 'Unknown')
        attributes = candidate_fact.get('attributes', [])
        relation = candidate_fact.get('relationships', {})
        
        attrs_str = ", ".join(attributes) if attributes else "none"
        rel_str = f"{relation.get('type', '')} {relation.get('target_id', '')}" if relation.get('type') else "none"
        
        prompt = f"""Verify if this cropped region matches:
Object: {fact_name}
Attributes: {attrs_str}
Relationship: {rel_str}

Question: {question}
Task: {plan_step}

Verify:
1. Object identity correct?
2. All attributes present?
3. Relationship valid?

Answer "Yes" ONLY if ALL verified. Otherwise "No".
Your answer:"""
        return prompt
    
    def _parse_verification_result(self, response: str) -> str:
        response_lower = response.strip().lower()
        
        if 'yes' in response_lower[:10]:
            return "Yes"
        else:
            return "No"


class ReflectorRePlanner:
    def __init__(self, llm_driver: LLMDriver):
        self.llm = llm_driver
    
    def run(self,
            question: str,
            original_plan: List[str],
            execution_history: List[Dict[str, Any]],
            image_path: str) -> List[str]:
        prompt = self._create_reflection_prompt(question, original_plan, execution_history)
        
        for i, entry in enumerate(execution_history):
            if entry.get('status', '').startswith('Failed'):
                step = entry.get('plan_step', 'Unknown')
                print(f"    {Colors.RED}✗ Step {i+1}:{Colors.RESET} {Colors.DIM}{step}{Colors.RESET}")
        
        try:
            print(f"\n{Colors.BRIGHT_RED}    → [Reflector] Calling LLM API for revised plan...{Colors.RESET}")
            response = self.llm.call_llm(
                image=image_path,
                text_prompt=prompt,
                max_tokens=1024,
                temperature=0.3   
            )
            
            print_llm_response("Reflector Analysis & Revised Plan", response, max_length=500)
            
            new_plan = self._parse_new_plan(response)
            
            print_success(f"Revised plan generated with {len(new_plan)} steps:")
            for i, step in enumerate(new_plan, 1):
                print(f"        {Colors.CYAN}{i}.{Colors.RESET} {Colors.WHITE}{step}{Colors.RESET}")
            
            return new_plan
        except Exception as e:
            print_error(f"Reflector Error: {e}")
            return [f"1. Carefully re-examine the image to answer: {question}"]
    
    def _create_reflection_prompt(self,
                                 question: str,
                                 original_plan: List[str],
                                 execution_history: List[Dict[str, Any]]) -> str:
        plan_str = "\n".join([f"{i+1}. {step}" for i, step in enumerate(original_plan)])
        
        history_str = "Failed steps:\n"
        for i, entry in enumerate(execution_history):
            if 'Failed' in entry.get('status', ''):
                history_str += f"{i+1}. {entry.get('plan_step', 'Unknown')}\n"
        
        prompt = f"""Analyze why the plan failed and create a new one.

Question: {question}
Original Plan:
{plan_str}

{history_str}

Categories:
A. Object not in image
B. Wrong search region
C. Attribute mismatch
D. Relationship ambiguity
E. Dependency chain broken

Output JSON:
{{
  "root_cause": "<category>",
  "diagnosis": "<explanation>",
  "new_plan": ["<step1>", "<step2>", ...]
}}

Provide diagnostic JSON:"""
        return prompt
    
    def _parse_new_plan(self, response: str) -> List[str]:
        result = parse_json_response(response)
        
        if result and 'new_plan' in result:
            plan = result['new_plan']
            if isinstance(plan, list) and len(plan) > 0:
                cleaned_plan = []
                for step in plan:
                    if isinstance(step, str):
                        import re
                        cleaned = re.sub(r'^(\d+\.?\s*|Step\s+\d+:?\s*)', '', step, flags=re.IGNORECASE)
                        if cleaned.strip():
                            cleaned_plan.append(cleaned.strip())
                if cleaned_plan:
                    return cleaned_plan
        
        lines = response.strip().split('\n')
        plan = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line[0].isdigit() or line.lower().startswith('step'):
                import re
                cleaned = re.sub(r'^(\d+\.?\s*|Step\s+\d+:?\s*)', '', line, flags=re.IGNORECASE)
                if cleaned:
                    plan.append(cleaned.strip())
        
        if not plan:
            plan = [response.strip()]
        
        return plan


class Synthesizer:
    def __init__(self, llm_driver: LLMDriver):
        self.llm = llm_driver
    
    def run(self,
            question: str,
            verified_facts: Dict[str, Any],
            image_path: str) -> str:
        prompt = self._create_synthesis_prompt(question, verified_facts)
        
        if verified_facts:
            for fact_id, fact_data in verified_facts.items():
                print_fact(f"    {fact_id}", fact_data)
        else:
            print_warning("    No verified facts available")
        
        try:
            response = self.llm.call_llm(
                image=image_path,
                text_prompt=prompt,
                max_tokens=512,
                temperature=0
            )
            
            print_llm_response("Synthesizer Final Answer", response, max_length=400)
            
            return response.strip()
        except Exception as e:
            print_error(f"Synthesizer Error: {e}")
            return "Unable to generate answer due to error."
    
    def _create_synthesis_prompt(self,
                                question: str,
                                verified_facts: Dict[str, Any]) -> str:
        if verified_facts:
            facts_json = []
            for fact_id, fact_data in verified_facts.items():
                facts_json.append({
                    "id": fact_id,
                    "name": fact_data.get('name', 'Unknown'),
                    "attributes": fact_data.get('attributes', [])
                })
            scene_graph_str = json.dumps(facts_json, indent=2)
        else:
            scene_graph_str = "[]"
        
        prompt = f"""Answer this question based on the verified scene graph.

Question: {question}

Scene Graph (JSON):
{scene_graph_str}

Rules:
- Answer ONLY from the scene graph
- For Yes/No: Start with Yes/No then justify
- If uncertain: State "Cannot determine based on available evidence"
- DO NOT guess or use external knowledge

Your answer:"""
        return prompt
