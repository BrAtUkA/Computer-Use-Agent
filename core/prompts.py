# prompt templates

import json
from datetime import datetime
from typing import Any, List
from PIL import Image
from pydantic import BaseModel, Field

from .actions import NavigationStep, ValidationResult


class ClickCoordinates(BaseModel):
    x: int = Field(ge=0, le=1000, description="The x coordinate, normalized between 0 and 1000.")
    y: int = Field(ge=0, le=1000, description="The y coordinate, normalized between 0 and 1000.")


def get_localization_prompt(pil_image: Image.Image, instruction: str) -> List[dict[str, Any]]:
    prompt = f"""Localize an element on the GUI image according to the provided target and output a click position.
     * You must output a valid JSON following the format: {ClickCoordinates.model_json_schema()}
     Your target is:"""
    
    return [{"role": "user", "content": [
        {"type": "image", "image": pil_image}, 
        {"type": "text", "text": f"{prompt}\n{instruction}"}
    ]}]


NAVIGATION_SYSTEM_PROMPT = """You are a robot controlling a computer. You will be given a task and an observation of the screen.
In each step, you must analyze the screen to determine the next action towards completing the task.
Detail your reasoning in the 'thought' field. If you find relevant information, add it to the 'note' field.
Once you have all information to answer the task, use the 'answer' action.

Guidelines:
- If you need to type in a search bar or form and submit, use the `type_and_enter` action. This will type and then press Enter.
- For other cases of filling forms, you may need to explicitly click a 'submit' or 'login' button after typing.
- Always accept cookies if a notice is present.
- Don't scroll more than 3 times unless necessary.
- The current date is {timestamp}.

# <output_json_format>
# ```json
# {output_format}
# ```
# </output_json_format>
"""


def get_navigation_prompt(task: str, image: Image.Image, history: List[str], step: int) -> List[dict[str, Any]]:
    system_prompt = NAVIGATION_SYSTEM_PROMPT.format(
        output_format=json.dumps(NavigationStep.model_json_schema(), indent=2),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    
    history_str = "\n".join(history) if history else "No history yet."
    
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [
            {"type": "text", "text": f"<task>\n{task}\n</task>\n"},
            {"type": "text", "text": f"<history>\n{history_str}\n</history>\n"},
            {"type": "text", "text": f"<observation step={step}>\n<screenshot>\n"},
            {"type": "image", "image": image},
            {"type": "text", "text": "\n</screenshot>\n</observation>\n"},
        ]},
    ]


VALIDATOR_SYSTEM_PROMPT = """You are an expert evaluator for a computer agent. Your task is to validate if the agent's last action was successful in progressing towards the goal.
You will be given the main goal, the action the agent just took, and a screenshot of the result.
Analyze the evidence. Was the action performed correctly? Did it help achieve the goal?
Respond in the specified JSON format. Provide clear, concise feedback.
# <output_json_format>
# ```json
# {output_format}
# ```
# </output_json_format>
"""


def get_validator_prompt(goal: str, action_taken: str, image: Image.Image) -> List[dict[str, Any]]:
    system_prompt = VALIDATOR_SYSTEM_PROMPT.format(
        output_format=json.dumps(ValidationResult.model_json_schema(), indent=2)
    )
    user_prompt = (f"<goal>\n{goal}\n</goal>\n\n"
                   f"<agent_action_taken>\n{action_taken}\n</agent_action_taken>\n\n"
                   "<screenshot_of_result>\n")
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image", "image": image},
            {"type": "text", "text": "\n</screenshot_of_result>\n"},
        ]},
    ]
