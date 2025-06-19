# gradio interface

import gradio as gr
import json
import re
import time
import traceback
from PIL import Image, ImageDraw

from core import model
from core.actions import (
    ActionSpace, AnswerAction, NavigationStep, ValidationResult,
    capture_screen, execute_action, parse_coordinates
)
from core.prompts import (
    get_localization_prompt, get_navigation_prompt, get_validator_prompt
)

_agent_stop_requested = False

def request_stop():
    global _agent_stop_requested
    _agent_stop_requested = True

def reset_stop():
    global _agent_stop_requested
    _agent_stop_requested = False

def is_stop_requested():
    return _agent_stop_requested


def predict_click_location(input_image: Image.Image, instruction: str):
    if not model.model_loaded:
        yield "Error: Model not loaded", None
        return
    
    if input_image is None:
        yield "Error: Please upload an image", None
        return
    
    if not instruction or not instruction.strip():
        yield "Error: Please provide an instruction", None
        return
    
    try:
        resized_image = model.resize_image_for_model(input_image)
        loc_prompt = get_localization_prompt(resized_image, instruction)
        
        yield "Localizing element...", None
        
        coords_str = ""
        for partial_output in model.run_inference_streaming(
            loc_prompt, resized_image, max_new_tokens=32, thinking=False
        ):
            coords_str = partial_output
            yield f"Generating: {partial_output}", None
        
        print(f"[DEBUG] Localization output: {repr(coords_str)}")
        
        x, y = parse_coordinates(coords_str, resized_image.width, resized_image.height)
        
        if x is None or y is None:
            yield f"Could not parse coordinates from: {coords_str}", None
            return
        
        marked_image = resized_image.copy()
        draw = ImageDraw.Draw(marked_image)
        
        radius = 15
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline="red", width=3)
        draw.line([x - radius - 5, y, x + radius + 5, y], fill="red", width=2)
        draw.line([x, y - radius - 5, x, y + radius + 5], fill="red", width=2)
        
        result_text = f"Predicted coordinates: x={x}, y={y}\n\nRaw model output: {coords_str}"
        yield result_text, marked_image
        
    except Exception as e:
        error_msg = f"Error during localization: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        yield error_msg, None


def predict_navigation_step(input_image: Image.Image, task: str):
    if not model.model_loaded:
        yield "Error: Model not loaded"
        return
    
    if input_image is None:
        yield "Error: Please upload an image"
        return
    
    if not task or not task.strip():
        yield "Error: Please provide a task description"
        return
    
    try:
        resized_image = model.resize_image_for_model(input_image)
        nav_prompt = get_navigation_prompt(task, resized_image, history=[], step=1)
        
        yield "Generating... (watching model think)\n\n"
        
        nav_str = ""
        for partial_output in model.run_inference_streaming(
            nav_prompt, resized_image, max_new_tokens=1024, thinking=True
        ):
            nav_str = partial_output
            yield f"GENERATING...\n\n{partial_output}"
        
        print(f"[DEBUG] Navigation output: {repr(nav_str[:500] if len(nav_str) > 500 else nav_str)}")
        
        clean_output = nav_str.strip()
        
        if "```json" in clean_output:
            start = clean_output.find("```json") + 7
            end = clean_output.find("```", start)
            if end != -1:
                clean_output = clean_output[start:end].strip()
        elif "```" in clean_output:
            start = clean_output.find("```") + 3
            end = clean_output.find("```", start)
            if end != -1:
                clean_output = clean_output[start:end].strip()
        
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', clean_output, re.DOTALL)
        if json_match:
            clean_output = json_match.group(0)
        
        try:
            nav_data = json.loads(clean_output)
            nav_step = NavigationStep(**nav_data)
            
            result = f"PARSED NAVIGATION STEP\n\n"
            result += f"Note: {nav_step.note}\n\n"
            result += f"Thought: {nav_step.thought}\n\n"
            result += f"Action: {nav_step.action.action}\n"
            
            action_dict = nav_step.action.model_dump()
            for key, value in action_dict.items():
                if key != "action":
                    result += f"  {key}: {value}\n"
            
            result += f"\n=== RAW OUTPUT ===\n{nav_str}"
            yield result
            
        except (json.JSONDecodeError, Exception) as parse_error:
            yield f"Could not parse JSON:\n{parse_error}\n\n=== RAW OUTPUT ===\n{nav_str}"
        
    except Exception as e:
        error_msg = f"Error during navigation: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        yield error_msg


def validate_action(input_image: Image.Image, goal: str, action_taken: str):
    if not model.model_loaded:
        return "Error: Model not loaded"
    
    if input_image is None:
        return "Error: Please upload an image"
    
    if not goal or not goal.strip():
        return "Error: Please provide the original goal"
    
    if not action_taken or not action_taken.strip():
        return "Error: Please describe the action that was taken"
    
    try:
        resized_image = model.resize_image_for_model(input_image)
        val_prompt = get_validator_prompt(goal, action_taken, resized_image)
        val_str = model.run_inference(val_prompt, resized_image, max_new_tokens=256, thinking=True)
        
        print(f"[DEBUG] Validation output: {repr(val_str)}")
        
        clean_output = val_str.strip()
        
        if "```json" in clean_output:
            start = clean_output.find("```json") + 7
            end = clean_output.find("```", start)
            if end != -1:
                clean_output = clean_output[start:end].strip()
        elif "```" in clean_output:
            start = clean_output.find("```") + 3
            end = clean_output.find("```", start)
            if end != -1:
                clean_output = clean_output[start:end].strip()
        
        json_match = re.search(r'\{[^{}]*\}', clean_output)
        if json_match:
            clean_output = json_match.group(0)
        
        try:
            val_data = json.loads(clean_output)
            validation = ValidationResult(**val_data)
            
            status = "VALID" if validation.is_correct else "INVALID"
            result = f"=== VALIDATION RESULT ===\n\n"
            result += f"Status: {status}\n\n"
            result += f"Feedback: {validation.feedback}\n\n"
            result += f"=== RAW OUTPUT ===\n{val_str}"
            return result
            
        except (json.JSONDecodeError, Exception) as parse_error:
            return f"Could not parse JSON:\n{parse_error}\n\n=== RAW OUTPUT ===\n{val_str}"
        
    except Exception as e:
        error_msg = f"Error during validation: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg


def localize_element(resized_image: Image.Image, element: str):
    loc_prompt = get_localization_prompt(resized_image, element)
    coords_str = ""
    for partial in model.run_inference_streaming(loc_prompt, resized_image, max_new_tokens=32, thinking=False):
        coords_str = partial
    print(f"[DEBUG] Localization output: {repr(coords_str)}")
    return parse_coordinates(coords_str, resized_image.width, resized_image.height)


def run_autonomous_agent(task: str, max_steps: int = 10, enable_validation: bool = False, enable_thinking: bool = True):
    if not model.model_loaded:
        yield "Model not loaded.", None, gr.update(value="Start Agent", interactive=True), gr.update(interactive=False)
        return

    reset_stop()
    history = []
    full_log = ""
    
    for i in range(1, max_steps + 1):
        if is_stop_requested():
            yield full_log + f"\n\n{'='*50}\nSTOPPED BY USER\n{'='*50}", None, gr.update(value="Start Agent", interactive=True), gr.update(interactive=False)
            return
            
        step_header = f"\n{'='*50}\nSTEP {i}\n{'='*50}"
        try:
            yield full_log + step_header + "\nCapturing screen...", None, gr.update(), gr.update()
            time.sleep(1)
            
            if is_stop_requested():
                yield full_log + f"\n\n{'='*50}\nSTOPPED BY USER\n{'='*50}", None, gr.update(value="Start Agent", interactive=True), gr.update(interactive=False)
                return
                
            screenshot = capture_screen()
            resized_screenshot = model.resize_image_for_model(screenshot)

            yield full_log + step_header + "\nThinking...", screenshot, gr.update(), gr.update()
            nav_prompt = get_navigation_prompt(task, resized_screenshot, history, i)
            
            nav_str = ""
            for partial_output in model.run_inference_streaming(nav_prompt, resized_screenshot, max_new_tokens=1024, thinking=enable_thinking):
                if is_stop_requested():
                    yield full_log + f"\n\n{'='*50}\nSTOPPED BY USER\n{'='*50}", screenshot, gr.update(value="Start Agent", interactive=True), gr.update(interactive=False)
                    return
                nav_str = partial_output
                thinking_display = f"\nThinking...\n|  {partial_output[-800:].replace(chr(10), chr(10) + '|  ') if len(partial_output) > 800 else partial_output.replace(chr(10), chr(10) + '|  ')}"
                yield full_log + step_header + thinking_display, screenshot, gr.update(), gr.update()
            
            print(f"[DEBUG] Raw model output: {repr(nav_str[:500] if len(nav_str) > 500 else nav_str)}")
            
            clean_nav_output = nav_str.strip()
            
            if "```json" in clean_nav_output:
                start = clean_nav_output.find("```json") + 7
                end = clean_nav_output.find("```", start)
                if end != -1:
                    clean_nav_output = clean_nav_output[start:end].strip()
            elif "```" in clean_nav_output:
                start = clean_nav_output.find("```") + 3
                end = clean_nav_output.find("```", start)
                if end != -1:
                    clean_nav_output = clean_nav_output[start:end].strip()
            
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', clean_nav_output, re.DOTALL)
            if json_match:
                clean_nav_output = json_match.group(0)
            
            print(f"[DEBUG] Cleaned output for JSON: {repr(clean_nav_output[:300] if len(clean_nav_output) > 300 else clean_nav_output)}")
            
            if not clean_nav_output:
                raise ValueError("Model returned empty output")
                
            nav_data = json.loads(clean_nav_output)
            current_step = NavigationStep(**nav_data)
            current_action = current_step.action

            action_desc = f"{current_action.action}"
            if hasattr(current_action, 'element'):
                action_desc += f" on '{current_action.element}'"
            
            step_summary = step_header
            step_summary += f"\nThought: {current_step.thought}"
            step_summary += f"\nAction: {action_desc}"
            
            yield full_log + step_summary + "\nExecuting...", screenshot, gr.update(), gr.update()

            if isinstance(current_action, AnswerAction):
                final_log = full_log + step_summary + f"\n\n{'='*50}\nTASK COMPLETE\n{'='*50}\nFinal Answer: {current_action.content}"
                yield final_log, screenshot, gr.update(value="Start Agent", interactive=True), gr.update(interactive=False)
                return

            execute_action(current_action, localize_element, model.resize_image_for_model)
            time.sleep(2)
            
            if is_stop_requested():
                yield full_log + step_summary + f"\n\n{'='*50}\nSTOPPED BY USER\n{'='*50}", None, gr.update(value="Start Agent", interactive=True), gr.update(interactive=False)
                return

            if enable_validation:
                yield full_log + step_summary + "\nValidating...", None, gr.update(), gr.update()
                after_screenshot = capture_screen()
                resized_after_screenshot = model.resize_image_for_model(after_screenshot)
                
                action_str_for_validation = json.dumps(current_action.model_dump())
                val_prompt = get_validator_prompt(task, action_str_for_validation, resized_after_screenshot)
                
                val_str = ""
                for partial_val in model.run_inference_streaming(val_prompt, resized_after_screenshot, max_new_tokens=256, thinking=enable_thinking):
                    if is_stop_requested():
                        yield full_log + step_summary + f"\n\n{'='*50}\nSTOPPED BY USER\n{'='*50}", after_screenshot, gr.update(value="Start Agent", interactive=True), gr.update(interactive=False)
                        return
                    val_str = partial_val
                    val_display = f"\nValidating...\n|  {partial_val[-400:].replace(chr(10), chr(10) + '|  ') if len(partial_val) > 400 else partial_val.replace(chr(10), chr(10) + '|  ')}"
                    yield full_log + step_summary + val_display, after_screenshot, gr.update(), gr.update()

                clean_val_output = val_str.strip()
                if "```json" in clean_val_output:
                    start = clean_val_output.find("```json") + 7
                    end = clean_val_output.find("```", start)
                    if end != -1:
                        clean_val_output = clean_val_output[start:end].strip()
                elif "```" in clean_val_output:
                    start = clean_val_output.find("```") + 3
                    end = clean_val_output.find("```", start)
                    if end != -1:
                        clean_val_output = clean_val_output[start:end].strip()
                
                json_match = re.search(r'\{[^{}]*\}', clean_val_output)
                if json_match:
                    clean_val_output = json_match.group(0)
                
                val_data = json.loads(clean_val_output)
                validation = ValidationResult(**val_data)
                
                val_status = "[OK]" if validation.is_correct else "[FAILED]"
                step_summary += f"\nResult: {val_status} {validation.feedback}"
                
                full_log += step_summary
                yield full_log, after_screenshot, gr.update(), gr.update()
                
                history.append(f"Step {i}: Thought: '{current_step.thought}'. Action: '{action_str_for_validation}'. Result: {val_status}. Feedback: '{validation.feedback}'")
            else:
                action_str_for_history = json.dumps(current_action.model_dump())
                step_summary += "\nResult: [SKIPPED VALIDATION]"
                full_log += step_summary
                after_screenshot = capture_screen()
                yield full_log, after_screenshot, gr.update(), gr.update()
                history.append(f"Step {i}: Thought: '{current_step.thought}'. Action: '{action_str_for_history}'.")

        except Exception as e:
            error_message = f"{full_log}\n\n{'='*50}\nERROR at Step {i}\n{'='*50}\n{traceback.format_exc()}"
            print(error_message)
            yield error_message, None, gr.update(value="Start Agent", interactive=True), gr.update(interactive=False)
            return
    
    yield full_log + f"\n\n{'='*50}\nREACHED MAX STEPS ({max_steps})\n{'='*50}", None, gr.update(value="Start Agent", interactive=True), gr.update(interactive=False)


def create_app():
    with gr.Blocks(title="Computer Use Agent") as demo:
        gr.Markdown("<h1 style='text-align: center;'>Action VLM: Autonomous Computer Agent</h1>")
        
        if not model.model_loaded:
            gr.Markdown(f"<h2 style='text-align: center; color: red;'>Error: Model Failed to Load</h2>")
            gr.Markdown(f"<center>{model.load_error_message}</center>")
        else:
            gr.Markdown(f"<p style='text-align: center'>Model loaded from: {model.MODEL_PATH}</p>")

        with gr.Tabs():
            with gr.TabItem("Autonomous Agent"):
                gr.Markdown("### Run the Agent to Autonomously Complete a Task")
                gr.Markdown("<p style='color: #888; font-size: 0.9em;'><b>Tip:</b> When you start the agent, a new browser tab will open automatically to prevent the agent from seeing its own interface.</p>")
                with gr.Row():
                    with gr.Column(scale=1):
                        agent_task = gr.Textbox(
                            label="High-Level Task",
                            placeholder="e.g., Search for 'latest AI news' on Google",
                            info="Provide the overall goal for the agent."
                        )
                        agent_max_steps = gr.Slider(minimum=1, maximum=20, value=10, step=1, label="Max Steps")
                        agent_speed = gr.Dropdown(
                            choices=["Quality", "Balanced", "Fast", "Fastest"],
                            value="Balanced",
                            label="Speed/Quality",
                            info="Lower quality = faster (reduces image resolution)"
                        )
                        agent_enable_thinking = gr.Checkbox(
                            label="Enable Thinking",
                            value=True,
                            info="Allow model to reason before acting (recommended)"
                        )
                        agent_enable_validation = gr.Checkbox(
                            label="Enable Validation",
                            value=False,
                            info="Validate each action (slower but more reliable)"
                        )
                        with gr.Row():
                            agent_start_button = gr.Button("Start Agent", variant="primary")
                            agent_stop_button = gr.Button("Stop", variant="stop", interactive=False)
                    
                    with gr.Column(scale=2):
                        agent_status = gr.Textbox(
                            label="Agent Status & Logs", 
                            interactive=False, 
                            lines=20
                        )
                        agent_screenshot = gr.Image(type="pil", label="Current View", height=400, interactive=False)
                
                def run_agent_with_button_state(task, max_steps, speed_preset, enable_thinking, enable_validation):
                    model.set_speed_preset(speed_preset)
                    import pyautogui
                    pyautogui.hotkey('ctrl', 't')
                    time.sleep(0.5)
                    yield "Starting agent...\n\n(New browser tab opened)", None, gr.update(value="Running...", interactive=False), gr.update(interactive=True)
                    for status, screenshot, start_btn_update, stop_btn_update in run_autonomous_agent(task, max_steps, enable_validation, enable_thinking):
                        yield status, screenshot, start_btn_update, stop_btn_update
                
                def stop_agent():
                    request_stop()
                    return gr.update(interactive=False, value="Stopping...")
                
                agent_start_button.click(
                    fn=run_agent_with_button_state,
                    inputs=[agent_task, agent_max_steps, agent_speed, agent_enable_thinking, agent_enable_validation],
                    outputs=[agent_status, agent_screenshot, agent_start_button, agent_stop_button]
                )
                
                agent_stop_button.click(
                    fn=stop_agent,
                    inputs=[],
                    outputs=[agent_stop_button]
                )

            with gr.TabItem("Manual Testing: Localization"):
                gr.Markdown("### Manually Localize a UI Element")
                gr.Markdown("Upload a screenshot and describe the element you want to locate. The model will predict click coordinates.")
                with gr.Row():
                    loc_input_image = gr.Image(type="pil", label="Input UI Image", height=400)
                    with gr.Column():
                        loc_instruction = gr.Textbox(
                            label="Instruction", 
                            placeholder="e.g., Click the 'Login' button, Click the search bar, Find the close button",
                            lines=2
                        )
                        loc_submit_button = gr.Button("Localize Click", variant="primary")
                        loc_output_coords = gr.Textbox(label="Predicted Coordinates", interactive=False, lines=5)
                        loc_output_image = gr.Image(type="pil", label="Image with Click Marker", height=300, interactive=False)
                
                loc_submit_button.click(
                    fn=predict_click_location, 
                    inputs=[loc_input_image, loc_instruction], 
                    outputs=[loc_output_coords, loc_output_image]
                )

            with gr.TabItem("Manual Testing: Navigation"):
                gr.Markdown("### Manually Determine the Next Navigation Step")
                gr.Markdown("Upload a screenshot and describe your task. The model will suggest the next action to take.")
                with gr.Row():
                    nav_input_image = gr.Image(type="pil", label="Input UI Image", height=400)
                    with gr.Column():
                        nav_task = gr.Textbox(
                            label="Task", 
                            placeholder="e.g., Book a flight from London to New York, Search for AI news",
                            lines=2
                        )
                        nav_submit_button = gr.Button("Determine Next Action", variant="primary")
                        nav_output_action = gr.Textbox(label="Predicted Navigation Step", interactive=False, lines=15)
                
                nav_submit_button.click(
                    fn=predict_navigation_step, 
                    inputs=[nav_input_image, nav_task], 
                    outputs=[nav_output_action]
                )

            with gr.TabItem("Manual Testing: Validator"):
                gr.Markdown("### Manually Validate an Action")
                gr.Markdown("Upload a result screenshot, describe the goal and action taken. The model will validate if the action was successful.")
                with gr.Row():
                    val_input_image = gr.Image(type="pil", label="Result Screenshot", height=400)
                    with gr.Column():
                        val_goal = gr.Textbox(
                            label="Original Goal", 
                            placeholder="e.g., Find the price of a flight to NYC",
                            lines=2
                        )
                        val_answer = gr.Textbox(
                            label="Agent's Action Taken", 
                            placeholder="e.g., Clicked on 'Search', Typed 'New York' in destination field",
                            lines=2
                        )
                        val_submit_button = gr.Button("Validate", variant="primary")
                        val_output_feedback = gr.Textbox(label="Validation Result", interactive=False, lines=10)
                
                val_submit_button.click(
                    fn=validate_action, 
                    inputs=[val_input_image, val_goal, val_answer], 
                    outputs=[val_output_feedback]
                )

    return demo
