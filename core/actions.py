# action classes and execution

import json
import re
import time
from typing import Literal, Union

import pyautogui
from pydantic import BaseModel, Field
from PIL import Image
import mss


class ClickElementAction(BaseModel):
    action: Literal["click_element"] = Field(description="Click at absolute coordinates of a web element")
    element: str = Field(description="Text description of the element")

class WriteElementAction(BaseModel):
    action: Literal["write_element_abs"] = Field(description="Write content at absolute coordinates of a web page")
    content: str = Field(description="Content to write")
    element: str = Field(description="Text description of the element")

class TypeAndEnterAction(BaseModel):
    action: Literal["type_and_enter"] = Field(description="Write content into an element and then press Enter.")
    content: str = Field(description="Content to write.")
    element: str = Field(description="Text description of the element to type into.")

class ScrollAction(BaseModel):
    action: Literal["scroll"] = Field(description="Scroll the page or a specific element")
    direction: Literal["down", "up", "left", "right"] = Field(description="The direction to scroll in")

class GoBackAction(BaseModel):
    action: Literal["go_back"] = Field(description="Navigate to the previous page")

class RefreshAction(BaseModel):
    action: Literal["refresh"] = Field(description="Refresh the current page")

class WaitAction(BaseModel):
    action: Literal["wait"] = Field(description="Wait for a particular amount of time")
    seconds: int = Field(default=2, ge=0, le=10, description="The number of seconds to wait")

class AnswerAction(BaseModel):
    action: Literal["answer"] = "answer"
    content: str = Field(description="The answer content")


ActionSpace = Union[
    ClickElementAction, WriteElementAction, TypeAndEnterAction, ScrollAction,
    GoBackAction, RefreshAction, WaitAction, AnswerAction
]


class NavigationStep(BaseModel):
    note: str = Field(default="", description="Task-relevant information extracted from the previous observation.")
    thought: str = Field(description="Reasoning about next steps (<4 lines)")
    action: ActionSpace = Field(description="Next action to take")

class ValidationResult(BaseModel):
    is_correct: bool = Field(description="Whether the provided answer correctly achieves the goal based on the screenshot.")
    feedback: str = Field(description="A detailed explanation for the decision. If incorrect, provide guidance on what went wrong.")


def capture_screen() -> Image.Image:
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)
        return Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")


def execute_action(action: ActionSpace, localize_fn, resize_fn):
    if isinstance(action, (ClickElementAction, WriteElementAction, TypeAndEnterAction)):
        print(f"Localizing element: {action.element}")
        screen_capture = capture_screen()
        resized_image = resize_fn(screen_capture)
        
        x_resized, y_resized = localize_fn(resized_image, action.element)
        
        if x_resized is None or y_resized is None:
            raise Exception(f"Could not find coordinates for element: {action.element}")

        screen_width, screen_height = pyautogui.size()
        x_scaled = int(x_resized * (screen_width / resized_image.width))
        y_scaled = int(y_resized * (screen_height / resized_image.height))

        pyautogui.moveTo(x_scaled, y_scaled, duration=0.25)
        pyautogui.click(x_scaled, y_scaled)
        print(f"Clicked at ({x_scaled}, {y_scaled})")

        if isinstance(action, (WriteElementAction, TypeAndEnterAction)):
            print(f"Typing: '{action.content}'")
            pyautogui.write(action.content, interval=0.05)
            if isinstance(action, TypeAndEnterAction):
                pyautogui.press('enter')
                print("Pressed Enter.")

    elif isinstance(action, ScrollAction):
        scroll_amount = -500 if action.direction == "down" else 500
        pyautogui.scroll(scroll_amount)
        print(f"Scrolled {action.direction}")

    elif isinstance(action, GoBackAction):
        pyautogui.hotkey('alt', 'left')
        print("Executed 'Go Back'")

    elif isinstance(action, RefreshAction):
        pyautogui.hotkey('ctrl', 'r')
        print("Refreshed page")
        
    elif isinstance(action, WaitAction):
        print(f"Waiting for {action.seconds} seconds...")
        time.sleep(action.seconds)


def parse_coordinates(coords_str: str, image_width: int = None, image_height: int = None):
    # model outputs coords normalized to 0-1000, need to convert to pixels
    x_norm, y_norm = None, None
    
    try:
        clean_output = coords_str.strip()
        if "```" in clean_output:
            clean_output = re.sub(r'```json?\s*', '', clean_output)
            clean_output = re.sub(r'```', '', clean_output).strip()
        
        json_match = re.search(r'\{[^{}]*\}', clean_output)
        if json_match:
            coords_json = json.loads(json_match.group(0))
            x_norm = int(coords_json.get("x", coords_json.get("X")))
            y_norm = int(coords_json.get("y", coords_json.get("Y")))
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        print(f"[DEBUG] JSON parse failed: {e}, trying Click format...")
    
    # fallback to Click(x, y) format
    if x_norm is None or y_norm is None:
        match = re.search(r"Click\((\d+),\s*(\d+)\)", coords_str)
        if match:
            x_norm, y_norm = int(match.group(1)), int(match.group(2))
    
    if x_norm is None or y_norm is None:
        return None, None
    
    # convert from 0-1000 normalized to actual pixels
    if image_width is not None and image_height is not None:
        x_pixel = int((x_norm * image_width) / 1000)
        y_pixel = int((y_norm * image_height) / 1000)
        print(f"[DEBUG] Converted normalized ({x_norm}, {y_norm}) to pixels ({x_pixel}, {y_pixel}) for {image_width}x{image_height}")
        return x_pixel, y_pixel
    
    print(f"[DEBUG] Returning normalized coords: ({x_norm}, {y_norm})")
    return x_norm, y_norm
