# computer use agent - main entry point

import torch
from core import model
from ui.gradio_app import create_app


def main():
    model.load_model()
    
    if torch.cuda.is_available():
        print(f"CUDA available: True, Device: {torch.cuda.get_device_name()}")
    else:
        print("CUDA available: False, running on CPU")
    
    demo = create_app()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
