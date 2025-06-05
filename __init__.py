# Define the global mappings as empty dictionaries first
# This ensures they exist before we try to update them
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Import your custom node files here
# For example, your conditional_lora_selector node (now an applier):
from . import conditional_lora_selector as conditional_lora_applier # Use an alias for clarity

# Update the global mappings with the nodes from conditional_lora_applier.py
NODE_CLASS_MAPPINGS.update(conditional_lora_applier.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(conditional_lora_applier.NODE_DISPLAY_NAME_MAPPINGS)

# This part is crucial for ComfyUI to discover your nodes
# It must return the MAPPINGS dictionaries
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
