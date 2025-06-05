import os
import folder_paths # ComfyUI utility for finding paths
import comfy.sd # For load_lora_for_models
import comfy.utils # For load_torch_file

from nodes import LoraLoader

# Helper function to get filenames recursively within a ComfyUI models subdirectory
def get_recursive_filenames(folder_name):
    full_path_dir = folder_paths.get_folder_paths(folder_name)
    if not full_path_dir:
        return []

    filenames = []
    for base_dir in full_path_dir:
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                relative_path = os.path.relpath(os.path.join(root, file), base_dir)
                filenames.append(relative_path.replace("\\", "/")) # Ensure forward slashes for consistency
    return filenames

class ConditionalLoRAApplierCreepybits:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        lora_list = get_recursive_filenames("loras")
        lora_names = ["None"] + [l for l in lora_list if l.lower().endswith((".safetensors", ".ckpt"))]
        lora_names.sort(key=lambda x: x.lower()) # Sort for better readability in dropdown

        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "lora_definitions": ("STRING", {
                    "multiline": True,
                    "default": """
# Define your LoRA rules here.
# Format: keyword_phrase: lora_full_relative_path, lora_strength, clip_strength
# Example (use forward slashes for paths):
# portrait: Flux/Details/amateur_photo_v1.safetensors, 0.75, 1.0
# cinematic scene: MyLoRAs/Styles/retro_cinematic_v2.safetensors, 0.8, 0.9
# fantasy creature: Custom/Creatures/mythic_beast_lora.safetensors, 0.9, 0.9
# Use comma-separated values for strength. Default is 1.0 if omitted.
# Keep strength between -2.0 and 2.0.
# The keyword_phrase should be found anywhere in the prompt (case-insensitive by default).
"""
                }),
            },
            "optional": {
                "default_lora_name": (lora_names, {"default": "None"}),
                "default_lora_strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "default_clip_strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "case_sensitive": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP",)
    RETURN_NAMES = ("MODEL", "CLIP",)

    FUNCTION = "apply_conditional_lora"
    CATEGORY = "Creepybits/Model Patcher"

    def apply_conditional_lora(self, model, clip, prompt, lora_definitions, default_lora_name, default_lora_strength, default_clip_strength, case_sensitive):
        loras_to_apply = [] # List to store (filename, model_strength, clip_strength) for all matches

        processed_prompt = prompt if case_sensitive else prompt.lower()

        rules = lora_definitions.strip().split('\n')
        for rule_line in rules:
            rule_line = rule_line.strip()
            if not rule_line or rule_line.startswith('#'):
                continue

            try:
                parts = rule_line.split(':', 1)
                if len(parts) < 2:
                    print(f"Warning: Malformed LoRA rule (missing colon): '{rule_line}'")
                    continue

                keyword_phrase = parts[0].strip()
                lora_info_str = parts[1].strip()

                processed_keyword = keyword_phrase if case_sensitive else keyword_phrase.lower()

                if processed_keyword and processed_keyword in processed_prompt:
                    lora_details = [x.strip() for x in lora_info_str.split(',', 2)]

                    filename = lora_details[0]
                    strength = float(lora_details[1]) if len(lora_details) > 1 else 1.0
                    clip_strength = float(lora_details[2]) if len(lora_details) > 2 else 1.0

                    # Validate filename against available loras (recursively)
                    if filename.replace("\\", "/") not in get_recursive_filenames("loras"):
                        print(f"Warning: LoRA file '{filename}' not found in 'loras' directory (or subdirectories) for rule '{rule_line}'. Skipping this rule.")
                        continue

                    # If a match is found and validated, add it to the list (DO NOT BREAK)
                    loras_to_apply.append((filename, strength, clip_strength))

            except ValueError as e:
                print(f"Warning: Error parsing LoRA rule '{rule_line}': {e}. Skipping.")
            except Exception as e:
                print(f"An unexpected error occurred while parsing rule '{rule_line}': {e}. Skipping.")

        # --- LoRA Application Logic ---
        final_model = model
        final_clip = clip

        if not loras_to_apply: # If no rules matched, consider the default LoRA
            if default_lora_name and default_lora_name != "None":
                loras_to_apply.append((default_lora_name, default_lora_strength, default_clip_strength))

        if not loras_to_apply: # If still no LoRAs to apply, return original model/clip
            print("No LoRA selected or found for the given prompt/rules. Returning original model and clip.")
            return (final_model, final_clip,)

        # Apply all collected LoRAs sequentially
        for lora_filename, lora_strength, clip_strength in loras_to_apply:
            lora_path = folder_paths.get_full_path("loras", lora_filename)
            if not lora_path:
                print(f"Error: Could not find full path for LoRA '{lora_filename}'. Skipping this LoRA.")
                continue # Skip to the next LoRA if path is invalid

            print(f"Attempting to apply LoRA: {lora_filename} (model_strength={lora_strength}, clip_strength={clip_strength}) based on prompt.")

            try:
                # Load the raw LoRA data (state_dict) using comfy.utils.load_torch_file
                lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)

                # Use the comfy.sd.load_lora_for_models function directly!
                # This function is confirmed to exist and be used by Comfyroll in your setup.
                #final_model, final_clip = comfy.sd.load_lora_for_models(final_model, final_clip, lora_data, lora_strength, clip_strength)
                final_model, final_clip = LoraLoader().load_lora(final_model, final_clip, lora_filename, lora_strength, clip_strength)
                
                print(f"Successfully applied LoRA: {lora_filename}")
            except Exception as e:
                print(f"FATAL ERROR applying LoRA '{lora_filename}': {e}. Skipping this LoRA.")
                print(f"Details: {e}") # Print full exception details for debugging
                # Do NOT return here, continue to apply other LoRAs if possible or just return current state

        return (final_model, final_clip,)

NODE_CLASS_MAPPINGS = {
    "ConditionalLoRAApplierCreepybits": ConditionalLoRAApplierCreepybits
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConditionalLoRAApplierCreepybits": "Conditional LoRA Applier (Creepybits)"
}
