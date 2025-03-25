"""
llm.py - module with helper function to load prompts from external
    YAML files. All YAML files are in the config folder

Author: Manish Bhobe
My experiments with Python, ML and Generative AI.
Code is meant for illustration purposes ONLY. Use at your own risk!
Author is not liable for any damages arising from direct/indirect use of this code.
"""
import pathlib
import yaml

def load_prompts_from_config(prompts_file_path: pathlib.Path):
    if not prompts_file_path.exists():
        raise RuntimeError(f"FATAL ERROR: prompts file {str(prompts_file_path)} does not exist!")
    
    config = None
    with open(str(prompts_file_path), "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise RuntimeError(
            f"FATAL ERROR: unable to read from configuration file at {prompts_file_path}"
        )

    return config


