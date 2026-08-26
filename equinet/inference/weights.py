from huggingface_hub import hf_hub_download

REPO_ID = "cjmcgill/equinet-weights"

_WEIGHT_FILES = (
    "equinet_v0.2.0.pt",
    "equinet_no-self-activity-correction_v0.2.0.pt",
)

def get_pretrained_model_path(filename: str) -> str:
    """Get the path to a pretrained model checkpoint, downloading it if necessary."""
    if filename not in _WEIGHT_FILES:
        raise ValueError(f"Invalid filename '{filename}'. Must be one of {_WEIGHT_FILES}.")
    return hf_hub_download(repo_id=REPO_ID, filename=filename)