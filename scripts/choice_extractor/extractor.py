from typing import Any, Dict, Optional

import requests


class ChoiceExtractor:
    def __init__(
        self,
        base_url: str = "http://localhost:7761",
    ):
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/api/v1/choice/extract-choices"

    def extract_frame(
        self,
        image_bytes: bytes,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Sends image bytes to the FastAPI backend to extract choice titles.
        """
        # FastAPI's UploadFile expects a file tuple: (filename, content, content_type)
        # We assign a generic filename and standard image/png type.
        files = {"file": ("screenshot.png", image_bytes, "image/png")}

        # Any additional parameters are sent as form data or query params
        params = {}
        if prompt:
            params["prompt"] = prompt
        if model:
            params["model"] = model

        try:
            # Make the POST request to the FastAPI route
            response = requests.post(self.endpoint, files=files, params=params)

            # Raise an HTTPError if the status code is 4xx or 5xx
            response.raise_for_status()

            # Returns the validated dictionary: {'choices': [...], 'selected_choice': '...'}
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Failed to communicate with the extraction API: {e}")
            raise
