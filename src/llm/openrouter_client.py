import os
import logging
import requests
import json
import time
from . import prompt_templates # Import from local package
from ..utils.config import Config # Import config

# Configure logging (might be overridden by main)
log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenRouterClient:
    """
    Client for interacting with the OpenRouter API to get LLM analysis.
    Uses API key from Config and specific prompt templates.
    """
    def __init__(self, model="openai/gpt-3.5-turbo-1106"): # Default to a known good model
        """
        Initializes the OpenRouter client.

        Args:
            model (str): The specific LLM model to use via OpenRouter.
        """
        self.config = Config() # Get config instance
        self.api_key = self.config.get_openrouter_api_key() # Get key from config
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        # Define site_url and app_name for OpenRouter headers
        self.site_url = self.config.get_setting('APP_URL', 'http://localhost:3000') # Example, get from config if needed
        self.app_name = self.config.get_setting('APP_NAME', 'TrueGenomeGuard')

        if not self.api_key:
            logger.warning("OpenRouter API key not found in config or environment. LLM analysis will be disabled.")
        else:
            logger.info(f"OpenRouter client initialized for model '{self.model}'.")

    def _get_prompt_template(self, prompt_type):
        """Retrieves the appropriate prompt template."""
        if prompt_type == "summarize_current_assessment":
            return prompt_templates.PROMPT_SUMMARIZE_CURRENT
        elif prompt_type == "summarize_session_log":
            return prompt_templates.PROMPT_SUMMARIZE_HISTORY
        else:
            logger.warning(f"Unknown prompt type '{prompt_type}'. Using default.")
            # Fallback to a generic template or the current summary one
            return prompt_templates.PROMPT_SUMMARIZE_CURRENT

    def _format_prompt(self, template, data, prompt_type):
        """Formats the prompt using the provided template and data, adapting to prompt type."""
        prompt_data = {}
        patient_info = data.get('patient_info', {})
        prompt_data['ethnicity'] = patient_info.get('ethnicity', 'N/A')
        prompt_data['age'] = patient_info.get('age', 'N/A') # Assuming age might be added
        prompt_data['medical_history'] = patient_info.get('medical_history', 'N/A')

        if prompt_type == "summarize_current_assessment":
            eye_summary = data.get('current_assessment', {}) # Use a clearer key
            eye_raw = data.get('eye_metrics_raw', {}) # Get raw metrics separately
            genomic_res = data.get('genomic_results', {})

            prompt_data['current_risk'] = eye_summary.get('risk_level', 0.0)
            prompt_data['contributing_factors'] = str(eye_summary.get('contributing_factors', {}))
            prompt_data['blink_rate'] = eye_raw.get('blink_rate_bpm', 0.0)
            prompt_data['fixation_stability'] = eye_raw.get('fixation_stability', 0.0)
            prompt_data['saccade_velocity'] = eye_raw.get('saccade_velocity_deg_s', 0.0)

            prompt_data['genomic_profile'] = genomic_res.get('variant_profile', 'N/A')
            prompt_data['combined_genetic_risk'] = genomic_res.get('combined_risk_score', 1.0) # Default to 1x
            prompt_data['dominant_pathways'] = ', '.join(genomic_res.get('dominant_pathways', [])) or 'None'
            variants = genomic_res.get('variants_detected', [])
            prompt_data['key_variants'] = ', '.join([f"{v['gene']}-{v['variant']}" for v in variants[:3]]) or 'None'

        elif prompt_type == "summarize_session_log":
            log_data = data.get('session_log', []) # Log data might be needed for more complex prompts
            summary_stats = data.get('summary_stats', {}) # Expect pre-calculated stats

            prompt_data['session_id'] = data.get('session_id', 'N/A')
            prompt_data['num_entries'] = summary_stats.get('num_entries', len(log_data))
            prompt_data['duration'] = summary_stats.get('duration', 0.0)
            prompt_data['avg_risk'] = summary_stats.get('avg_risk', 0.0)
            prompt_data['max_risk'] = summary_stats.get('max_risk', 0.0)
            prompt_data['avg_saccade'] = summary_stats.get('avg_saccade', 0.0)
            prompt_data['avg_stability'] = summary_stats.get('avg_stability', 0.0)

        else: # Default/fallback case
             logger.warning(f"Data formatting not fully defined for prompt type '{prompt_type}'.")
             # Include basic info if available
             prompt_data['current_risk'] = data.get('current_assessment', {}).get('risk_level', 'N/A')
             prompt_data['combined_genetic_risk'] = data.get('genomic_results', {}).get('combined_risk_score', 'N/A')

        # Ensure None values are handled before formatting
        for key, value in prompt_data.items():
            if value is None:
                prompt_data[key] = "N/A"
            # Ensure boolean values are represented appropriately if needed
            # elif isinstance(value, bool):
            #     prompt_data[key] = str(value)

        # *** FIX: Removed pre-formatting loop. Let format_map handle types based on template. ***
        try:
            # Use format_map which ignores extra keys in prompt_data and handles types based on format specifiers in the template string
            return template.format_map(prompt_data)
        except KeyError as e:
            logger.error(f"Missing key in prompt template '{prompt_type}': {e}. Available data keys: {list(prompt_data.keys())}")
            return f"Error: Missing data for prompt key {e}"
        except ValueError as e:
             logger.error(f"Formatting error for prompt '{prompt_type}': {e}. Check template specifiers vs data types. Data: {prompt_data}", exc_info=True)
             return f"Error: Formatting mismatch for prompt. {e}"
        except Exception as e:
            logger.error(f"Error formatting prompt '{prompt_type}': {e}", exc_info=True)
            return "Error: Could not format prompt."


    def generate_summary(self, prompt_type, data):
        """
        Sends data to OpenRouter LLM for analysis based on the prompt type.

        Args:
            prompt_type (str): The type of summary needed (e.g., "summarize_current_assessment").
            data (dict): The input data required by the corresponding prompt template.

        Returns:
            str: The analysis text from the LLM, or an error message.
        """
        if not self.api_key:
            return "LLM analysis disabled: API key not configured."

        template = self._get_prompt_template(prompt_type)
        prompt = self._format_prompt(template, data, prompt_type)

        if prompt.startswith("Error:"):
            return prompt # Return formatting error

        logger.info(f"Sending request to OpenRouter model: {self.model} (Prompt Type: {prompt_type})")
        # Avoid logging full prompt if it contains sensitive data
        # logger.debug(f"Formatted Prompt Snippet: {prompt[:100]}...")

        max_retries = 3
        retry_delay = 2 # seconds
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url=self.api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": self.site_url, # Optional, but recommended by OpenRouter
                        "X-Title": self.app_name,     # Optional, but recommended by OpenRouter
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        # Optional parameters:
                        # "max_tokens": 150,
                        # "temperature": 0.7,
                    },
                    timeout=45 # Increased timeout for potentially slower models
                )
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                result = response.json()
                analysis = result.get('choices', [{}])[0].get('message', {}).get('content')

                if analysis:
                    logger.info("Received analysis from OpenRouter.")
                    # logger.debug(f"Analysis:\n{analysis}")
                    return analysis.strip()
                else:
                    logger.error(f"OpenRouter response missing content. Response: {result}")
                    return "Error: Received empty response from OpenRouter."

            except requests.exceptions.Timeout:
                 logger.warning(f"OpenRouter API request timed out (Attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay}s...")
                 if attempt == max_retries - 1:
                      return f"Error: OpenRouter API request timed out after {max_retries} attempts."
                 time.sleep(retry_delay)
            except requests.exceptions.RequestException as e:
                logger.error(f"Error calling OpenRouter API (Attempt {attempt + 1}/{max_retries}): {e}")
                # Check for specific status codes if needed (e.g., 429 Too Many Requests)
                status_code = e.response.status_code if e.response is not None else None
                if status_code == 429 or status_code >= 500: # Retry on rate limits or server errors
                     if attempt == max_retries - 1:
                          return f"Error: Could not connect to OpenRouter API after {max_retries} attempts. Status: {status_code}. {e}"
                     logger.warning(f"Retrying after API error (Status: {status_code})...")
                     time.sleep(retry_delay * (attempt + 1)) # Exponential backoff
                     continue
                else: # Don't retry for other client errors (e.g., 401 Unauthorized, 400 Bad Request)
                     return f"Error: Could not connect to OpenRouter API. Status: {status_code}. {e}"

            except KeyError as e:
                # Ensure result is defined before logging
                log_result = result if 'result' in locals() else 'Response not available'
                logger.error(f"Error parsing OpenRouter response: Missing key {e}. Response: {log_result}")
                return "Error: Invalid response format from OpenRouter."
            except Exception as e:
                logger.error(f"An unexpected error occurred during OpenRouter analysis: {e}", exc_info=True)
                # Don't retry on unexpected errors immediately
                return f"Error: An unexpected error occurred during LLM analysis. {e}"

        return f"Error: LLM analysis failed after {max_retries} attempts." # Should not be reached if retry logic is correct


# Example Usage
if __name__ == '__main__':
    # Set logger level to DEBUG for detailed output when run directly
    logging.getLogger(__name__).setLevel(logging.DEBUG)
    for handler in logging.getLogger(__name__).handlers:
        handler.setLevel(logging.DEBUG)

    # Make sure to set OPENROUTER_API_KEY environment variable for this test
    # or have it in a .env file at the project root (TrueGenomeGuard)
    client = OpenRouterClient()

    if not client.api_key:
        print("Skipping OpenRouter test: OPENROUTER_API_KEY not set in environment or config.")
    else:
        print("\n--- Testing OpenRouter Client ---")

        # --- Test Case 1: Current Assessment ---
        print("\n--- Test Case 1: Summarize Current Assessment ---")
        sample_patient = {'ethnicity': 'Malay', 'medical_history': 'None relevant', 'age': 55}
        sample_assessment = {
            'risk_level': 0.65,
            'contributing_factors': {'Fixation Stability': 0.3, 'Saccade Velocity (Inv)': 0.25, 'Blink Rate Deviation': 0.1},
        }
        sample_eye_raw = {'blink_rate_bpm': 12.5, 'fixation_stability': 0.7, 'saccade_velocity_deg_s': 150.0}
        sample_genomic = {
            'combined_risk_score': 1.8,
            'variant_profile': 'moderate_risk_profile',
            'dominant_pathways': ['lysosomal'],
            'variants_detected': [{'gene':'GBA', 'variant':'N370S'}, {'gene':'LRRK2', 'variant':'R1441G'}]
        }
        current_data = {
            "patient_info": sample_patient,
            "current_assessment": sample_assessment,
            "genomic_results": sample_genomic,
            "eye_metrics_raw": sample_eye_raw # Add raw metrics here
        }
        analysis_result_current = client.generate_summary(
            prompt_type="summarize_current_assessment",
            data=current_data
        )
        print("\nLLM Analysis Result (Current):")
        print(analysis_result_current)


        # --- Test Case 2: Historical Log ---
        print("\n--- Test Case 2: Summarize Historical Log ---")
        # Simulate summary stats calculated from a log
        sample_history_stats = {
             'num_entries': 150,
             'duration': 35.2,
             'avg_risk': 0.25,
             'max_risk': 0.40,
             'avg_saccade': 310.5,
             'avg_stability': 0.35
        }
        history_data = {
            "patient_info": sample_patient,
            "session_id": 12345,
            "summary_stats": sample_history_stats
            # "session_log": [...] # Actual log not needed if summary stats provided
        }
        analysis_result_history = client.generate_summary(
            prompt_type="summarize_session_log",
            data=history_data
        )
        print("\nLLM Analysis Result (History):")
        print(analysis_result_history)
