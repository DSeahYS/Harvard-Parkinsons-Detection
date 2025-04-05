import os
import logging
import requests # Add 'requests' to requirements.txt if not already there
# from dotenv import load_dotenv

# load_dotenv() # Load .env file if using one

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Default Prompt Template ---
DEFAULT_PROMPT_TEMPLATE = """
Analyze the following patient data for potential Parkinson's Disease (PD) risk indicators.
Combine the eye-tracking metrics and simulated genomic factors to provide a concise summary (2-3 sentences)
highlighting key risk contributors and potential implications. Be cautious and avoid definitive diagnoses.

Patient Information:
- Ethnicity: {ethnicity}
- Age (if available): {age}
- Relevant Medical History: {medical_history}

Eye-Tracking Metrics Summary:
- PD Risk Score (0-1): {eye_risk_level:.3f}
- Key Contributing Eye Factors: {eye_factors}
- Blink Rate (bpm): {blink_rate:.1f}
- Fixation Stability Score: {fixation_stability:.3f}
- Saccade Velocity (deg/s): {saccade_velocity:.1f}

Simulated Genomic Analysis:
- Profile: {genomic_profile}
- Combined Genetic Risk Score (0-1): {combined_genetic_risk:.3f}
- Simulated LRRK2 Risk: {lrrk2_risk:.3f}
- Simulated GBA Risk: {gba_risk:.3f}
- Ethnicity Adjustment Factor Applied: {ethnicity_adj_factor:.2f}

Analysis:
"""

class OpenRouterClient:
    """
    Client for interacting with the OpenRouter API to get LLM analysis.
    """
    def __init__(self, api_key=None, model="openai/gpt-3.5-turbo"):
        """
        Initializes the OpenRouter client.

        Args:
            api_key (str, optional): OpenRouter API key. Defaults to OPENROUTER_API_KEY env var.
            model (str): The specific LLM model to use via OpenRouter.
        """
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

        if not self.api_key:
            logging.warning("OpenRouter API key not found. LLM analysis will be disabled.")
        else:
            logging.info(f"OpenRouter client initialized for model '{self.model}'.")

    def _format_prompt(self, template, data):
        """Formats the prompt using the provided template and data."""
        # Prepare data with defaults for missing keys to avoid errors
        prompt_data = {
            'ethnicity': data.get('patient_info', {}).get('ethnicity', 'N/A'),
            'age': data.get('patient_info', {}).get('age', 'N/A'), # Assuming age might be added
            'medical_history': data.get('patient_info', {}).get('medical_history', 'N/A'),
            'eye_risk_level': data.get('eye_metrics_summary', {}).get('risk_level', 0.0),
            'eye_factors': str(data.get('eye_metrics_summary', {}).get('contributing_factors', {})),
            'blink_rate': data.get('eye_metrics_raw', {}).get('blink_rate', 0.0),
            'fixation_stability': data.get('eye_metrics_raw', {}).get('fixation_stability', 0.0),
            'saccade_velocity': data.get('eye_metrics_raw', {}).get('saccade_velocity', 0.0),
            'genomic_profile': data.get('genomic_results', {}).get('variant_profile', 'N/A'),
            'combined_genetic_risk': data.get('genomic_results', {}).get('combined_genetic_risk', 0.0),
            'lrrk2_risk': data.get('genomic_results', {}).get('simulated_lrrk2_risk', 0.0),
            'gba_risk': data.get('genomic_results', {}).get('simulated_gba_risk', 0.0),
            'ethnicity_adj_factor': data.get('genomic_results', {}).get('ethnicity_adjustment_factor', 1.0),
        }
        try:
            return template.format(**prompt_data)
        except KeyError as e:
            logging.error(f"Missing key in prompt data: {e}")
            return f"Error: Missing data for prompt key {e}"
        except Exception as e:
            logging.error(f"Error formatting prompt: {e}")
            return "Error: Could not format prompt."


    def analyze_combined_data(self, eye_metrics_summary, eye_metrics_raw, genomic_results, patient_info=None, prompt_template=DEFAULT_PROMPT_TEMPLATE):
        """
        Sends combined data to OpenRouter LLM for analysis.

        Args:
            eye_metrics_summary (dict): Summary from PDDetector (risk_level, factors).
            eye_metrics_raw (dict): Raw metrics from EyeTracker.
            genomic_results (dict): Results from BioNeMoClient.
            patient_info (dict, optional): Basic patient details (ethnicity, etc.).
            prompt_template (str, optional): Custom prompt template string.

        Returns:
            str: The analysis text from the LLM, or an error message.
        """
        if not self.api_key:
            return "LLM analysis disabled: API key not configured."

        if not all([eye_metrics_summary, eye_metrics_raw, genomic_results]):
             logging.warning("Attempted LLM analysis with incomplete data.")
             return "LLM analysis requires eye metrics and genomic results."

        combined_data = {
            "patient_info": patient_info or {},
            "eye_metrics_summary": eye_metrics_summary,
            "eye_metrics_raw": eye_metrics_raw,
            "genomic_results": genomic_results
        }

        prompt = self._format_prompt(prompt_template, combined_data)
        if prompt.startswith("Error:"):
            return prompt # Return formatting error

        logging.info(f"Sending request to OpenRouter model: {self.model}")
        # logging.debug(f"Prompt:\n{prompt}") # Be careful logging PII

        try:
            response = requests.post(
                url=self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                },
                timeout=30 # Set a timeout
            )
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            result = response.json()
            analysis = result['choices'][0]['message']['content']
            logging.info("Received analysis from OpenRouter.")
            # logging.debug(f"Analysis:\n{analysis}")
            return analysis.strip()

        except requests.exceptions.RequestException as e:
            logging.error(f"Error calling OpenRouter API: {e}")
            return f"Error: Could not connect to OpenRouter API. {e}"
        except KeyError as e:
            logging.error(f"Error parsing OpenRouter response: Missing key {e}. Response: {result}")
            return "Error: Invalid response format from OpenRouter."
        except Exception as e:
            logging.error(f"An unexpected error occurred during OpenRouter analysis: {e}")
            return f"Error: An unexpected error occurred. {e}"


# Example Usage
if __name__ == '__main__':
    # Make sure to set OPENROUTER_API_KEY environment variable for this test
    client = OpenRouterClient()

    if not client.api_key:
        print("Skipping OpenRouter test: OPENROUTER_API_KEY not set.")
    else:
        print("\n--- Testing OpenRouter Client ---")
        # Sample data (mimicking outputs from other modules)
        sample_patient = {'ethnicity': 'Malay', 'medical_history': 'None relevant'}
        sample_eye_summary = {'risk_level': 0.65, 'contributing_factors': {'Fixation Stability': 0.3, 'Saccade Velocity (Inv)': 0.25}}
        sample_eye_raw = {'blink_rate': 12.5, 'fixation_stability': 0.7, 'saccade_velocity': 150.0}
        sample_genomic = {
            'simulated_lrrk2_risk': 0.18,
            'simulated_gba_risk': 0.12,
            'combined_genetic_risk': 0.30,
            'variant_profile': 'high_risk_profile',
            'ethnicity_adjustment_factor': 1.22
        }

        analysis_result = client.analyze_combined_data(
            eye_metrics_summary=sample_eye_summary,
            eye_metrics_raw=sample_eye_raw,
            genomic_results=sample_genomic,
            patient_info=sample_patient
        )

        print("\nLLM Analysis Result:")
        print(analysis_result)
