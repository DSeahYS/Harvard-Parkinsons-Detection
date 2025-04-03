# src/llm/openrouter_client.py
import os
import requests
import json
import time

class OpenRouterClient:
    """Client for accessing DeepSeek V3 via OpenRouter API"""

    def __init__(self, api_key=None):
        """Initialize with API key from parameter or environment variable"""
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            print("Warning: No OpenRouter API key found. Will use mock analysis.")
        self.base_url = "https://openrouter.ai/api/v1"
        # Using the specific free model mentioned
        self.model = "deepseek/deepseek-chat:free" 

    def analyze_patient(self, patient_data, eye_metrics, genomic_data=None):
        """Generate clinical analysis using DeepSeek V3"""
        if not self.api_key:
            print("OpenRouter API key not provided. Generating mock analysis.")
            return self._generate_mock_analysis(patient_data, eye_metrics, genomic_data)

        # Format the prompt for clinical analysis
        prompt = self._format_prompt(patient_data, eye_metrics, genomic_data)

        try:
            # Setup headers
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "GenomicGuard-PD-Detection",  # Recommended for OpenRouter
                "X-Title": "GenomicGuard",                    # Recommended for OpenRouter
                "Content-Type": "application/json"
            }

            # Prepare request payload
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a neurological diagnostic assistant specializing in Parkinson's disease detection. You analyze ocular biomarkers and genetic information to provide clinical assessments."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3 # Recommended temperature for DeepSeek
            }

            # Make API call
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60 # Add a timeout
            )

            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                 # Check if message content exists
                message_content = result["choices"][0].get("message", {}).get("content")
                if message_content:
                    return {
                        "analysis": message_content,
                        "model_used": result.get("model", self.model), # Use model from response if available
                        "success": True
                    }
                else:
                    print("Error: LLM response missing message content.")
                    return self._generate_error_analysis("LLM response format error (missing content)")
            else:
                print(f"Error: Unexpected response format from OpenRouter: {result}")
                return self._generate_error_analysis("Unexpected LLM response format")


        except requests.exceptions.RequestException as e:
            print(f"Error calling OpenRouter API: {e}")
            # Fall back to mock analysis on network/API errors
            return self._generate_mock_analysis(patient_data, eye_metrics, genomic_data)
        except Exception as e:
            print(f"An unexpected error occurred during LLM analysis: {e}")
            return self._generate_error_analysis(f"Unexpected error: {e}")

    def _format_prompt(self, patient_data, eye_metrics, genomic_data):
        """Format clinical data for LLM analysis"""
        # Create a comprehensive prompt with all available data
        prompt = f"""
# Patient Clinical Assessment Request

## Patient Information
- Name: {patient_data.get('name', 'Anonymous')}
- Age: {patient_data.get('age', 'Unknown')}
- Sex: {patient_data.get('sex', 'Unknown')}
- Medical History: {patient_data.get('medical_history', 'None provided')}

## Ocular Biomarkers (Primary Diagnostic Data)
- Fixation Stability: {eye_metrics.get('fixation_stability', 'N/A')} (Normal range: <0.3)
- Saccade Velocity (Avg): {eye_metrics.get('avg_saccade_velocity', 'N/A')}°/s (Normal range: 400-600°/s)
- Vertical Saccade Velocity (Avg): {eye_metrics.get('avg_vertical_saccade_velocity', 'N/A')}°/s (Threshold: <200°/s suggests PD)
- Eye Aspect Ratio (Avg EAR): {eye_metrics.get('avg_ear', 'N/A')} (Normal range: 0.2-0.5)
- Blink Rate: {eye_metrics.get('blink_rate', 'N/A')} per minute (Normal range: 8-21)
- Overall Eye-Based Risk Score: {eye_metrics.get('risk_level', 0) * 100:.1f}%
"""

        if genomic_data and genomic_data.get('patient_variants'):
            prompt += f"""
## Genomic Analysis (Secondary Data - Simulated)
- Genomic Risk Score: {genomic_data.get('genomic_risk_score', 0) * 100:.1f}%
- Detected Variants:
"""
            variants_list = genomic_data.get('patient_variants', {})
            if variants_list:
                 for gene, data in variants_list.items():
                    prompt += f"  - {gene}: {data.get('variant', 'Unknown variant')} (Risk Contribution: {data.get('risk_contribution', 'N/A'):.2f}x)\n"
            else:
                prompt += "  - None detected in simulation.\n"
        else:
             prompt += "\n## Genomic Analysis (Secondary Data - Simulated)\n- No simulated genomic variants generated based on risk level.\n"


        prompt += """
## Requested Analysis

Based on the ocular biomarkers and any available simulated genomic data, please provide:

1.  **Risk Assessment**: Synthesize the eye-tracking and genomic data to estimate the likelihood this patient is developing Parkinson's disease. Explain the confidence level.
2.  **Biomarker Analysis**: Detail which specific indicators (ocular and genomic, if present) most strongly suggest pre-symptomatic Parkinson's and why. Correlate these with known PD pathology.
3.  **Clinical Recommendations**: Suggest concrete next steps, including follow-up tests, monitoring frequency, and potential specialist referrals.
4.  **Preventive/Management Considerations**: Discuss potential lifestyle or other interventions that could be considered based on the risk profile.
5.  **Summary Report**: Generate a concise clinical summary suitable for inclusion in a patient record or referral to a neurologist.

Focus particularly on how the ocular metrics correlate with early-stage Parkinson's pathology. The fixation stability and saccade velocities (both overall and vertical) are especially relevant clinical indicators based on recent research. Ensure the language is professional and clinically appropriate.

**MOH Requirements (Singapore Context):**
- Include the standard NHG disclaimer for AI-assisted analysis if applicable.
- Reference the latest SingHealth Parkinson's Disease Clinical Practice Guidelines (currently 2024).
- Use terminology approved by the Singapore Medical Council (SMC) where relevant (e.g., specific referral pathways).
"""

        return prompt.strip() # Remove leading/trailing whitespace

    def _generate_mock_analysis(self, patient_data, eye_metrics, genomic_data):
        """Generate mock analysis when API key not available or API fails"""
        print("Generating mock analysis...")
        # Calculate risk level from eye metrics
        risk_level = eye_metrics.get('risk_level', 0.5) # Default to medium risk if unavailable
        name = patient_data.get('name', 'the patient')

        # Create simulated analysis with formatting similar to LLM output
        analysis = f"""
# Clinical Assessment for {name} (Simulated Analysis)

## Risk Assessment

Based on the provided ocular biomarkers, there is a **{risk_level*100:.1f}% estimated probability** that this patient shows patterns consistent with the pre-symptomatic stage of Parkinson's disease. This simulation uses threshold-based rules derived from eye movement patterns known to be affected in early PD.

## Biomarker Analysis (Simulated Interpretation)

The following indicators contributed to the risk score:

1.  **Fixation Stability**: {eye_metrics.get('fixation_stability', 'N/A')} (Threshold: >0.3). Values above the threshold suggest potential issues with gaze control, sometimes seen in early PD.
2.  **Saccade Velocity**: {eye_metrics.get('avg_saccade_velocity', 'N/A')}°/s (Threshold: <400°/s). Reduced velocity can indicate basal ganglia dysfunction.
3.  **Vertical Saccade Velocity**: {eye_metrics.get('avg_vertical_saccade_velocity', 'N/A')}°/s (Threshold: <200°/s). Vertical saccades are often disproportionately affected in PD.
"""

        if genomic_data and genomic_data.get('patient_variants'):
            variants = list(genomic_data['patient_variants'].keys())
            analysis += f"""
4.  **Simulated Genetic Factors**: The simulation included variants in {', '.join(variants)}, contributing {genomic_data.get('genomic_risk_score', 0)*100:.1f}% to the simulated genomic risk.
"""
        else:
             analysis += "\n4.  **Simulated Genetic Factors**: No high-risk variants were included in this simulation based on the eye-tracking risk level.\n"


        analysis += """
## Clinical Recommendations (Simulated)

1.  **Neurological Consultation**: Recommend consultation with a movement disorder specialist for clinical evaluation.
2.  **Baseline Monitoring**: Establish baseline clinical scores (e.g., UPDRS) and consider follow-up ocular assessment in 6-12 months.
3.  **Consider DaTscan**: If clinical suspicion remains high, dopamine transporter imaging could be informative.

## Preventive/Management Considerations (Simulated)

General recommendations often include:
1.  **Lifestyle Factors**: Regular exercise (aerobic, balance), Mediterranean diet.
2.  **Symptom Monitoring**: Patient education on early motor and non-motor symptoms of PD.

## Summary Report (Simulated)

Patient {name} presents with ocular metrics yielding a {risk_level*100:.1f}% risk score based on algorithmic analysis. Key contributing factors include [mention 1-2 key factors like fixation stability or saccade velocity if abnormal, otherwise state 'metrics within normal limits']. Simulated genomic data [mention if variants were included or not]. Recommend neurological consultation for clinical correlation and baseline assessment.
"""

        return {
            "analysis": analysis.strip(),
            "model_used": "Mock Analysis (Simulated DeepSeek V3)",
            "success": True # Mock analysis is considered successful for demo purposes
        }

    def _generate_error_analysis(self, error_message):
         """Generate an analysis indicating an error occurred."""
         print(f"Generating error analysis message: {error_message}")
         analysis = f"""
# Clinical Assessment Error

An error occurred while attempting to generate the AI clinical assessment.

**Error Details:** {error_message}

Please check the application logs and ensure the OpenRouter API key is valid and the service is reachable.

**Recommendation:** Review the raw eye-tracking metrics and genomic simulation results manually. Consider retrying the analysis later.
"""
         return {
            "analysis": analysis.strip(),
            "model_used": "Error State",
            "success": False
         }
