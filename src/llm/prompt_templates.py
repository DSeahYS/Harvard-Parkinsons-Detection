# src/llm/prompt_templates.py

"""
Prompt templates for GenomicGuard LLM integration.
These templates are optimized for DeepSeek V3 and focus on Parkinson's disease detection
using ocular biomarkers and genomic data.
"""

def clinical_assessment_prompt(patient_data, eye_metrics, genomic_data=None):
    """
    Generate a comprehensive clinical assessment prompt for neurologists.
    
    Args:
        patient_data (dict): Patient demographic and medical history information
        eye_metrics (dict): Ocular biomarkers and measurements from eye tracking
        genomic_data (dict, optional): Genomic analysis results if available
        
    Returns:
        str: Formatted prompt for clinical assessment
    """
    prompt = f"""# Clinical Assessment Request for Parkinson's Disease Analysis

## Patient Information
- **Name**: {patient_data.get('name', 'Anonymous')}
- **Age**: {patient_data.get('age', 'Unknown')}
- **Sex**: {patient_data.get('sex', 'Unknown')}
- **Medical History**: {patient_data.get('medical_history', 'None reported')}

## Ocular Biomarkers
The following metrics were obtained through high-precision eye tracking:

- **Fixation Stability**: {eye_metrics.get('fixation_stability', 'Unknown')} 
  *(Normal range: <0.3; Higher values indicate reduced stability characteristic of pre-motor PD)*
  
- **Saccade Velocity**: {eye_metrics.get('avg_saccade_velocity', 'Unknown')}°/s 
  *(Normal range: 400-600°/s; Reduced velocity observed in pre-symptomatic PD patients)*
  
- **Eye Aspect Ratio**: {eye_metrics.get('avg_ear', 'Unknown')} 
  *(Normal range: 0.2-0.5; Altered in various neurological conditions)*
  
- **Blink Rate**: {eye_metrics.get('blink_rate', 'Unknown')} blinks/minute 
  *(Normal range: 8-21; Both increased and decreased rates observed in PD)*

- **Overall Eye-Based Risk Assessment**: {eye_metrics.get('risk_level', 0) * 100:.1f}%
"""

    if genomic_data and genomic_data.get('patient_variants'):
        prompt += f"""
## Genomic Analysis
The genomic assessment revealed:

- **Overall Genomic Risk Score**: {genomic_data.get('genomic_risk_score', 0) * 100:.1f}%

- **Detected Variants**:
"""
        for gene, data in genomic_data.get('patient_variants', {}).items():
            prompt += f"  - **{gene}**: {data.get('variant', 'Unknown variant')} (Risk contribution: {data.get('risk_contribution', 'Unknown')}x)\n"

    prompt += """
## Analysis Request

Based on these ocular biomarkers and genomic findings, please provide:

1. **Risk Assessment**: Evaluate the likelihood that this patient is in the pre-symptomatic or early stage of Parkinson's disease.

2. **Biomarker Interpretation**: Analyze each ocular metric and its specific relevance to PD pathophysiology, particularly focusing on basal ganglia involvement in eye movement control.

3. **Clinical Recommendations**: Suggest appropriate follow-up evaluations, additional tests (e.g., DaTscan, neuropsychological assessment), and monitoring intervals.

4. **Differential Considerations**: Identify other conditions that might present with similar ocular findings.

5. **Treatment & Management**: Outline potential preventive strategies, lifestyle modifications, or monitoring protocols appropriate for this patient's risk profile.

6. **Prognostic Timeline**: Estimate the potential timeframe for symptom development based on the pre-symptomatic markers identified.

7. **Concise Summary**: Provide a brief clinical summary suitable for inclusion in medical records.

Please ensure that your analysis is evidence-based, referencing the established relationship between ocular biomarkers and pre-motor PD pathology. Current research indicates these metrics can detect PD 5-7 years before motor symptoms emerge.
"""
    return prompt


def technical_explanation_prompt(eye_metrics, genomic_data=None):
    """
    Generate a technical explanation of the metrics for healthcare professionals.
    
    Args:
        eye_metrics (dict): Ocular biomarkers and measurements from eye tracking
        genomic_data (dict, optional): Genomic analysis results if available
        
    Returns:
        str: Formatted prompt for technical explanation
    """
    prompt = f"""# Technical Analysis of Parkinson's Disease Biomarkers

## Request for Detailed Explanation of Ocular and Genomic Metrics

Please provide a technical explanation of the following metrics and their specific relationship to Parkinson's disease pathophysiology:

### Ocular Biomarkers

1. **Fixation Stability**: Measured value: {eye_metrics.get('fixation_stability', 'Unknown')}
   - Explain the neurological basis of fixation instability in PD
   - Describe the role of the superior colliculus and basal ganglia in fixation control
   - Detail how dopaminergic deficiency affects this metric

2. **Saccade Velocity**: Measured value: {eye_metrics.get('avg_saccade_velocity', 'Unknown')}°/s
   - Outline the neural circuitry involved in saccade generation
   - Explain why saccadic velocity specifically decreases in PD
   - Describe the physiological differences between horizontal and vertical saccades in PD

3. **Blink Rate**: Measured value: {eye_metrics.get('blink_rate', 'Unknown')} blinks/minute
   - Detail the relationship between dopamine levels and spontaneous blink rate
   - Explain the paradoxical findings of both increased and decreased blink rates in different PD stages

4. **Eye Aspect Ratio**: Measured value: {eye_metrics.get('avg_ear', 'Unknown')}
   - Explain how this metric relates to facial masking in PD
   - Describe its neurophysiological basis
"""

    if genomic_data and genomic_data.get('patient_variants'):
        prompt += """
### Genomic Markers
"""
        for gene, data in genomic_data.get('patient_variants', {}).items():
            prompt += f"""
1. **{gene} - {data.get('variant', 'Unknown variant')}**
   - Explain the molecular function of this gene and how mutations affect neuronal health
   - Describe the pathophysiological mechanism leading to increased PD risk
   - Detail the typical clinical phenotype associated with this variant
   - Provide penetrance data and age-dependent risk profiles
"""

    prompt += """
### Integration of Multimodal Biomarkers

Please explain how these ocular and genomic biomarkers can be integrated to form a more complete assessment, including:

1. The relative diagnostic weight that should be given to each biomarker
2. How these biomarkers might interact (e.g., specific genomic variants correlating with particular ocular findings)
3. The scientific basis for using these combined markers for pre-symptomatic detection

Please include relevant references to recent research where appropriate.
"""
    return prompt


def patient_friendly_explanation_prompt(patient_data, risk_level):
    """
    Generate a patient-friendly explanation of findings.
    
    Args:
        patient_data (dict): Patient demographic information
        risk_level (float): The calculated risk level (0-1)
        
    Returns:
        str: Formatted prompt for patient explanation
    """
    risk_category = "low"
    if risk_level > 0.3:
        risk_category = "moderate"
    if risk_level > 0.7:
        risk_category = "high"
        
    prompt = f"""# Patient-Friendly Explanation Request

## Background
I need to explain eye tracking and genetic test results to {patient_data.get('name', 'a patient')} who is {patient_data.get('age', 'an adult')} years old. Their test results show a {risk_category} risk level ({risk_level*100:.1f}%) for early Parkinson's disease based on eye movement patterns.

## Request
Please create a clear, compassionate explanation that:

1. Explains what the eye tracking test measures in simple terms
2. Describes how eye movements can reveal early brain changes before other symptoms appear
3. Puts the {risk_category} risk level in context without causing unnecessary alarm
4. Outlines practical next steps and what they might expect

Use plain language, avoid jargon, use helpful analogies, and maintain a hopeful but honest tone. Focus on empowering the patient with knowledge rather than causing anxiety.
"""
    return prompt


def follow_up_protocol_prompt(eye_metrics, risk_level):
    """
    Generate a follow-up protocol based on risk level.
    
    Args:
        eye_metrics (dict): Ocular biomarkers and measurements
        risk_level (float): The calculated risk level (0-1)
        
    Returns:
        str: Formatted prompt for follow-up protocol
    """
    prompt = f"""# Parkinson's Disease Follow-Up Protocol Request

## Patient Risk Profile
The patient has undergone ocular biomarker screening for pre-symptomatic Parkinson's disease with the following results:

- **Overall Risk Level**: {risk_level*100:.1f}%
- **Fixation Stability**: {eye_metrics.get('fixation_stability', 'Unknown')}
- **Saccade Velocity**: {eye_metrics.get('avg_saccade_velocity', 'Unknown')}°/s

## Request
Please create a detailed follow-up protocol appropriate for this risk level, including:

1. Recommended monitoring schedule (frequency of repeat eye tracking assessments)
2. Additional diagnostic evaluations to consider (neuroimaging, neuropsychological testing, etc.)
3. Appropriate specialist referrals based on this risk profile
4. Early intervention strategies that might be discussed
5. Key warning signs that would trigger more immediate medical attention

This protocol should be evidence-based and aligned with current best practices for pre-symptomatic Parkinson's disease monitoring.
"""
    return prompt


def full_diagnostic_report_prompt(patient_data, eye_metrics, genomic_data=None, tracking_history=None):
    """
    Generate a comprehensive diagnostic report for medical records.
    
    Args:
        patient_data (dict): Patient information
        eye_metrics (dict): Current ocular metrics
        genomic_data (dict, optional): Genomic findings
        tracking_history (list, optional): Historical tracking data
        
    Returns:
        str: Formatted prompt for diagnostic report
    """
    prompt = f"""# Comprehensive Diagnostic Report Generation

## Patient Information
- Name: {patient_data.get('name', 'Anonymous')}
- ID: {patient_data.get('id', 'Unknown')}
- DOB: {patient_data.get('dob', 'Unknown')}
- Sex: {patient_data.get('sex', 'Unknown')}
- Referral Source: {patient_data.get('referral', 'Unknown')}
- Assessment Date: {patient_data.get('assessment_date', 'Current')}

## Ocular Biomarker Assessment
"""

    for key, value in eye_metrics.items():
        if isinstance(value, (int, float)):
            prompt += f"- {key}: {value:.4f}\n"
        else:
            prompt += f"- {key}: {value}\n"

    # Add trending data if available
    if tracking_history and len(tracking_history) > 1:
        prompt += "\n## Longitudinal Tracking Data\n"
        prompt += "The following metrics have been tracked over time:\n\n"
        
        # Get first and most recent measurements for key metrics
        first = tracking_history[0]
        latest = tracking_history[-1]
        
        for metric in ['fixation_stability', 'avg_saccade_velocity']:
            if metric in first and metric in latest:
                change = latest.get(metric, 0) - first.get(metric, 0)
                prompt += f"- {metric}: Changed from {first.get(metric, 0):.4f} to {latest.get(metric, 0):.4f} ({change:.4f} change)\n"

    if genomic_data:
        prompt += "\n## Genomic Findings\n"
        for gene, data in genomic_data.get('patient_variants', {}).items():
            prompt += f"- {gene}: {data.get('variant', 'Unknown')} variant detected\n"
        
    prompt += """
## Report Request

Please generate a formal diagnostic report that includes:

1. **Executive Summary**: A concise overview of findings and risk assessment
2. **Methodology**: Brief description of eye tracking and genomic analysis techniques used
3. **Results**: Detailed interpretation of all metrics, including normative comparisons
4. **Clinical Impression**: Diagnostic impression and risk stratification
5. **Recommendations**: Specific next steps, referrals, and follow-up timeline
6. **Appendices**: Technical data for specialist review

The report should be structured as a formal medical document suitable for inclusion in the patient's medical record and sharing with referring physicians.
"""
    return prompt


# Dictionary mapping prompt types to their respective functions
PROMPT_TEMPLATES = {
    'clinical_assessment': clinical_assessment_prompt,
    'technical_explanation': technical_explanation_prompt,
    'patient_friendly': patient_friendly_explanation_prompt,
    'follow_up_protocol': follow_up_protocol_prompt,
    'diagnostic_report': full_diagnostic_report_prompt
}
