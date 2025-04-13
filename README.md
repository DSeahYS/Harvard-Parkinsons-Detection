# GenomeGuard: Early Parkinson's Detection System
## Multi-modal diagnostic platform combining eye movement analysis and genomic simulation for pre-symptomatic Parkinson's detection (revamp in progress)

## Table of Contents
Key Innovations
System Architecture
Features
Tech Stack
Setup & Deployment
Clinical Validation
Contributors
License

## Key Innovations 🧠
To-Be First Singapore-optimized pre-symptomatic Parkinson's screening combining:
MediaPipe Face Mesh for precision eye movement tracking
Ethnicity-specific analysis (Chinese, Malay, Indian) for Singapore population
BioNeMo Evo-2 genomic simulation with ethnicity-adjusted risk profiles
DeepSeek V3 LLM integration for clinical assessment
Detects neurological markers 5-7 years before motor symptoms appear

## System Architecture 🖥️
    A[Webcam Input] --> B[EyeTracker]
    B --> C[Eye Metrics Collection]
    C --> D[ParkinsonsDetector]
    D --> E[Risk Assessment]
    E --> F[BioNeMoSimulator]
    F --> G[Combined Analysis]
    G --> H[OpenRouter LLM Analysis]
    H --> I[Dashboard]

## Flowchart
![Flowchart](https://github.com/user-attachments/assets/a318a3d6-7e4e-4aee-a0c7-3bb498472d83)

## Features ✨
Component	Capabilities
Dashboard     All in 1 UI. (Currently broken)
Eye Tracking	Real-time measurement of saccadic velocity (400-600°/s norm), fixation stability, vertical saccades with ethnicity compensation
Genomic Simulation	Singapore-specific simulation of LRRK2, GBA, and SNCA variants with ethnicity adjustments for Chinese, Malay, and Indian populations
Medication Detection	Detects Levodopa efficacy through pupillary hippus analysis
Risk Assessment	Multi-modal scoring system with eye metrics (60%) and genomic factors (40%)
Clinical Assessment	DeepSeek V3 integration for comprehensive analysis and recommendations

## Tech Stack 🔧
Category	Technologies
Core	Python 3.12, OpenCV, MediaPipe Face Mesh
Data Analysis	NumPy, SciPy
UI	Tkinter with custom visualization
Storage	SQLite, JSON serialization
LLM Integration	OpenRouter API (DeepSeek V3)
Simulation	Custom BioNeMo Evo-2 simulation engine

# Run application
python main.py
Clinical Validation 📊 (Simulated)
Ethnicity-specific thresholds validated using Singapore population data
Eye tracking metrics correlated with UPDRS-III scores
Genomic simulation based on Singapore-specific variant prevalence
Risk scoring system aligns with MOH clinical guidelines

# Example (Old):
 ![image](https://github.com/user-attachments/assets/564ab41c-1dce-4eb8-9040-8ea24d06115e)
 ![image](https://github.com/user-attachments/assets/cc8359d9-1459-4811-b085-b6fd85761473)
 ![image](https://github.com/user-attachments/assets/f181eb41-6635-4a95-9160-773c3617ae24)

# Example (Newest)
![image](https://github.com/user-attachments/assets/7920eb11-a51f-49ec-a7fb-bd8a4d848a4c)
![image](https://github.com/user-attachments/assets/f863a320-1c90-4697-8756-257e028361f0)
![image](https://github.com/user-attachments/assets/f96fe56f-b0b4-4c7e-b134-137767e4dee2)
![image](https://github.com/user-attachments/assets/7b02659c-23df-429d-a9aa-6968e3f19d15)
![image](https://github.com/user-attachments/assets/dd934b22-2ba5-46c8-a395-72e0b8daf5a0)
![image](https://github.com/user-attachments/assets/b0a3c9c8-a928-401e-9bd2-8c60ae8bdb3e)

## Contributors 🤝
Team Lead: Dave Seah Yong Sheng
Physical Prototype: Timothy Wong
Bio and Medical Portions: Timothy Chua

License 📄
Apache 2.0 License - See LICENSE for details

Last Updated: April 4, 2025
Made with ❤️ for Singapore
