# MIMIC-IV-Ext-iDS

**Status:** Under review on [PhysioNet](https://physionet.org); the dataset will be published upon completion of the review.

MIMIC-IV-Ext-iDS is a context-enriched dataset built for clinical text generation. It contains 317,598 de-identified discharge summaries from MIMIC-IV, each segmented into 31 structured fields using the DeepSeek-LLM 67B model. Segmentation outputs were validated with rule-based checks and cross-referenced with MIMIC-IV and MIMIC-IV-ED tables to ensure high fidelity.

## Field Groups
The 31 fields are organized into seven categories:

- **Identifiers:** `note_id`, `subject_id`, `hadm_id`
- **Patient Demographics:** `age_at_charttime`, `sex`, `race`
- **Admission Information:** `admission_type`, `admission_location`, `arrival_transport`, `service`, `insurance`
- **Clinical History & Initial Presentation:** `allergies`, `chief_complaint`, `history_present_illness`, `past_medical_history`, `social_family_history`, `medications_admission`
- **Diagnostic Findings:** `physical_examination`, `imaging_diagnostics`, `laboratory_results`, `pathology_biopsy_results`
- **Hospitalization Course & Interventions:** `major_procedures`, `brief_hospital_course`, `acute_issues`, `chronic_issues`
- **Discharge Summary & Planning:** `discharge_medications`, `discharge_diagnosis`, `discharge_condition`, `discharge_disposition`, `discharge_instructions`, `followup_instructions`

## Data Format
The dataset is distributed as `MIMIC-IV-Ext-iDS.csv` (~2.8 GB). Each row represents a single clinical encounter and includes both the structured fields above and the original discharge note text. Access requires PhysioNet credentialed approval and adherence to the corresponding data use agreement.

## Intended Use
MIMIC-IV-Ext-iDS supports tasks such as structure-to-text generation, clinical summarization, and fine-tuning large language models for healthcare applications.

## Contact
For questions or collaboration requests, please contact Qingyang Shen (<jackflynn@stu.scu.edu.cn>).

