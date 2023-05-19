# Zero-Shot BioNER
Zero-Shot and Few-Shot methods for NER in biomedical domain
# Dataset Conversion

Each of the datasets has been converted using a specific script to a format where the named entity (NE) has been transformed to 1, while everything else has been labeled as 0.

The conversion process can be implemented using the following steps:

1. Load the original dataset.
2. Define a conversion function that maps NE to 1 and other entities to 0.
3. Apply the conversion function to each dataset separately.
4. Merge the converted datasets into a single dataset.

## Converted datasets

#### Chemical NER 
- CHEMDNER 
- CDR-Chemical 

#### Disease NER 
- NCBI-Disease 
- CDR-Disease 

#### Gene/Protein NER 
- JNLPBA 

#### Drugs 
- n2c2/i2b2 

## Merged dataset is in table below
<img width="599" alt="objedinjeni_dataset" src="https://github.com/br-ai-ns-institute/Zero-ShotNER/assets/8451505/de4a9f46-f5f2-4574-aacc-0df3f3325990">
