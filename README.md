# IAHED: Imbalance-Adaptive Hybrid Encoder-Decoder

A deep learning framework for early detection of ICU-acquired Multidrug-Resistant Bacteria (MDRB)

## Overview

IAHED (Imbalance-Adaptive Hybrid Encoder-Decoder) is an innovative deep learning framework specifically designed to address the challenges of early detection of ICU-acquired Multidrug-Resistant Bacteria (MDRB). This framework effectively handles the extreme data imbalance in critical care settings while maintaining high precision and recall in predictions.

## Key Features

- **Dual-pathway architecture**: Integrates both dynamic and static patient data
- **Imbalance-adaptive design**: Specially tailored for the extreme class imbalance in ICU settings
- **Advanced data augmentation**: Utilizes Givens rotations on randomly cropped segments of time series data
- **Composite loss function**: Based on contrastive learning to improve minority class detection

## Model Architecture

IAHED employs a dual-pathway framework to process dynamic and static data streams:

1. **Dynamic Pathway**: Uses an encoder-decoder configuration with dilated convolutional layers to capture temporal dependencies
2. **Static Pathway**: Processes time-invariant features such as demographics and medical history
3. **Feature Fusion**: Combines dynamic and static features using an attention mechanism
4. **Pre-trained Class-Specific Weights**: Extracts nuanced, class-specific feature representations

The model's loss function consists of three components:
- Class Equilibrium Contrastive Loss (CECL)
- Multi-Scale Contrastive Loss (MTCL)
- Class-Balanced Focal Loss (CBFL)

## Dataset

The IAHED model was trained and evaluated using data from the MIMIC-IV version 2.2 database, which includes:

- ICU stays from unique patients admitted between 2008 and 2019
- Data types: laboratory measurements, medication administration, demographics, interventions, etc.

The study considered two cohorts:
- 7-day observation cohort
- 14-day observation cohort

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## License

This project is licensed under the MIT License - see the LICENSE file for details.
