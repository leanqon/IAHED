import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import existing modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')
import parameters
from parameters import *
from encoder import *
from losses import *
from main_model import *
from train import *

class EICUCrossDomainTrainer:
    """
    Cross-domain trainer for evaluating IAHED model generalizability using eICU data.
    """

    def __init__(self, mimic_model_path, eicu_data_path, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.mimic_model_path = mimic_model_path
        self.eicu_data_path = eicu_data_path
        self.results = {}

    def load_mimic_model(self):
        """Load pre-trained MIMIC-IV model."""
        print("Loading pre-trained MIMIC-IV model...")
        checkpoint = torch.load(self.mimic_model_path, map_location=self.device)

        # Extract model architecture parameters from checkpoint
        model_config = checkpoint.get('config', {})

        # Initialize model with same architecture
        self.model = IAHED(
            device=self.device,
            input_dim=model_config.get('input_dim', 128),
            hidden_dim=model_config.get('hidden_dim', 256),
            output_dim=model_config.get('output_dim', 32)
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        print("MIMIC-IV model loaded successfully")

    def load_eicu_data(self):
        """Load processed eICU data."""
        print("Loading eICU data...")

        # Load eICU data dictionaries
        with open(f"{self.eicu_data_path}/dict_eicu/dataDic", 'rb') as f:
            self.eicu_data = pickle.load(f)

        with open(f"{self.eicu_data_path}/dict_eicu/hadmDic", 'rb') as f:
            self.eicu_ids = pickle.load(f)

        with open(f"{self.eicu_data_path}/dict_eicu/metaDic", 'rb') as f:
            self.eicu_meta = pickle.load(f)

        # Load labels
        labels_df = pd.read_csv(f"{self.eicu_data_path}/csv_eicu/labels.csv")
        self.eicu_labels = dict(zip(labels_df['stay_id'], labels_df['icu_mdr']))

        print(f"Loaded {len(self.eicu_ids)} eICU patient stays")
        print(f"Positive cases: {sum(self.eicu_labels.values())}")
        print(f"Class imbalance ratio: {sum(self.eicu_labels.values())/len(self.eicu_labels):.4f}")

    def prepare_eicu_features(self, stay_ids):
        """Prepare eICU features in format compatible with MIMIC-IV model."""
        features = []
        labels = []

        for stay_id in tqdm(stay_ids, desc="Preparing eICU features"):
            try:
                # Load dynamic features
                dyn_path = f"{self.eicu_data_path}/csv_eicu/{stay_id}/dynamic.csv"
                if os.path.exists(dyn_path):
                    dyn_data = pd.read_csv(dyn_path)
                else:
                    continue

                # Load static features
                static_path = f"{self.eicu_data_path}/csv_eicu/{stay_id}/static.csv"
                if os.path.exists(static_path):
                    static_data = pd.read_csv(static_path)
                else:
                    continue

                # Load demographics
                demo_path = f"{self.eicu_data_path}/csv_eicu/{stay_id}/demo.csv"
                if os.path.exists(demo_path):
                    demo_data = pd.read_csv(demo_path)
                else:
                    continue

                # Combine features
                combined_features = self._combine_eicu_features(dyn_data, static_data, demo_data)

                features.append(combined_features)
                labels.append(self.eicu_labels[stay_id])

            except Exception as e:
                print(f"Error processing stay_id {stay_id}: {e}")
                continue

        return np.array(features), np.array(labels)

    def _combine_eicu_features(self, dynamic, static, demo):
        """Combine eICU features into format expected by IAHED model."""
        # Normalize dynamic features
        dynamic_norm = (dynamic - dynamic.mean()) / (dynamic.std() + 1e-8)

        # Process demographics
        age_norm = (demo['Age'].iloc[0] - 65) / 25  # Normalize around typical ICU age
        gender_enc = 1 if demo['gender'].iloc[0] == 'Male' else 0

        # Combine all features
        temporal_features = dynamic_norm.values.flatten()
        static_features = static.values.flatten()
        demo_features = np.array([age_norm, gender_enc])

        # Pad or truncate to consistent length
        max_temporal = 1000  # Adjust based on your data
        max_static = 500

        if len(temporal_features) > max_temporal:
            temporal_features = temporal_features[:max_temporal]
        else:
            temporal_features = np.pad(temporal_features, (0, max_temporal - len(temporal_features)))

        if len(static_features) > max_static:
            static_features = static_features[:max_static]
        else:
            static_features = np.pad(static_features, (0, max_static - len(static_features)))

        combined = np.concatenate([temporal_features, static_features, demo_features])
        return combined

    def evaluate_zero_shot(self):
        """Evaluate pre-trained MIMIC-IV model on eICU data without fine-tuning."""
        print("Performing zero-shot evaluation on eICU data...")

        # Prepare test data
        test_features, test_labels = self.prepare_eicu_features(self.eicu_ids)

        # Convert to tensors
        test_features = torch.FloatTensor(test_features).to(self.device)
        test_labels = torch.FloatTensor(test_labels).to(self.device)

        # Evaluate
        self.model.eval()
        with torch.no_grad():
            outputs, _, _ = self.model(test_features, test_labels)
            predictions = outputs.cpu().numpy().flatten()

        # Calculate metrics
        auc_score = roc_auc_score(test_labels.cpu().numpy(), predictions)
        ap_score = average_precision_score(test_labels.cpu().numpy(), predictions)

        self.results['zero_shot'] = {
            'auc': auc_score,
            'ap': ap_score,
            'predictions': predictions,
            'labels': test_labels.cpu().numpy()
        }

        print(f"Zero-shot AUC: {auc_score:.4f}")
        print(f"Zero-shot AP: {ap_score:.4f}")

    def fine_tune_on_eicu(self, num_epochs=50, learning_rate=0.0001):
        """Fine-tune MIMIC-IV model on eICU training data."""
        print("Fine-tuning model on eICU data...")

        # Split eICU data
        train_ids, test_ids = train_test_split(
            self.eicu_ids,
            test_size=0.2,
            stratify=[self.eicu_labels[id] for id in self.eicu_ids],
            random_state=42
        )

        train_features, train_labels = self.prepare_eicu_features(train_ids)
        test_features, test_labels = self.prepare_eicu_features(test_ids)

        # Convert to tensors
        train_features = torch.FloatTensor(train_features).to(self.device)
        train_labels = torch.FloatTensor(train_labels).to(self.device)
        test_features = torch.FloatTensor(test_features).to(self.device)
        test_labels = torch.FloatTensor(test_labels).to(self.device)

        # Setup optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = ClassBalancedFocalLoss(alpha=2, gamma=2)

        # Training loop
        best_auc = 0
        train_losses = []
        val_aucs = []

        for epoch in tqdm(range(num_epochs), desc="Fine-tuning"):
            # Training
            self.model.train()
            optimizer.zero_grad()

            outputs, contrastive_loss, _ = self.model(train_features, train_labels)
            focal_loss = criterion(outputs, train_labels.unsqueeze(1))
            total_loss = focal_loss + 0.3 * contrastive_loss

            total_loss.backward()
            optimizer.step()
            train_losses.append(total_loss.item())

            # Validation
            if epoch % 5 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_outputs, _, _ = self.model(test_features, test_labels)
                    val_predictions = val_outputs.cpu().numpy().flatten()
                    val_auc = roc_auc_score(test_labels.cpu().numpy(), val_predictions)
                    val_aucs.append(val_auc)

                    if val_auc > best_auc:
                        best_auc = val_auc
                        best_predictions = val_predictions

        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            final_outputs, _, _ = self.model(test_features, test_labels)
            final_predictions = final_outputs.cpu().numpy().flatten()

        final_auc = roc_auc_score(test_labels.cpu().numpy(), final_predictions)
        final_ap = average_precision_score(test_labels.cpu().numpy(), final_predictions)

        self.results['fine_tuned'] = {
            'auc': final_auc,
            'ap': final_ap,
            'predictions': final_predictions,
            'labels': test_labels.cpu().numpy(),
            'train_losses': train_losses,
            'val_aucs': val_aucs
        }

        print(f"Fine-tuned AUC: {final_auc:.4f}")
        print(f"Fine-tuned AP: {final_ap:.4f}")

    def domain_adaptation_analysis(self):
        """Analyze domain differences between MIMIC-IV and eICU."""
        print("Performing domain adaptation analysis...")

        # Load MIMIC-IV data for comparison
        try:
            with open("./data/dict/dataDic", 'rb') as f:
                mimic_data = pickle.load(f)
            with open("./data/dict/hadmDic", 'rb') as f:
                mimic_ids = pickle.load(f)

            # Sample subset for comparison
            sample_mimic_ids = np.random.choice(mimic_ids, min(1000, len(mimic_ids)), replace=False)
            sample_eicu_ids = np.random.choice(self.eicu_ids, min(1000, len(self.eicu_ids)), replace=False)

            # Extract demographic distributions
            mimic_ages = [mimic_data[id]['age'] for id in sample_mimic_ids]
            eicu_ages = [self.eicu_data[id]['age'] for id in sample_eicu_ids]

            mimic_genders = [mimic_data[id]['gender'] for id in sample_mimic_ids]
            eicu_genders = [self.eicu_data[id]['gender'] for id in sample_eicu_ids]

            # Create comparison plots
            self._plot_domain_comparison(mimic_ages, eicu_ages, mimic_genders, eicu_genders)

        except Exception as e:
            print(f"Could not load MIMIC-IV data for comparison: {e}")

    def _plot_domain_comparison(self, mimic_ages, eicu_ages, mimic_genders, eicu_genders):
        """Create visualization comparing MIMIC-IV and eICU distributions."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Age distributions
        axes[0,0].hist(mimic_ages, alpha=0.7, label='MIMIC-IV', bins=20)
        axes[0,0].hist(eicu_ages, alpha=0.7, label='eICU', bins=20)
        axes[0,0].set_title('Age Distributions')
        axes[0,0].set_xlabel('Age')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].legend()

        # Gender distributions
        mimic_male_pct = sum(1 for g in mimic_genders if g == 'Male') / len(mimic_genders)
        eicu_male_pct = sum(1 for g in eicu_genders if g == 'Male') / len(eicu_genders)

        axes[0,1].bar(['MIMIC-IV', 'eICU'], [mimic_male_pct, eicu_male_pct])
        axes[0,1].set_title('Male Percentage')
        axes[0,1].set_ylabel('Proportion')

        # Performance comparison
        if 'zero_shot' in self.results and 'fine_tuned' in self.results:
            methods = ['Zero-shot', 'Fine-tuned']
            aucs = [self.results['zero_shot']['auc'], self.results['fine_tuned']['auc']]
            aps = [self.results['zero_shot']['ap'], self.results['fine_tuned']['ap']]

            axes[1,0].bar(methods, aucs)
            axes[1,0].set_title('AUC Comparison')
            axes[1,0].set_ylabel('AUC Score')

            axes[1,1].bar(methods, aps)
            axes[1,1].set_title('AP Comparison')
            axes[1,1].set_ylabel('AP Score')

        plt.tight_layout()
        plt.savefig('./results/eicu_domain_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_cross_domain_report(self):
        """Generate comprehensive cross-domain validation report."""
        print("Generating cross-domain validation report...")

        report = {
            'dataset_statistics': {
                'mimic_iv': {
                    'description': 'Single-center academic medical center (Beth Israel Deaconess Medical Center)',
                    'time_period': '2008-2019',
                    'patient_population': 'Urban academic medical center'
                },
                'eicu': {
                    'description': 'Multi-center collaborative database',
                    'time_period': '2014-2015',
                    'hospitals': 'Multiple hospitals across the United States',
                    'patient_count': len(self.eicu_ids),
                    'positive_cases': sum(self.eicu_labels.values()),
                    'class_imbalance': sum(self.eicu_labels.values())/len(self.eicu_labels)
                }
            },
            'performance_metrics': self.results,
            'domain_challenges': {
                'data_format_differences': [
                    'Different time encoding (minutes vs hours)',
                    'Different patient identifiers',
                    'Varying medication and procedure coding systems',
                    'Different demographic encoding'
                ],
                'clinical_practice_differences': [
                    'Multi-center vs single-center variation',
                    'Different documentation practices',
                    'Varying MDRB detection protocols',
                    'Different patient populations'
                ],
                'recommendations': [
                    'Domain adaptation techniques needed for optimal performance',
                    'Feature harmonization important for cross-domain generalization',
                    'Consider hospital-specific fine-tuning for deployment',
                    'Regular model recalibration recommended'
                ]
            }
        }

        # Save report
        os.makedirs('./results', exist_ok=True)
        with open('./results/eicu_cross_domain_report.json', 'w') as f:
            import json
            json.dump(report, f, indent=2, default=str)

        # Generate markdown report
        self._generate_markdown_report(report)

        return report

    def _generate_markdown_report(self, report):
        """Generate markdown version of the cross-domain report."""
        markdown = f"""
# eICU Cross-Domain Validation Report

## Dataset Comparison

### MIMIC-IV
- **Description**: {report['dataset_statistics']['mimic_iv']['description']}
- **Time Period**: {report['dataset_statistics']['mimic_iv']['time_period']}
- **Setting**: {report['dataset_statistics']['mimic_iv']['patient_population']}

### eICU-CRD
- **Description**: {report['dataset_statistics']['eicu']['description']}
- **Time Period**: {report['dataset_statistics']['eicu']['time_period']}
- **Hospitals**: {report['dataset_statistics']['eicu']['hospitals']}
- **Patient Count**: {report['dataset_statistics']['eicu']['patient_count']:,}
- **Positive Cases**: {report['dataset_statistics']['eicu']['positive_cases']}
- **Class Imbalance**: {report['dataset_statistics']['eicu']['class_imbalance']:.4f}

## Performance Results

"""
        if 'zero_shot' in self.results:
            markdown += f"""
### Zero-Shot Transfer (MIMIC-IV → eICU)
- **AUC**: {self.results['zero_shot']['auc']:.4f}
- **AP**: {self.results['zero_shot']['ap']:.4f}
"""

        if 'fine_tuned' in self.results:
            markdown += f"""
### Fine-Tuned Performance
- **AUC**: {self.results['fine_tuned']['auc']:.4f}
- **AP**: {self.results['fine_tuned']['ap']:.4f}
"""

        markdown += f"""
## Domain Adaptation Challenges

### Data Format Differences
"""
        for challenge in report['domain_challenges']['data_format_differences']:
            markdown += f"- {challenge}\n"

        markdown += f"""
### Clinical Practice Differences
"""
        for challenge in report['domain_challenges']['clinical_practice_differences']:
            markdown += f"- {challenge}\n"

        markdown += f"""
## Recommendations
"""
        for rec in report['domain_challenges']['recommendations']:
            markdown += f"- {rec}\n"

        with open('./results/eicu_cross_domain_report.md', 'w') as f:
            f.write(markdown)

def main():
    """Main execution function for eICU cross-domain validation."""

    # Initialize trainer
    trainer = EICUCrossDomainTrainer(
        mimic_model_path='./models/iahed_mimic_best.pth',  # Path to pre-trained model
        eicu_data_path='./data'  # Path to processed eICU data
    )

    # Load data and model
    trainer.load_eicu_data()
    trainer.load_mimic_model()

    # Perform evaluations
    trainer.evaluate_zero_shot()
    trainer.fine_tune_on_eicu()
    trainer.domain_adaptation_analysis()

    # Generate report
    report = trainer.generate_cross_domain_report()

    print("Cross-domain validation complete!")
    print("Results saved to ./results/")

if __name__ == "__main__":
    main()