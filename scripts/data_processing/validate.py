#!/usr/bin/env python3

"""
Data Validation Script for BEACON

This script validates data quality and structure.
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple

from beacon.data import DataValidator
from beacon.utils.metrics import Metrics

class DataValidationRunner:
    """Run data validation checks"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the validation runner.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.input_dir = Path(config['input_dir'])
        self.output_dir = Path(config.get('output_dir', 'validation_results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.validator = DataValidator(config.get('validator_config', {}))
        self.results = []
        self.errors = []

    def validate_clinical_data(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Validate clinical data.

        Args:
            file_path: Path to clinical data file

        Returns:
            Tuple of (is_valid, error_messages)
        """
        try:
            data = pd.read_csv(file_path)
            
            # Basic validation
            is_valid, messages = self.validator.validate(data)
            if not is_valid:
                return False, messages

            # Additional checks
            errors = []
            
            # Check required columns
            required_columns = ['patient_id', 'age', 'sex', 'diagnosis']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")

            # Check data types
            if not pd.to_numeric(data['age'], errors='coerce').notnull().all():
                errors.append("Invalid age values")

            if not data['sex'].isin(['M', 'F']).all():
                errors.append("Invalid sex values")

            # Check value ranges
            if (data['age'] < 0).any() or (data['age'] > 120).any():
                errors.append("Age values out of range")

            # Check duplicates
            if data['patient_id'].duplicated().any():
                errors.append("Duplicate patient IDs found")

            return len(errors) == 0, errors

        except Exception as e:
            return False, [f"Error processing clinical data: {str(e)}"]

    def validate_imaging_data(self, directory: Path) -> Tuple[bool, List[str]]:
        """Validate imaging data.

        Args:
            directory: Path to imaging data directory

        Returns:
            Tuple of (is_valid, error_messages)
        """
        try:
            errors = []
            
            # Check directory exists
            if not directory.exists():
                return False, ["Imaging directory not found"]

            # Check image files
            image_files = list(directory.glob('*.npy'))
            if not image_files:
                return False, ["No image files found"]

            # Validate each image
            for image_file in image_files:
                try:
                    image = np.load(image_file)
                    
                    # Check dimensions
                    if len(image.shape) != 3:  # (height, width, channels)
                        errors.append(f"Invalid image dimensions in {image_file.name}")
                    
                    # Check value range
                    if image.min() < 0 or image.max() > 255:
                        errors.append(f"Invalid pixel values in {image_file.name}")
                    
                    # Check for NaN/Inf
                    if not np.isfinite(image).all():
                        errors.append(f"Invalid values (NaN/Inf) in {image_file.name}")

                except Exception as e:
                    errors.append(f"Error loading {image_file.name}: {str(e)}")

            return len(errors) == 0, errors

        except Exception as e:
            return False, [f"Error processing imaging data: {str(e)}"]

    def validate_genomic_data(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Validate genomic data.

        Args:
            file_path: Path to genomic data file

        Returns:
            Tuple of (is_valid, error_messages)
        """
        try:
            data = np.load(file_path, allow_pickle=True)
            errors = []

            # Check required arrays
            required_arrays = ['expression', 'mutations', 'cnv']
            missing_arrays = [arr for arr in required_arrays if arr not in data]
            if missing_arrays:
                errors.append(f"Missing required arrays: {missing_arrays}")

            # Validate expression data
            if 'expression' in data:
                expression = data['expression']
                if not np.isfinite(expression).all():
                    errors.append("Invalid values in expression data")
                if expression.min() < -10 or expression.max() > 10:
                    errors.append("Expression values out of expected range")

            # Validate mutation data
            if 'mutations' in data:
                mutations = data['mutations']
                if not np.isin(mutations, [0, 1]).all():
                    errors.append("Invalid mutation values (should be binary)")

            # Validate CNV data
            if 'cnv' in data:
                cnv = data['cnv']
                if not np.isin(cnv, [-2, -1, 0, 1, 2]).all():
                    errors.append("Invalid CNV values")

            return len(errors) == 0, errors

        except Exception as e:
            return False, [f"Error processing genomic data: {str(e)}"]

    def run_validation(self) -> None:
        """Run validation on all data types."""
        # Validate clinical data
        clinical_file = self.input_dir / 'clinical.csv'
        if clinical_file.exists():
            is_valid, messages = self.validate_clinical_data(clinical_file)
            self.results.append({
                'data_type': 'clinical',
                'file': str(clinical_file),
                'is_valid': is_valid,
                'messages': messages
            })

        # Validate imaging data
        image_dir = self.input_dir / 'images'
        if image_dir.exists():
            is_valid, messages = self.validate_imaging_data(image_dir)
            self.results.append({
                'data_type': 'imaging',
                'file': str(image_dir),
                'is_valid': is_valid,
                'messages': messages
            })

        # Validate genomic data
        genomic_file = self.input_dir / 'genomic.npz'
        if genomic_file.exists():
            is_valid, messages = self.validate_genomic_data(genomic_file)
            self.results.append({
                'data_type': 'genomic',
                'file': str(genomic_file),
                'is_valid': is_valid,
                'messages': messages
            })

        # Save validation report
        self.save_report()

    def save_report(self) -> None:
        """Save validation results to a report."""
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / f'validation_report_{timestamp}.json'
        
        report = {
            'timestamp': timestamp,
            'input_directory': str(self.input_dir),
            'results': self.results,
            'summary': {
                'total_validations': len(self.results),
                'passed': sum(1 for r in self.results if r['is_valid']),
                'failed': sum(1 for r in self.results if not r['is_valid'])
            }
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)
        
        # Print summary
        print("\nValidation Summary:")
        print(f"Total validations: {report['summary']['total_validations']}")
        print(f"Passed: {report['summary']['passed']}")
        print(f"Failed: {report['summary']['failed']}")
        print(f"\nDetailed report saved to: {report_file}")
        
        # Exit with error if any validation failed
        if report['summary']['failed'] > 0:
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Validate data for BEACON")
    parser.add_argument("--input-dir", required=True, help="Input data directory")
    parser.add_argument("--output-dir", help="Output directory for validation results")
    parser.add_argument("--config", help="Configuration file path")
    args = parser.parse_args()

    # Load configuration
    config = {
        'input_dir': args.input_dir
    }
    if args.output_dir:
        config['output_dir'] = args.output_dir

    if args.config:
        import yaml
        with open(args.config) as f:
            config.update(yaml.safe_load(f))

    # Run validation
    validator = DataValidationRunner(config)
    validator.run_validation()

if __name__ == "__main__":
    main() 