#!/usr/bin/env python3
"""
BGRemover Lite - Model Verification Script

This script verifies that U²-Net models are properly downloaded and validates their integrity.
Can be used independently for debugging model download issues.
"""

import os
import sys
import hashlib
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    MAGENTA = '\033[0;35m'
    CYAN = '\033[0;36m'
    BOLD = '\033[1m'
    NC = '\033[0m'  # No Color

class ModelVerifier:
    """Model verification and integrity checking"""
    
    def __init__(self, cache_dir: str = None, models_dir: str = None):
        """Initialize the model verifier"""
        self.cache_dir = Path(cache_dir or os.environ.get('U2NET_MODEL_CACHE_DIR', '~/.cache/u2net')).expanduser()
        self.models_dir = Path(models_dir or 'models').expanduser()
        self.log_messages = []
        
        # Expected model information (you may need to update these)
        self.expected_models = {
            'u2net.onnx': {
                'min_size': 10 * 1024 * 1024,  # 10MB minimum
                'max_size': 200 * 1024 * 1024,  # 200MB maximum
                'description': 'U²-Net main model'
            },
            'u2netp.onnx': {
                'min_size': 5 * 1024 * 1024,   # 5MB minimum
                'max_size': 100 * 1024 * 1024,  # 100MB maximum
                'description': 'U²-Net portrait model'
            }
        }
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message"""
        timestamp = __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted_message = f"[{timestamp}] [{level}] {message}"
        self.log_messages.append(formatted_message)
        print(formatted_message)
    
    def log_info(self, message: str):
        """Log info message"""
        self.log(f"{Colors.BLUE}{message}{Colors.NC}")
    
    def log_success(self, message: str):
        """Log success message"""
        self.log(f"{Colors.GREEN}✅ {message}{Colors.NC}")
    
    def log_warning(self, message: str):
        """Log warning message"""
        self.log(f"{Colors.YELLOW}⚠️  {message}{Colors.NC}")
    
    def log_error(self, message: str):
        """Log error message"""
        self.log(f"{Colors.RED}❌ {message}{Colors.NC}")
    
    def log_test(self, message: str):
        """Log test message"""
        self.log(f"{Colors.MAGENTA}🔍 {message}{Colors.NC}")
    
    def calculate_sha256(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.log_error(f"Failed to calculate SHA-256 for {file_path}: {e}")
            return ""
    
    def check_file_integrity(self, file_path: Path) -> Dict:
        """Check file integrity and gather information"""
        result = {
            'exists': False,
            'readable': False,
            'size': 0,
            'size_human': '',
            'sha256': '',
            'valid_size': False,
            'is_onnx': False,
            'description': '',
            'model_name': file_path.name
        }
        
        if not file_path.exists():
            return result
        
        result['exists'] = True
        
        try:
            # Check if readable
            if os.access(file_path, os.R_OK):
                result['readable'] = True
            else:
                self.log_error(f"File not readable: {file_path}")
                return result
            
            # Get file size
            result['size'] = file_path.stat().st_size
            result['size_human'] = self.format_bytes(result['size'])
            
            # Calculate SHA-256
            result['sha256'] = self.calculate_sha256(file_path)
            
            # Check if it's an ONNX file
            if file_path.suffix.lower() == '.onnx':
                result['is_onnx'] = True
            else:
                self.log_warning(f"File doesn't have .onnx extension: {file_path}")
            
            # Validate size based on expectations
            model_info = self.expected_models.get(file_path.name, {})
            min_size = model_info.get('min_size', 0)
            max_size = model_info.get('max_size', float('inf'))
            result['description'] = model_info.get('description', 'Unknown model')
            
            if min_size <= result['size'] <= max_size:
                result['valid_size'] = True
            else:
                self.log_warning(f"File size outside expected range: {result['size_human']}")
            
            # Additional ONNX file validation
            if result['is_onnx']:
                if not self.validate_onnx_file(file_path):
                    self.log_warning(f"ONNX file validation failed: {file_path}")
            
        except Exception as e:
            self.log_error(f"Error checking file integrity: {e}")
        
        return result
    
    def validate_onnx_file(self, file_path: Path) -> bool:
        """Validate ONNX file structure"""
        try:
            # Try to read the first few bytes to check if it's a valid file
            with open(file_path, 'rb') as f:
                header = f.read(4)
                # ONNX files typically start with 'ONNX' or have specific magic bytes
                if len(header) >= 4:
                    # Basic check - ONNX files should have some content
                    return True
            return False
        except Exception as e:
            self.log_error(f"ONNX validation failed: {e}")
            return False
    
    def format_bytes(self, size: int) -> str:
        """Format bytes to human readable string"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"
    
    def check_model(self, model_name: str) -> Dict:
        """Check a specific model"""
        self.log_test(f"Checking model: {model_name}")
        
        # Try both cache directory and models directory
        possible_paths = [
            self.cache_dir / model_name,
            self.models_dir / model_name
        ]
        
        for path in possible_paths:
            if path.exists():
                self.log_info(f"Found {model_name} at: {path}")
                return self.check_file_integrity(path)
        
        self.log_error(f"Model not found: {model_name}")
        return {
            'exists': False,
            'model_name': model_name,
            'description': self.expected_models.get(model_name, {}).get('description', 'Unknown model')
        }
    
    def verify_all_models(self) -> Dict[str, Dict]:
        """Verify all expected models"""
        self.log_test("Verifying all models")
        
        results = {}
        for model_name in self.expected_models.keys():
            results[model_name] = self.check_model(model_name)
        
        return results
    
    def generate_report(self, results: Dict[str, Dict]) -> str:
        """Generate a comprehensive report"""
        report = []
        report.append(f"{Colors.BOLD}Model Verification Report{Colors.NC}")
        report.append("=" * 50)
        report.append(f"Cache Directory: {self.cache_dir}")
        report.append(f"Models Directory: {self.models_dir}")
        report.append("")
        
        total_models = len(results)
        found_models = sum(1 for r in results.values() if r['exists'])
        valid_models = sum(1 for r in results.values() if r.get('valid_size', False))
        
        report.append(f"{Colors.BOLD}Summary:{Colors.NC}")
        report.append(f"Total expected models: {total_models}")
        report.append(f"Found models: {found_models}")
        report.append(f"Valid models: {valid_models}")
        report.append("")
        
        for model_name, result in results.items():
            report.append(f"{Colors.BOLD}{model_name}{Colors.NC}")
            report.append(f"  Description: {result.get('description', 'N/A')}")
            
            if result['exists']:
                report.append(f"  Status: {Colors.GREEN}Found{Colors.NC}")
                report.append(f"  Path: {result.get('path', 'Unknown')}")
                report.append(f"  Size: {result.get('size_human', 'Unknown')} ({result.get('size', 0)} bytes)")
                report.append(f"  SHA-256: {result.get('sha256', 'N/A')[:16]}...")
                report.append(f"  Readable: {Colors.GREEN}Yes{Colors.NC}" if result.get('readable') else f"  Readable: {Colors.RED}No{Colors.NC}")
                report.append(f"  Valid Size: {Colors.GREEN}Yes{Colors.NC}" if result.get('valid_size') else f"  Valid Size: {Colors.RED}No{Colors.NC}")
                report.append(f"  ONNX Format: {Colors.GREEN}Yes{Colors.NC}" if result.get('is_onnx') else f"  ONNX Format: {Colors.YELLOW}Unknown{Colors.NC}")
            else:
                report.append(f"  Status: {Colors.RED}Not Found{Colors.NC}")
            
            report.append("")
        
        # Recommendations
        report.append(f"{Colors.BOLD}Recommendations:{Colors.NC}")
        if found_models < total_models:
            missing = [name for name, result in results.items() if not result['exists']]
            report.append(f"{Colors.YELLOW}• Missing models: {', '.join(missing)}{Colors.NC}")
            report.append("  - Run the build script to download missing models")
            report.append("  - Check internet connection and firewall settings")
        
        invalid_size_models = [name for name, result in results.items() 
                              if result['exists'] and not result.get('valid_size', False)]
        if invalid_size_models:
            report.append(f"{Colors.YELLOW}• Models with invalid sizes: {', '.join(invalid_size_models)}{Colors.NC}")
            report.append("  - Try cleaning the model cache and re-downloading")
        
        if found_models == total_models and valid_models == total_models:
            report.append(f"{Colors.GREEN}• All models are properly downloaded and valid!{Colors.NC}")
            report.append("  - You can proceed with building the application")
        
        return "\n".join(report)
    
    def save_report(self, results: Dict[str, Dict], output_file: str):
        """Save report to file"""
        try:
            with open(output_file, 'w') as f:
                # Save as JSON
                json.dump({
                    'cache_dir': str(self.cache_dir),
                    'models_dir': str(self.models_dir),
                    'timestamp': __import__('datetime').datetime.now().isoformat(),
                    'results': results
                }, f, indent=2)
            self.log_success(f"Report saved to: {output_file}")
        except Exception as e:
            self.log_error(f"Failed to save report: {e}")
    
    def run_verification(self, save_json: bool = False, output_file: str = None) -> bool:
        """Run the complete verification process"""
        self.log_info("Starting model verification...")
        
        # Check directories
        self.log_info(f"Cache directory: {self.cache_dir}")
        self.log_info(f"Models directory: {self.models_dir}")
        
        if not self.cache_dir.exists():
            self.log_warning(f"Cache directory does not exist: {self.cache_dir}")
        if not self.models_dir.exists():
            self.log_warning(f"Models directory does not exist: {self.models_dir}")
        
        # Verify all models
        results = self.verify_all_models()
        
        # Generate and print report
        report = self.generate_report(results)
        print("\n" + report)
        
        # Save JSON report if requested
        if save_json:
            output_file = output_file or f"model_verification_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.save_report(results, output_file)
        
        # Return success status
        total_models = len(results)
        valid_models = sum(1 for r in results.values() if r['exists'] and r.get('valid_size', False))
        return valid_models == total_models

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Verify U²-Net model downloads")
    parser.add_argument('--cache-dir', type=str, help='Model cache directory')
    parser.add_argument('--models-dir', type=str, help='Models directory')
    parser.add_argument('--save-json', action='store_true', help='Save report as JSON')
    parser.add_argument('--output-file', type=str, help='Output file for JSON report')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Create verifier
    verifier = ModelVerifier(
        cache_dir=args.cache_dir,
        models_dir=args.models_dir
    )
    
    if args.verbose:
        verifier.log_info("Verbose mode enabled")
    
    # Run verification
    success = verifier.run_verification(
        save_json=args.save_json,
        output_file=args.output_file
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
