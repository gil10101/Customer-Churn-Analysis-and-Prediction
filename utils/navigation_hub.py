"""
NavigationHub: Interactive notebook launcher with dependency checking and runtime estimation.
Provides comprehensive project navigation, environment validation, and execution management.
"""

import os
import sys
import subprocess
import time
import psutil
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import pandas as pd

from .config import CONFIG, NotebookConfig
from .logging_setup import setup_notebook_logging, NotebookLogger
from .runtime_monitor import RuntimeEstimator, DependencyChecker, create_runtime_estimator, create_dependency_checker

@dataclass
class NotebookMetadata:
    """Comprehensive metadata for notebook management."""
    filename: str
    title: str
    description: str
    estimated_runtime_minutes: Tuple[int, int]  # (min, max)
    prerequisites: List[str]
    key_outputs: List[str]
    dependencies: List[str] = field(default_factory=list)
    memory_requirement_gb: float = 2.0
    cpu_intensive: bool = False
    gpu_recommended: bool = False
    output_artifacts: List[str] = field(default_factory=list)
    
    @property
    def estimated_runtime_str(self) -> str:
        """Get formatted runtime estimate."""
        min_time, max_time = self.estimated_runtime_minutes
        if min_time == max_time:
            return f"{min_time} minutes"
        return f"{min_time}-{max_time} minutes"

@dataclass
class SystemRequirements:
    """System requirements and validation results."""
    python_version: Tuple[int, int, int]
    memory_gb: float
    cpu_cores: int
    gpu_available: bool
    disk_space_gb: float
    dependencies_installed: Dict[str, bool]
    environment_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)

@dataclass
class ExecutionEstimate:
    """Runtime estimation based on system specifications."""
    estimated_duration_minutes: float
    confidence_level: float
    resource_usage: Dict[str, float]
    optimization_recommendations: List[str] = field(default_factory=list)

class NavigationHub:
    """
    Interactive navigation system with dependency checking and runtime estimation.
    Provides comprehensive project management and notebook execution coordination.
    """
    
    def __init__(self, config: Optional[NotebookConfig] = None):
        """
        Initialize NavigationHub.
        
        Args:
            config: Optional configuration override
        """
        self.config = config or CONFIG
        self.logger = setup_notebook_logging("navigation_hub")
        self.notebooks = self._initialize_notebook_metadata()
        self.system_info = self._collect_system_info()
        
        # Initialize runtime estimation and dependency checking
        self.runtime_estimator = create_runtime_estimator(config)
        self.dependency_checker = create_dependency_checker()
        
        # Track execution history
        self.execution_history: List[Dict[str, Any]] = []
        self._load_execution_history()
    
    def _initialize_notebook_metadata(self) -> Dict[str, NotebookMetadata]:
        """Initialize comprehensive notebook metadata."""
        notebooks = {
            "01_exploratory_data_analysis": NotebookMetadata(
                filename="01_exploratory_data_analysis.ipynb",
                title="Exploratory Data Analysis",
                description="Comprehensive statistical profiling with custom DataQualityMetrics dataclass, Cramér's V correlation, and mutual information analysis",
                estimated_runtime_minutes=(10, 15),
                prerequisites=["Raw dataset in data/ directory"],
                key_outputs=["Data quality assessment", "Feature correlations", "Distribution analysis", "Statistical profiles"],
                dependencies=["pandas", "numpy", "matplotlib", "seaborn", "scipy", "plotly"],
                memory_requirement_gb=2.0,
                cpu_intensive=False,
                output_artifacts=["data_quality_report.json", "correlation_matrix.csv", "feature_profiles.pkl"]
            ),
            
            "02_customer_segmentation": NotebookMetadata(
                filename="02_customer_segmentation.ipynb",
                title="Customer Segmentation",
                description="Multi-algorithm clustering with K-Means, DBSCAN, Hierarchical, Gaussian Mixture, and Yellowbrick integration",
                estimated_runtime_minutes=(15, 20),
                prerequisites=["Completed EDA notebook", "Preprocessed data"],
                key_outputs=["Customer segments", "Cluster profiles", "Validation metrics", "Segment visualizations"],
                dependencies=["sklearn", "yellowbrick", "umap-learn", "hdbscan"],
                memory_requirement_gb=3.0,
                cpu_intensive=True,
                output_artifacts=["customer_segments.csv", "cluster_models.pkl", "segment_profiles.json"]
            ),
            
            "03_churn_prediction_modeling": NotebookMetadata(
                filename="03_churn_prediction_modeling.ipynb", 
                title="Churn Prediction Modeling",
                description="PyTorch custom architectures, Optuna optimization, ensemble methods, and SHAP/LIME interpretability",
                estimated_runtime_minutes=(20, 30),
                prerequisites=["Preprocessed data from EDA"],
                key_outputs=["Trained models", "Performance metrics", "Feature importance", "Model interpretability"],
                dependencies=["torch", "optuna", "shap", "lime", "sklearn"],
                memory_requirement_gb=4.0,
                cpu_intensive=True,
                gpu_recommended=True,
                output_artifacts=["trained_models.pkl", "performance_metrics.json", "feature_importance.csv"]
            ),
            
            "04_survival_analysis": NotebookMetadata(
                filename="04_survival_analysis.ipynb",
                title="Survival Analysis", 
                description="Lifelines integration with Kaplan-Meier estimation, Cox Proportional Hazards, and customer lifetime value calculations",
                estimated_runtime_minutes=(15, 20),
                prerequisites=["Customer tenure data", "Preprocessed features"],
                key_outputs=["Survival curves", "Hazard ratios", "Retention models", "CLV calculations"],
                dependencies=["lifelines", "numpy", "matplotlib"],
                memory_requirement_gb=2.5,
                cpu_intensive=False,
                output_artifacts=["survival_models.pkl", "clv_estimates.csv", "hazard_ratios.json"]
            ),
            
            "05_ab_testing_framework": NotebookMetadata(
                filename="05_ab_testing_framework.ipynb",
                title="A/B Testing Framework",
                description="PyMC Bayesian testing, power analysis, Monte Carlo simulation, and ROI quantification",
                estimated_runtime_minutes=(10, 15),
                prerequisites=["Historical campaign data", "Statistical baselines"],
                key_outputs=["Power analysis", "Experiment design", "ROI calculations", "Statistical tests"],
                dependencies=["pymc", "arviz", "scipy", "numpy"],
                memory_requirement_gb=2.0,
                cpu_intensive=True,
                output_artifacts=["experiment_designs.json", "power_analysis.csv", "roi_projections.json"]
            ),
            
            "06_cost_sensitive_modeling": NotebookMetadata(
                filename="06_cost_sensitive_modeling.ipynb",
                title="Cost-Sensitive Modeling",
                description="Cost matrix development, mathematical threshold optimization, and costcla integration",
                estimated_runtime_minutes=(15, 20),
                prerequisites=["Completed prediction models", "Business cost structure"],
                key_outputs=["Optimized thresholds", "Cost-sensitive models", "Profit maximization", "Business scenarios"],
                dependencies=["costcla", "scipy", "numpy"],
                memory_requirement_gb=2.5,
                cpu_intensive=True,
                output_artifacts=["cost_matrices.json", "optimized_thresholds.csv", "profit_models.pkl"]
            ),
            
            "07_model_comparison_evaluation": NotebookMetadata(
                filename="07_model_comparison_evaluation.ipynb",
                title="Model Comparison and Evaluation",
                description="McNemar's test, calibration analysis, fairness assessment, and comprehensive evaluation framework",
                estimated_runtime_minutes=(20, 25),
                prerequisites=["All model notebooks completed"],
                key_outputs=["Performance comparison", "Statistical significance tests", "Model recommendations", "Fairness metrics"],
                dependencies=["sklearn", "scipy", "matplotlib", "seaborn"],
                memory_requirement_gb=3.0,
                cpu_intensive=False,
                output_artifacts=["model_comparison.json", "evaluation_report.html", "recommendations.md"]
            ),
            
            "08_business_insights_recommendations": NotebookMetadata(
                filename="08_business_insights_recommendations.ipynb",
                title="Business Insights and Recommendations",
                description="Strategic synthesis, numpy-financial integration, priority matrices, and implementation roadmap",
                estimated_runtime_minutes=(10, 15),
                prerequisites=["All analytical components completed"],
                key_outputs=["Business strategy", "Implementation roadmap", "KPI framework", "ROI projections"],
                dependencies=["numpy-financial", "pandas", "matplotlib"],
                memory_requirement_gb=2.0,
                cpu_intensive=False,
                output_artifacts=["business_strategy.md", "implementation_roadmap.json", "kpi_dashboard.html"]
            )
        }
        
        return notebooks
    
    def _collect_system_info(self) -> SystemRequirements:
        """Collect comprehensive system information and validate environment."""
        # Basic system info
        python_version = sys.version_info[:3]
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_cores = psutil.cpu_count()
        
        # Check GPU availability
        gpu_available = False
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except ImportError:
            pass
        
        # Check disk space
        disk_usage = psutil.disk_usage('.')
        disk_space_gb = disk_usage.free / (1024**3)
        
        # Check dependencies
        dependencies_installed = self._check_dependencies()
        
        # Validate environment
        validation_errors = []
        environment_valid = True
        
        if python_version < (3, 8):
            validation_errors.append("Python 3.8+ required")
            environment_valid = False
        
        if memory_gb < 4:
            validation_errors.append("Minimum 4GB RAM recommended")
            environment_valid = False
        
        if disk_space_gb < 2:
            validation_errors.append("Minimum 2GB free disk space required")
            environment_valid = False
        
        return SystemRequirements(
            python_version=python_version,
            memory_gb=memory_gb,
            cpu_cores=cpu_cores,
            gpu_available=gpu_available,
            disk_space_gb=disk_space_gb,
            dependencies_installed=dependencies_installed,
            environment_valid=environment_valid,
            validation_errors=validation_errors
        )
    
    def _check_dependencies(self) -> Dict[str, bool]:
        """Check if required dependencies are installed."""
        required_packages = [
            "pandas", "numpy", "matplotlib", "seaborn", "plotly", "scipy",
            "sklearn", "torch", "optuna", "shap", "lime", "lifelines",
            "pymc", "arviz", "yellowbrick", "umap-learn", "hdbscan",
            "costcla", "numpy-financial", "ipywidgets"
        ]
        
        installed = {}
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                installed[package] = True
            except ImportError:
                installed[package] = False
        
        return installed
    
    def _load_execution_history(self) -> None:
        """Load execution history from file."""
        history_file = Path("logs/execution_history.json")
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    self.execution_history = json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load execution history: {e}")
                self.execution_history = []
    
    def _save_execution_history(self) -> None:
        """Save execution history to file."""
        history_file = Path("logs/execution_history.json")
        history_file.parent.mkdir(exist_ok=True)
        
        try:
            with open(history_file, 'w') as f:
                json.dump(self.execution_history, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Could not save execution history: {e}")
    
    def estimate_runtime(self, notebook_id: str, dataset_size_mb: Optional[float] = None) -> ExecutionEstimate:
        """
        Estimate runtime based on system specifications and historical data.
        
        Args:
            notebook_id: Notebook identifier
            dataset_size_mb: Optional dataset size for more accurate estimation
            
        Returns:
            ExecutionEstimate with duration and resource predictions
        """
        if notebook_id not in self.notebooks:
            raise ValueError(f"Unknown notebook: {notebook_id}")
        
        # Use advanced runtime estimator
        try:
            runtime_estimate = self.runtime_estimator.estimate_runtime(notebook_id, dataset_size_mb)
            
            return ExecutionEstimate(
                estimated_duration_minutes=runtime_estimate.estimated_minutes,
                confidence_level=runtime_estimate.confidence_level,
                resource_usage=runtime_estimate.resource_requirements,
                optimization_recommendations=runtime_estimate.optimization_recommendations
            )
            
        except Exception as e:
            self.logger.warning(f"Advanced runtime estimation failed, using fallback: {e}")
            
            # Fallback to simple estimation
            notebook = self.notebooks[notebook_id]
            min_time, max_time = notebook.estimated_runtime_minutes
            base_estimate = (min_time + max_time) / 2
            
            # Simple system adjustments
            memory_factor = max(0.5, min(2.0, self.system_info.memory_gb / 8.0))
            cpu_factor = max(0.5, min(2.0, self.system_info.cpu_cores / 4.0))
            
            adjusted_estimate = base_estimate * memory_factor * cpu_factor
            
            return ExecutionEstimate(
                estimated_duration_minutes=adjusted_estimate,
                confidence_level=0.6,
                resource_usage={
                    "memory_gb": notebook.memory_requirement_gb,
                    "cpu_utilization_percent": 60.0 if notebook.cpu_intensive else 30.0,
                    "disk_io_gb": 0.5
                },
                optimization_recommendations=["Use advanced runtime estimator for better predictions"]
            )
    
    def _get_historical_factor(self, notebook_id: str) -> float:
        """Get historical performance factor for runtime adjustment."""
        recent_runs = [
            h for h in self.execution_history 
            if h.get('notebook_id') == notebook_id and 
            datetime.fromisoformat(h.get('timestamp', '2020-01-01')) > datetime.now() - timedelta(days=30)
        ]
        
        if not recent_runs:
            return 1.0
        
        # Calculate average performance factor
        factors = []
        for run in recent_runs:
            actual_time = run.get('actual_duration_minutes', 0)
            estimated_time = run.get('estimated_duration_minutes', actual_time)
            if estimated_time > 0:
                factors.append(actual_time / estimated_time)
        
        return sum(factors) / len(factors) if factors else 1.0
    
    def validate_prerequisites(self, notebook_id: str) -> Tuple[bool, List[str]]:
        """
        Validate that prerequisites for a notebook are met.
        
        Args:
            notebook_id: Notebook identifier
            
        Returns:
            Tuple of (all_met, missing_prerequisites)
        """
        if notebook_id not in self.notebooks:
            return False, [f"Unknown notebook: {notebook_id}"]
        
        notebook = self.notebooks[notebook_id]
        missing = []
        
        # Check file prerequisites
        for prereq in notebook.prerequisites:
            if "dataset" in prereq.lower() or "data" in prereq.lower():
                data_file = self.config.data.data_path / self.config.data.raw_data_file
                if not data_file.exists():
                    missing.append(f"Missing data file: {data_file}")
            
            elif "completed" in prereq.lower():
                # Check for completed notebook artifacts
                prereq_notebook = prereq.lower().replace("completed ", "").replace(" notebook", "")
                if not self._check_notebook_completed(prereq_notebook):
                    missing.append(f"Prerequisite notebook not completed: {prereq_notebook}")
        
        # Check dependency installations
        for dep in notebook.dependencies:
            if not self.system_info.dependencies_installed.get(dep, False):
                missing.append(f"Missing dependency: {dep}")
        
        return len(missing) == 0, missing
    
    def _check_notebook_completed(self, notebook_name: str) -> bool:
        """Check if a notebook has been completed based on output artifacts."""
        # Simple heuristic: check if recent execution exists in history
        recent_runs = [
            h for h in self.execution_history 
            if notebook_name in h.get('notebook_id', '') and 
            h.get('status') == 'completed' and
            datetime.fromisoformat(h.get('timestamp', '2020-01-01')) > datetime.now() - timedelta(days=7)
        ]
        return len(recent_runs) > 0
    
    def create_interactive_launcher(self) -> None:
        """Create interactive widget-based notebook launcher."""
        # Notebook selection dropdown
        notebook_options = [(f"{nb.title} ({nb_id})", nb_id) for nb_id, nb in self.notebooks.items()]
        notebook_selector = widgets.Dropdown(
            options=notebook_options,
            description='Notebook:',
            style={'description_width': 'initial'}
        )
        
        # Information display area
        info_output = widgets.Output()
        
        # Action buttons
        validate_btn = widgets.Button(description="Validate Prerequisites", button_style='info')
        estimate_btn = widgets.Button(description="Estimate Runtime", button_style='warning')
        launch_btn = widgets.Button(description="Launch Notebook", button_style='success')
        
        # Status display
        status_output = widgets.Output()
        
        def update_info(change):
            """Update information display when notebook selection changes."""
            with info_output:
                clear_output()
                notebook_id = change['new']
                notebook = self.notebooks[notebook_id]
                
                print(f"📊 {notebook.title}")
                print("=" * 50)
                print(f"Description: {notebook.description}")
                print(f"Estimated Runtime: {notebook.estimated_runtime_str}")
                print(f"Memory Requirement: {notebook.memory_requirement_gb} GB")
                print(f"CPU Intensive: {'Yes' if notebook.cpu_intensive else 'No'}")
                print(f"GPU Recommended: {'Yes' if notebook.gpu_recommended else 'No'}")
                print("\nPrerequisites:")
                for prereq in notebook.prerequisites:
                    print(f"  • {prereq}")
                print("\nKey Outputs:")
                for output in notebook.key_outputs:
                    print(f"  • {output}")
        
        def validate_prerequisites_clicked(b):
            """Handle prerequisite validation."""
            with status_output:
                clear_output()
                notebook_id = notebook_selector.value
                valid, missing = self.validate_prerequisites(notebook_id)
                
                if valid:
                    print("✅ All prerequisites met!")
                else:
                    print("❌ Missing prerequisites:")
                    for item in missing:
                        print(f"  • {item}")
        
        def estimate_runtime_clicked(b):
            """Handle runtime estimation."""
            with status_output:
                clear_output()
                notebook_id = notebook_selector.value
                estimate = self.estimate_runtime(notebook_id)
                
                print(f"⏱️ Runtime Estimate: {estimate.estimated_duration_minutes:.1f} minutes")
                print(f"📊 Confidence Level: {estimate.confidence_level:.1%}")
                print("\n💾 Resource Usage:")
                for resource, usage in estimate.resource_usage.items():
                    print(f"  • {resource}: {usage:.2f}")
                
                if estimate.optimization_recommendations:
                    print("\n💡 Optimization Recommendations:")
                    for rec in estimate.optimization_recommendations:
                        print(f"  • {rec}")
        
        def launch_notebook_clicked(b):
            """Handle notebook launch."""
            with status_output:
                clear_output()
                notebook_id = notebook_selector.value
                notebook = self.notebooks[notebook_id]
                
                # Validate prerequisites first
                valid, missing = self.validate_prerequisites(notebook_id)
                if not valid:
                    print("❌ Cannot launch: Prerequisites not met")
                    for item in missing:
                        print(f"  • {item}")
                    return
                
                print(f"🚀 Launching {notebook.title}...")
                
                # Record execution start
                execution_record = {
                    'notebook_id': notebook_id,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'started',
                    'estimated_duration_minutes': self.estimate_runtime(notebook_id).estimated_duration_minutes
                }
                self.execution_history.append(execution_record)
                self._save_execution_history()
                
                # Create launch command
                notebook_path = Path("notebooks") / notebook.filename
                if notebook_path.exists():
                    print(f"📂 Opening: {notebook_path}")
                    # In Jupyter environment, this would open the notebook
                    display(HTML(f'<a href="{notebook_path}" target="_blank">Click here to open {notebook.title}</a>'))
                else:
                    print(f"❌ Notebook file not found: {notebook_path}")
        
        # Connect event handlers
        notebook_selector.observe(update_info, names='value')
        validate_btn.on_click(validate_prerequisites_clicked)
        estimate_btn.on_click(estimate_runtime_clicked)
        launch_btn.on_click(launch_notebook_clicked)
        
        # Initial info display
        update_info({'new': notebook_selector.value})
        
        # Layout
        button_box = widgets.HBox([validate_btn, estimate_btn, launch_btn])
        main_layout = widgets.VBox([
            widgets.HTML("<h2>🎯 Notebook Navigation Hub</h2>"),
            notebook_selector,
            info_output,
            button_box,
            status_output
        ])
        
        display(main_layout)
    
    def display_system_status(self) -> None:
        """Display comprehensive system status and environment validation."""
        print("🖥️  SYSTEM STATUS REPORT")
        print("=" * 60)
        
        # System Information
        print(f"Python Version: {'.'.join(map(str, self.system_info.python_version))}")
        print(f"Platform: {platform.platform()}")
        print(f"Memory: {self.system_info.memory_gb:.1f} GB")
        print(f"CPU Cores: {self.system_info.cpu_cores}")
        print(f"GPU Available: {'Yes' if self.system_info.gpu_available else 'No'}")
        print(f"Free Disk Space: {self.system_info.disk_space_gb:.1f} GB")
        
        # Environment Validation
        print(f"\n🔍 ENVIRONMENT VALIDATION")
        print("-" * 30)
        if self.system_info.environment_valid:
            print("✅ Environment validation passed")
        else:
            print("❌ Environment validation failed:")
            for error in self.system_info.validation_errors:
                print(f"  • {error}")
        
        # Dependency Status
        print(f"\n📦 DEPENDENCY STATUS")
        print("-" * 25)
        installed_count = sum(self.system_info.dependencies_installed.values())
        total_count = len(self.system_info.dependencies_installed)
        print(f"Installed: {installed_count}/{total_count}")
        
        missing_deps = [dep for dep, installed in self.system_info.dependencies_installed.items() if not installed]
        if missing_deps:
            print("Missing dependencies:")
            for dep in missing_deps:
                print(f"  ❌ {dep}")
        else:
            print("✅ All dependencies installed")
        
        # Performance Recommendations
        print(f"\n💡 PERFORMANCE RECOMMENDATIONS")
        print("-" * 35)
        
        recommendations = []
        if self.system_info.memory_gb < 8:
            recommendations.append("Consider upgrading to 8GB+ RAM for optimal performance")
        
        if not self.system_info.gpu_available:
            recommendations.append("GPU acceleration available for deep learning notebooks")
        
        if self.system_info.cpu_cores < 4:
            recommendations.append("Multi-core CPU recommended for parallel processing")
        
        if missing_deps:
            recommendations.append(f"Install missing dependencies: pip install {' '.join(missing_deps)}")
        
        if recommendations:
            for rec in recommendations:
                print(f"  • {rec}")
        else:
            print("✅ System optimally configured")
    
    def generate_project_documentation(self) -> str:
        """Generate comprehensive project architecture documentation."""
        doc = []
        
        doc.append("# Customer Churn Analysis: Project Architecture Documentation")
        doc.append("=" * 80)
        doc.append("")
        doc.append("## Project Overview")
        doc.append("")
        doc.append("This project implements a comprehensive customer churn analysis pipeline using")
        doc.append("modern data science best practices and enterprise-grade methodologies. The")
        doc.append("analysis follows a systematic approach from exploratory data analysis through")
        doc.append("business recommendations, with full statistical rigor and reproducibility.")
        doc.append("")
        
        doc.append("## Technical Architecture")
        doc.append("")
        doc.append("### Core Infrastructure")
        doc.append("- **Configuration Management**: Centralized configuration with type hints and dataclasses")
        doc.append("- **Logging Framework**: Structured logging with operation tracking and metrics")
        doc.append("- **Environment Validation**: Automated dependency checking and system requirements")
        doc.append("- **Reproducibility**: Fixed random seeds and version-controlled dependencies")
        doc.append("")
        
        doc.append("### Notebook Portfolio")
        doc.append("")
        
        for i, (notebook_id, notebook) in enumerate(self.notebooks.items(), 1):
            doc.append(f"#### {i:02d}. {notebook.title}")
            doc.append(f"**File**: `{notebook.filename}`")
            doc.append(f"**Description**: {notebook.description}")
            doc.append(f"**Runtime**: {notebook.estimated_runtime_str}")
            doc.append(f"**Memory**: {notebook.memory_requirement_gb} GB")
            doc.append("")
            doc.append("**Prerequisites**:")
            for prereq in notebook.prerequisites:
                doc.append(f"- {prereq}")
            doc.append("")
            doc.append("**Key Outputs**:")
            for output in notebook.key_outputs:
                doc.append(f"- {output}")
            doc.append("")
            doc.append("**Dependencies**:")
            for dep in notebook.dependencies:
                doc.append(f"- {dep}")
            doc.append("")
            if notebook.output_artifacts:
                doc.append("**Output Artifacts**:")
                for artifact in notebook.output_artifacts:
                    doc.append(f"- {artifact}")
                doc.append("")
        
        doc.append("## System Requirements")
        doc.append("")
        doc.append(f"### Minimum Requirements")
        doc.append(f"- Python 3.8+")
        doc.append(f"- 4GB RAM")
        doc.append(f"- 2GB free disk space")
        doc.append(f"- Multi-core CPU recommended")
        doc.append("")
        doc.append(f"### Recommended Configuration")
        doc.append(f"- Python 3.9+")
        doc.append(f"- 16GB RAM")
        doc.append(f"- 10GB free disk space")
        doc.append(f"- 8+ CPU cores")
        doc.append(f"- GPU for deep learning acceleration")
        doc.append("")
        
        doc.append("## Execution Workflow")
        doc.append("")
        doc.append("The notebooks should be executed in the following order:")
        doc.append("")
        
        for i, (notebook_id, notebook) in enumerate(self.notebooks.items(), 1):
            doc.append(f"{i}. **{notebook.title}** ({notebook.estimated_runtime_str})")
            doc.append(f"   - {notebook.description}")
            doc.append("")
        
        doc.append(f"**Total Estimated Time**: 2-3 hours")
        doc.append("")
        
        doc.append("## Quality Assurance")
        doc.append("")
        doc.append("### Code Quality Standards")
        doc.append("- Type hints for all functions and methods")
        doc.append("- Comprehensive docstrings and documentation")
        doc.append("- PEP 8 compliance with automated linting")
        doc.append("- Error handling for all external dependencies")
        doc.append("")
        doc.append("### Statistical Rigor")
        doc.append("- Hypothesis testing with multiple comparison corrections")
        doc.append("- Bootstrap confidence intervals for robust uncertainty quantification")
        doc.append("- Cross-validation with temporal consistency")
        doc.append("- Reproducible results with fixed random seeds")
        doc.append("")
        doc.append("### Business Integration")
        doc.append("- Direct connection between statistical findings and business KPIs")
        doc.append("- ROI calculations with realistic cost structures")
        doc.append("- Actionable recommendations with implementation timelines")
        doc.append("- Executive-ready summaries and visualizations")
        doc.append("")
        
        return "\n".join(doc)
    
    def check_dependencies(self) -> Dict[str, Any]:
        """
        Perform comprehensive dependency compatibility check.
        
        Returns:
            Dictionary with detailed dependency analysis
        """
        try:
            dependency_results = self.dependency_checker.check_compatibility()
            
            # Log results
            self.logger.log_data_operation(
                operation="dependency_check",
                status="completed",
                metrics={
                    "python_compatible": dependency_results["python_version"]["compatible"],
                    "missing_packages_count": len(dependency_results["missing_packages"]),
                    "conflicts_count": len(dependency_results["conflicts"]),
                    "overall_status": dependency_results["overall_status"]
                }
            )
            
            return dependency_results
            
        except Exception as e:
            self.logger.error(f"Dependency check failed: {e}")
            return {
                "error": str(e),
                "overall_status": "error"
            }
    
    def display_dependency_status(self) -> None:
        """Display comprehensive dependency status report."""
        print("📦 DEPENDENCY COMPATIBILITY REPORT")
        print("=" * 50)
        
        dependency_results = self.check_dependencies()
        
        if "error" in dependency_results:
            print(f"❌ Error checking dependencies: {dependency_results['error']}")
            return
        
        # Python version
        python_info = dependency_results["python_version"]
        python_status = "✅" if python_info["compatible"] else "❌"
        print(f"{python_status} Python Version: {'.'.join(map(str, python_info['current']))}")
        print(f"   Required: {'.'.join(map(str, python_info['min_required']))}+")
        print(f"   Recommended: {'.'.join(map(str, python_info['recommended']))}+")
        
        # Missing packages
        missing = dependency_results["missing_packages"]
        if missing:
            print(f"\n❌ Missing Packages ({len(missing)}):")
            for pkg in missing:
                print(f"   • {pkg}")
            print(f"\n💡 Install missing packages: pip install {' '.join(missing)}")
        else:
            print(f"\n✅ All required packages installed")
        
        # Conflicts
        conflicts = dependency_results["conflicts"]
        if conflicts:
            print(f"\n⚠️  Package Conflicts ({len(conflicts)}):")
            for conflict in conflicts:
                print(f"   • {', '.join(conflict['packages'])}: {conflict['reason']}")
        
        # Package versions
        package_versions = dependency_results["package_versions"]
        if package_versions:
            print(f"\n📋 Package Version Status:")
            for pkg, info in package_versions.items():
                status = "✅" if info["compatible"] else "❌"
                version = info.get("version", "unknown")
                print(f"   {status} {pkg}: {version}")
        
        # Overall status
        overall_status = dependency_results["overall_status"]
        if overall_status == "good":
            print(f"\n✅ Overall Status: Environment ready for execution")
        elif overall_status == "critical":
            print(f"\n❌ Overall Status: Critical issues found - resolve before execution")
        else:
            print(f"\n⚠️  Overall Status: {overall_status}")
    
    def export_documentation(self, output_path: Optional[Path] = None) -> Path:
        """
        Export project documentation to markdown file.
        
        Args:
            output_path: Optional custom output path
            
        Returns:
            Path to exported documentation
        """
        if output_path is None:
            output_path = Path("PROJECT_ARCHITECTURE.md")
        
        documentation = self.generate_project_documentation()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(documentation)
        
        self.logger.info(f"Project documentation exported to {output_path}")
        return output_path

# Convenience function for easy import
def create_navigation_hub(config: Optional[NotebookConfig] = None) -> NavigationHub:
    """Create and return a NavigationHub instance."""
    return NavigationHub(config)