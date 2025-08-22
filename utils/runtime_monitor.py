"""
Runtime estimation and resource monitoring system.
Provides dynamic execution time prediction and resource usage optimization.
"""

import psutil
import platform
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import numpy as np
from pathlib import Path
import subprocess
import sys

from .config import CONFIG, NotebookConfig
from .logging_setup import setup_notebook_logging, NotebookLogger

@dataclass
class SystemSpecs:
    """Comprehensive system specifications for runtime estimation."""
    cpu_count: int
    cpu_freq_mhz: float
    memory_total_gb: float
    memory_available_gb: float
    disk_io_speed_mbps: float
    gpu_available: bool
    gpu_memory_gb: float
    python_version: Tuple[int, int, int]
    platform_info: str
    
    @classmethod
    def collect_current_specs(cls) -> 'SystemSpecs':
        """Collect current system specifications."""
        # CPU information
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        cpu_freq_mhz = cpu_freq.current if cpu_freq else 2000.0  # Default fallback
        
        # Memory information
        memory = psutil.virtual_memory()
        memory_total_gb = memory.total / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        
        # Disk I/O speed estimation (simplified)
        disk_io_speed_mbps = cls._estimate_disk_speed()
        
        # GPU information
        gpu_available, gpu_memory_gb = cls._check_gpu_specs()
        
        # Platform information
        python_version = sys.version_info[:3]
        platform_info = platform.platform()
        
        return cls(
            cpu_count=cpu_count,
            cpu_freq_mhz=cpu_freq_mhz,
            memory_total_gb=memory_total_gb,
            memory_available_gb=memory_available_gb,
            disk_io_speed_mbps=disk_io_speed_mbps,
            gpu_available=gpu_available,
            gpu_memory_gb=gpu_memory_gb,
            python_version=python_version,
            platform_info=platform_info
        )
    
    @staticmethod
    def _estimate_disk_speed() -> float:
        """Estimate disk I/O speed in MB/s."""
        try:
            # Simple disk speed test
            test_file = Path("temp_speed_test.tmp")
            test_data = b"0" * (1024 * 1024)  # 1MB of data
            
            start_time = time.time()
            with open(test_file, 'wb') as f:
                for _ in range(10):  # Write 10MB
                    f.write(test_data)
            write_time = time.time() - start_time
            
            start_time = time.time()
            with open(test_file, 'rb') as f:
                while f.read(1024 * 1024):
                    pass
            read_time = time.time() - start_time
            
            # Clean up
            test_file.unlink(missing_ok=True)
            
            # Calculate average speed
            total_mb = 20  # 10MB write + 10MB read
            total_time = write_time + read_time
            speed_mbps = total_mb / total_time if total_time > 0 else 100.0
            
            return min(speed_mbps, 1000.0)  # Cap at 1GB/s
            
        except Exception:
            return 100.0  # Default fallback speed
    
    @staticmethod
    def _check_gpu_specs() -> Tuple[bool, float]:
        """Check GPU availability and memory."""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_gb = gpu_memory_bytes / (1024**3)
                return True, gpu_memory_gb
        except ImportError:
            pass
        
        return False, 0.0

@dataclass
class ResourceUsage:
    """Real-time resource usage metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    disk_io_read_mbps: float
    disk_io_write_mbps: float
    gpu_utilization: float = 0.0
    gpu_memory_used_gb: float = 0.0

@dataclass
class PerformanceProfile:
    """Performance profile for runtime estimation."""
    base_complexity_score: float
    memory_intensity_factor: float
    cpu_intensity_factor: float
    io_intensity_factor: float
    gpu_acceleration_factor: float
    parallel_efficiency: float
    
    @classmethod
    def create_for_notebook(cls, notebook_type: str) -> 'PerformanceProfile':
        """Create performance profile based on notebook type."""
        profiles = {
            "eda": cls(
                base_complexity_score=1.0,
                memory_intensity_factor=0.8,
                cpu_intensity_factor=0.6,
                io_intensity_factor=1.2,
                gpu_acceleration_factor=1.0,
                parallel_efficiency=0.7
            ),
            "segmentation": cls(
                base_complexity_score=1.5,
                memory_intensity_factor=1.2,
                cpu_intensity_factor=1.8,
                io_intensity_factor=0.8,
                gpu_acceleration_factor=1.0,
                parallel_efficiency=0.9
            ),
            "prediction": cls(
                base_complexity_score=2.0,
                memory_intensity_factor=1.5,
                cpu_intensity_factor=2.2,
                io_intensity_factor=1.0,
                gpu_acceleration_factor=0.4,  # 60% speedup with GPU
                parallel_efficiency=0.8
            ),
            "survival": cls(
                base_complexity_score=1.2,
                memory_intensity_factor=1.0,
                cpu_intensity_factor=1.4,
                io_intensity_factor=0.9,
                gpu_acceleration_factor=1.0,
                parallel_efficiency=0.6
            ),
            "ab_testing": cls(
                base_complexity_score=1.3,
                memory_intensity_factor=0.9,
                cpu_intensity_factor=1.6,
                io_intensity_factor=0.7,
                gpu_acceleration_factor=1.0,
                parallel_efficiency=0.8
            ),
            "cost_sensitive": cls(
                base_complexity_score=1.4,
                memory_intensity_factor=1.1,
                cpu_intensity_factor=1.7,
                io_intensity_factor=0.8,
                gpu_acceleration_factor=1.0,
                parallel_efficiency=0.7
            ),
            "evaluation": cls(
                base_complexity_score=1.6,
                memory_intensity_factor=1.3,
                cpu_intensity_factor=1.5,
                io_intensity_factor=1.1,
                gpu_acceleration_factor=1.0,
                parallel_efficiency=0.8
            ),
            "insights": cls(
                base_complexity_score=0.8,
                memory_intensity_factor=0.7,
                cpu_intensity_factor=0.9,
                io_intensity_factor=1.0,
                gpu_acceleration_factor=1.0,
                parallel_efficiency=0.5
            )
        }
        
        return profiles.get(notebook_type, profiles["eda"])

@dataclass
class RuntimeEstimate:
    """Comprehensive runtime estimation with confidence intervals."""
    estimated_minutes: float
    confidence_interval: Tuple[float, float]
    confidence_level: float
    bottleneck_analysis: Dict[str, float]
    optimization_recommendations: List[str]
    resource_requirements: Dict[str, float]

class ResourceMonitor:
    """Real-time resource monitoring and optimization system."""
    
    def __init__(self, sampling_interval: float = 1.0):
        """
        Initialize resource monitor.
        
        Args:
            sampling_interval: Sampling interval in seconds
        """
        self.sampling_interval = sampling_interval
        self.logger = setup_notebook_logging("resource_monitor")
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.usage_history: List[ResourceUsage] = []
        self.max_history_size = 1000
        
    def start_monitoring(self) -> None:
        """Start real-time resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring:
            try:
                usage = self._collect_usage_sample()
                self.usage_history.append(usage)
                
                # Limit history size
                if len(self.usage_history) > self.max_history_size:
                    self.usage_history = self.usage_history[-self.max_history_size:]
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.sampling_interval)
    
    def _collect_usage_sample(self) -> ResourceUsage:
        """Collect a single resource usage sample."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = (memory.total - memory.available) / (1024**3)
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        if hasattr(self, '_last_disk_io'):
            time_delta = time.time() - self._last_disk_time
            read_bytes_delta = disk_io.read_bytes - self._last_disk_io.read_bytes
            write_bytes_delta = disk_io.write_bytes - self._last_disk_io.write_bytes
            
            disk_io_read_mbps = (read_bytes_delta / (1024**2)) / time_delta if time_delta > 0 else 0
            disk_io_write_mbps = (write_bytes_delta / (1024**2)) / time_delta if time_delta > 0 else 0
        else:
            disk_io_read_mbps = 0.0
            disk_io_write_mbps = 0.0
        
        self._last_disk_io = disk_io
        self._last_disk_time = time.time()
        
        # GPU usage (if available)
        gpu_utilization, gpu_memory_used_gb = self._get_gpu_usage()
        
        return ResourceUsage(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            disk_io_read_mbps=disk_io_read_mbps,
            disk_io_write_mbps=disk_io_write_mbps,
            gpu_utilization=gpu_utilization,
            gpu_memory_used_gb=gpu_memory_used_gb
        )
    
    def _get_gpu_usage(self) -> Tuple[float, float]:
        """Get GPU utilization and memory usage."""
        try:
            import torch
            if torch.cuda.is_available():
                # GPU utilization (simplified)
                gpu_utilization = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0.0
                
                # GPU memory
                gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)
                
                return gpu_utilization, gpu_memory_used
        except (ImportError, RuntimeError):
            pass
        
        return 0.0, 0.0
    
    def get_current_usage(self) -> Optional[ResourceUsage]:
        """Get the most recent resource usage sample."""
        return self.usage_history[-1] if self.usage_history else None
    
    def get_usage_statistics(self, window_minutes: int = 5) -> Dict[str, float]:
        """
        Get resource usage statistics for the specified time window.
        
        Args:
            window_minutes: Time window in minutes
            
        Returns:
            Dictionary with usage statistics
        """
        if not self.usage_history:
            return {}
        
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_usage = [u for u in self.usage_history if u.timestamp >= cutoff_time]
        
        if not recent_usage:
            return {}
        
        cpu_values = [u.cpu_percent for u in recent_usage]
        memory_values = [u.memory_percent for u in recent_usage]
        disk_read_values = [u.disk_io_read_mbps for u in recent_usage]
        disk_write_values = [u.disk_io_write_mbps for u in recent_usage]
        
        return {
            "cpu_mean": np.mean(cpu_values),
            "cpu_max": np.max(cpu_values),
            "cpu_std": np.std(cpu_values),
            "memory_mean": np.mean(memory_values),
            "memory_max": np.max(memory_values),
            "memory_std": np.std(memory_values),
            "disk_read_mean": np.mean(disk_read_values),
            "disk_read_max": np.max(disk_read_values),
            "disk_write_mean": np.mean(disk_write_values),
            "disk_write_max": np.max(disk_write_values),
            "sample_count": len(recent_usage)
        }

class RuntimeEstimator:
    """Advanced runtime estimation system with machine learning-based predictions."""
    
    def __init__(self, config: Optional[NotebookConfig] = None):
        """
        Initialize runtime estimator.
        
        Args:
            config: Optional configuration override
        """
        self.config = config or CONFIG
        self.logger = setup_notebook_logging("runtime_estimator")
        self.system_specs = SystemSpecs.collect_current_specs()
        self.execution_history = self._load_execution_history()
        
    def _load_execution_history(self) -> List[Dict[str, Any]]:
        """Load historical execution data."""
        history_file = Path("logs/execution_history.json")
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load execution history: {e}")
        
        return []
    
    def estimate_runtime(self, notebook_id: str, dataset_size_mb: Optional[float] = None) -> RuntimeEstimate:
        """
        Estimate runtime for a specific notebook.
        
        Args:
            notebook_id: Notebook identifier
            dataset_size_mb: Optional dataset size in MB
            
        Returns:
            Comprehensive runtime estimate
        """
        # Get notebook type for performance profiling
        notebook_type = self._get_notebook_type(notebook_id)
        profile = PerformanceProfile.create_for_notebook(notebook_type)
        
        # Base estimation from historical data
        base_estimate = self._get_base_estimate(notebook_id)
        
        # System-specific adjustments
        system_factor = self._calculate_system_factor(profile)
        
        # Dataset size adjustment
        dataset_factor = self._calculate_dataset_factor(dataset_size_mb, notebook_type)
        
        # Calculate final estimate
        estimated_minutes = base_estimate * system_factor * dataset_factor
        
        # Confidence interval calculation
        confidence_interval, confidence_level = self._calculate_confidence_interval(
            notebook_id, estimated_minutes
        )
        
        # Bottleneck analysis
        bottleneck_analysis = self._analyze_bottlenecks(profile)
        
        # Optimization recommendations
        optimization_recommendations = self._generate_recommendations(profile, bottleneck_analysis)
        
        # Resource requirements
        resource_requirements = self._calculate_resource_requirements(profile, estimated_minutes)
        
        return RuntimeEstimate(
            estimated_minutes=estimated_minutes,
            confidence_interval=confidence_interval,
            confidence_level=confidence_level,
            bottleneck_analysis=bottleneck_analysis,
            optimization_recommendations=optimization_recommendations,
            resource_requirements=resource_requirements
        )
    
    def _get_notebook_type(self, notebook_id: str) -> str:
        """Extract notebook type from notebook ID."""
        type_mapping = {
            "01_exploratory": "eda",
            "02_customer": "segmentation", 
            "03_churn": "prediction",
            "04_survival": "survival",
            "05_ab": "ab_testing",
            "06_cost": "cost_sensitive",
            "07_model": "evaluation",
            "08_business": "insights"
        }
        
        for key, notebook_type in type_mapping.items():
            if key in notebook_id:
                return notebook_type
        
        return "eda"  # Default fallback
    
    def _get_base_estimate(self, notebook_id: str) -> float:
        """Get base time estimate from historical data or defaults."""
        # Historical average
        historical_times = [
            h.get('actual_duration_minutes', 0) 
            for h in self.execution_history 
            if h.get('notebook_id') == notebook_id and h.get('actual_duration_minutes', 0) > 0
        ]
        
        if historical_times:
            return np.median(historical_times)
        
        # Default estimates by notebook type
        defaults = {
            "01_exploratory": 12.5,
            "02_customer": 17.5,
            "03_churn": 25.0,
            "04_survival": 17.5,
            "05_ab": 12.5,
            "06_cost": 17.5,
            "07_model": 22.5,
            "08_business": 12.5
        }
        
        for key, default_time in defaults.items():
            if key in notebook_id:
                return default_time
        
        return 15.0  # General fallback
    
    def _calculate_system_factor(self, profile: PerformanceProfile) -> float:
        """Calculate system performance factor."""
        # CPU factor
        cpu_baseline = 4.0  # 4 cores baseline
        cpu_factor = (self.system_specs.cpu_count / cpu_baseline) ** profile.parallel_efficiency
        
        # Memory factor
        memory_baseline = 8.0  # 8GB baseline
        memory_factor = min(2.0, self.system_specs.memory_available_gb / memory_baseline)
        
        # GPU factor
        gpu_factor = profile.gpu_acceleration_factor if self.system_specs.gpu_available else 1.0
        
        # Disk I/O factor
        disk_baseline = 100.0  # 100 MB/s baseline
        disk_factor = min(2.0, self.system_specs.disk_io_speed_mbps / disk_baseline)
        
        # Weighted combination
        system_factor = (
            cpu_factor * profile.cpu_intensity_factor * 0.4 +
            memory_factor * profile.memory_intensity_factor * 0.3 +
            gpu_factor * 0.2 +
            disk_factor * profile.io_intensity_factor * 0.1
        )
        
        return max(0.2, min(5.0, system_factor))  # Clamp between 0.2x and 5x
    
    def _calculate_dataset_factor(self, dataset_size_mb: Optional[float], notebook_type: str) -> float:
        """Calculate dataset size impact factor."""
        if dataset_size_mb is None:
            return 1.0
        
        # Baseline dataset size (MB)
        baseline_size = 50.0
        
        # Scaling factors by notebook type
        scaling_factors = {
            "eda": 0.8,
            "segmentation": 1.2,
            "prediction": 1.5,
            "survival": 1.0,
            "ab_testing": 0.6,
            "cost_sensitive": 1.1,
            "evaluation": 1.3,
            "insights": 0.5
        }
        
        scaling_factor = scaling_factors.get(notebook_type, 1.0)
        size_ratio = dataset_size_mb / baseline_size
        
        # Logarithmic scaling to avoid extreme values
        dataset_factor = 1.0 + scaling_factor * np.log(max(1.0, size_ratio))
        
        return max(0.5, min(10.0, dataset_factor))
    
    def _calculate_confidence_interval(self, notebook_id: str, estimate: float) -> Tuple[Tuple[float, float], float]:
        """Calculate confidence interval for the estimate."""
        # Historical variance
        historical_times = [
            h.get('actual_duration_minutes', 0) 
            for h in self.execution_history 
            if h.get('notebook_id') == notebook_id and h.get('actual_duration_minutes', 0) > 0
        ]
        
        if len(historical_times) >= 3:
            std_dev = np.std(historical_times)
            confidence_level = min(0.95, 0.5 + len(historical_times) * 0.05)
        else:
            # Default uncertainty
            std_dev = estimate * 0.3  # 30% standard deviation
            confidence_level = 0.6
        
        # 95% confidence interval (approximately)
        margin = 1.96 * std_dev
        lower_bound = max(0.1, estimate - margin)
        upper_bound = estimate + margin
        
        return (lower_bound, upper_bound), confidence_level
    
    def _analyze_bottlenecks(self, profile: PerformanceProfile) -> Dict[str, float]:
        """Analyze potential performance bottlenecks."""
        bottlenecks = {}
        
        # CPU bottleneck
        cpu_demand = profile.cpu_intensity_factor * profile.base_complexity_score
        cpu_capacity = self.system_specs.cpu_count / 4.0  # Normalized to 4-core baseline
        bottlenecks["cpu"] = cpu_demand / cpu_capacity
        
        # Memory bottleneck
        memory_demand = profile.memory_intensity_factor * profile.base_complexity_score
        memory_capacity = self.system_specs.memory_available_gb / 8.0  # Normalized to 8GB baseline
        bottlenecks["memory"] = memory_demand / memory_capacity
        
        # I/O bottleneck
        io_demand = profile.io_intensity_factor * profile.base_complexity_score
        io_capacity = self.system_specs.disk_io_speed_mbps / 100.0  # Normalized to 100MB/s baseline
        bottlenecks["io"] = io_demand / io_capacity
        
        # GPU bottleneck (if applicable)
        if profile.gpu_acceleration_factor < 1.0:  # GPU can help
            gpu_capacity = 1.0 if self.system_specs.gpu_available else 0.1
            bottlenecks["gpu"] = (1.0 / profile.gpu_acceleration_factor) / gpu_capacity
        
        return bottlenecks
    
    def _generate_recommendations(self, profile: PerformanceProfile, bottlenecks: Dict[str, float]) -> List[str]:
        """Generate optimization recommendations based on bottleneck analysis."""
        recommendations = []
        
        # Find primary bottleneck
        primary_bottleneck = max(bottlenecks.items(), key=lambda x: x[1])
        bottleneck_type, bottleneck_severity = primary_bottleneck
        
        if bottleneck_severity > 1.5:  # Significant bottleneck
            if bottleneck_type == "cpu":
                recommendations.append("Consider upgrading to a multi-core CPU for better parallel processing")
                recommendations.append("Close other CPU-intensive applications during execution")
            
            elif bottleneck_type == "memory":
                recommendations.append("Increase system RAM or close memory-intensive applications")
                recommendations.append("Consider processing data in smaller batches")
            
            elif bottleneck_type == "io":
                recommendations.append("Use SSD storage for better I/O performance")
                recommendations.append("Ensure sufficient free disk space for temporary files")
            
            elif bottleneck_type == "gpu":
                recommendations.append("Install GPU acceleration (CUDA) for significant speedup")
                recommendations.append("Consider cloud instances with GPU support")
        
        # General recommendations
        if self.system_specs.memory_available_gb < 4.0:
            recommendations.append("Minimum 4GB RAM recommended for optimal performance")
        
        if not self.system_specs.gpu_available and profile.gpu_acceleration_factor < 1.0:
            recommendations.append("GPU acceleration available for this notebook type")
        
        return recommendations
    
    def _calculate_resource_requirements(self, profile: PerformanceProfile, estimated_minutes: float) -> Dict[str, float]:
        """Calculate expected resource requirements."""
        base_memory = 2.0  # Base memory requirement in GB
        memory_requirement = base_memory * profile.memory_intensity_factor
        
        cpu_utilization = min(100.0, profile.cpu_intensity_factor * 60.0)  # Percentage
        
        disk_io_gb = profile.io_intensity_factor * estimated_minutes * 0.1  # Rough estimate
        
        return {
            "memory_gb": memory_requirement,
            "cpu_utilization_percent": cpu_utilization,
            "disk_io_gb": disk_io_gb,
            "estimated_duration_minutes": estimated_minutes
        }

class DependencyChecker:
    """Automated dependency compatibility checking system."""
    
    def __init__(self):
        """Initialize dependency checker."""
        self.logger = setup_notebook_logging("dependency_checker")
        self.requirements_file = Path("requirements.txt")
        
    def check_compatibility(self) -> Dict[str, Any]:
        """
        Perform comprehensive dependency compatibility check.
        
        Returns:
            Dictionary with compatibility results
        """
        results = {
            "python_version": self._check_python_version(),
            "package_versions": self._check_package_versions(),
            "conflicts": self._check_conflicts(),
            "missing_packages": self._check_missing_packages(),
            "outdated_packages": self._check_outdated_packages(),
            "security_vulnerabilities": self._check_security(),
            "overall_status": "unknown"
        }
        
        # Determine overall status
        has_critical_issues = (
            not results["python_version"]["compatible"] or
            len(results["conflicts"]) > 0 or
            len(results["missing_packages"]) > 0
        )
        
        results["overall_status"] = "critical" if has_critical_issues else "good"
        
        return results
    
    def _check_python_version(self) -> Dict[str, Any]:
        """Check Python version compatibility."""
        current_version = sys.version_info[:3]
        min_required = (3, 8, 0)
        recommended = (3, 9, 0)
        
        compatible = current_version >= min_required
        is_recommended = current_version >= recommended
        
        return {
            "current": current_version,
            "min_required": min_required,
            "recommended": recommended,
            "compatible": compatible,
            "is_recommended": is_recommended
        }
    
    def _check_package_versions(self) -> Dict[str, Dict[str, Any]]:
        """Check installed package versions against requirements."""
        if not self.requirements_file.exists():
            return {}
        
        results = {}
        
        try:
            with open(self.requirements_file, 'r') as f:
                requirements = f.readlines()
            
            for req in requirements:
                req = req.strip()
                if req and not req.startswith('#'):
                    package_info = self._parse_requirement(req)
                    if package_info:
                        results[package_info["name"]] = self._check_single_package(package_info)
        
        except Exception as e:
            self.logger.error(f"Error checking package versions: {e}")
        
        return results
    
    def _parse_requirement(self, requirement: str) -> Optional[Dict[str, Any]]:
        """Parse a single requirement string."""
        try:
            # Simple parsing for package==version format
            if "==" in requirement:
                name, version = requirement.split("==", 1)
                return {"name": name.strip(), "required_version": version.strip()}
            elif ">=" in requirement:
                name, version = requirement.split(">=", 1)
                return {"name": name.strip(), "min_version": version.strip()}
            else:
                return {"name": requirement.strip()}
        except Exception:
            return None
    
    def _check_single_package(self, package_info: Dict[str, Any]) -> Dict[str, Any]:
        """Check a single package installation and version."""
        package_name = package_info["name"]
        
        try:
            # Try to import and get version
            if package_name == "sklearn":
                import sklearn
                installed_version = sklearn.__version__
            elif package_name == "torch":
                import torch
                installed_version = torch.__version__
            else:
                # Generic import
                module = __import__(package_name)
                installed_version = getattr(module, "__version__", "unknown")
            
            result = {
                "installed": True,
                "version": installed_version,
                "compatible": True
            }
            
            # Check version compatibility if specified
            if "required_version" in package_info:
                result["compatible"] = installed_version == package_info["required_version"]
            elif "min_version" in package_info:
                result["compatible"] = self._version_compare(installed_version, package_info["min_version"]) >= 0
            
            return result
            
        except ImportError:
            return {
                "installed": False,
                "version": None,
                "compatible": False
            }
    
    def _version_compare(self, version1: str, version2: str) -> int:
        """Compare two version strings. Returns -1, 0, or 1."""
        try:
            v1_parts = [int(x) for x in version1.split('.')]
            v2_parts = [int(x) for x in version2.split('.')]
            
            # Pad shorter version with zeros
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts.extend([0] * (max_len - len(v1_parts)))
            v2_parts.extend([0] * (max_len - len(v2_parts)))
            
            for v1, v2 in zip(v1_parts, v2_parts):
                if v1 < v2:
                    return -1
                elif v1 > v2:
                    return 1
            
            return 0
            
        except Exception:
            return 0  # Assume equal if can't parse
    
    def _check_conflicts(self) -> List[Dict[str, Any]]:
        """Check for package conflicts."""
        # Simplified conflict detection
        conflicts = []
        
        # Known conflict patterns
        known_conflicts = [
            {"packages": ["tensorflow", "torch"], "reason": "Both deep learning frameworks may conflict"},
            {"packages": ["matplotlib", "seaborn"], "reason": "Version compatibility issues possible"}
        ]
        
        for conflict in known_conflicts:
            installed_packages = []
            for pkg in conflict["packages"]:
                try:
                    __import__(pkg)
                    installed_packages.append(pkg)
                except ImportError:
                    pass
            
            if len(installed_packages) > 1:
                conflicts.append({
                    "packages": installed_packages,
                    "reason": conflict["reason"],
                    "severity": "warning"
                })
        
        return conflicts
    
    def _check_missing_packages(self) -> List[str]:
        """Check for missing required packages."""
        if not self.requirements_file.exists():
            return []
        
        missing = []
        
        try:
            with open(self.requirements_file, 'r') as f:
                requirements = f.readlines()
            
            for req in requirements:
                req = req.strip()
                if req and not req.startswith('#'):
                    package_info = self._parse_requirement(req)
                    if package_info:
                        try:
                            __import__(package_info["name"])
                        except ImportError:
                            missing.append(package_info["name"])
        
        except Exception as e:
            self.logger.error(f"Error checking missing packages: {e}")
        
        return missing
    
    def _check_outdated_packages(self) -> List[Dict[str, str]]:
        """Check for outdated packages (simplified)."""
        # This would typically use pip list --outdated
        # Simplified implementation for demonstration
        return []
    
    def _check_security(self) -> List[Dict[str, Any]]:
        """Check for security vulnerabilities (simplified)."""
        # This would typically use safety or similar tools
        # Simplified implementation for demonstration
        return []

# Convenience functions
def create_resource_monitor(sampling_interval: float = 1.0) -> ResourceMonitor:
    """Create a resource monitor instance."""
    return ResourceMonitor(sampling_interval)

def create_runtime_estimator(config: Optional[NotebookConfig] = None) -> RuntimeEstimator:
    """Create a runtime estimator instance."""
    return RuntimeEstimator(config)

def create_dependency_checker() -> DependencyChecker:
    """Create a dependency checker instance."""
    return DependencyChecker()