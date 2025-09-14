"""
Audit logging for full reproducibility and cost tracking.
Following Linus principle: Log everything that matters, nothing that doesn't.
"""

import json
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import os
import sys

class AuditLogger:
    """
    Unified audit logger for usage tracking and provenance.
    Writes to:
    - results/usage_log.jsonl: API calls and costs
    - results/run_provenance.json: Git hash, seeds, configs
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path(__file__).parent.parent.parent / "results"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.usage_log_path = self.output_dir / "usage_log.jsonl"
        self.provenance_path = self.output_dir / "run_provenance.json"
        
        # Initialize provenance
        self.provenance = self._init_provenance()
        
        # Track totals
        self.total_tokens = 0
        self.total_cost = 0.0
        self.api_calls = 0
    
    def _init_provenance(self) -> Dict[str, Any]:
        """Initialize provenance with system info."""
        return {
            'timestamp': datetime.now().isoformat(),
            'git_commit': self._get_git_commit(),
            'git_branch': self._get_git_branch(),
            'git_dirty': self._is_git_dirty(),
            'python_version': sys.version,
            'platform': sys.platform,
            'cwd': os.getcwd(),
            'env_vars': {
                k: v for k, v in os.environ.items() 
                if any(x in k for x in ['OPENAI', 'ANTHROPIC', 'GOOGLE', 'DEEPSEEK', 'PERPLEXITY'])
            },
            'runs': []
        }
    
    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except:
            return "unknown"
    
    def _get_git_branch(self) -> str:
        """Get current git branch."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except:
            return "unknown"
    
    def _is_git_dirty(self) -> bool:
        """Check if git working directory has uncommitted changes."""
        try:
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                check=True
            )
            return len(result.stdout.strip()) > 0
        except:
            return True
    
    def log_api_call(self, 
                     provider: str,
                     model: str,
                     prompt: str,
                     response: str,
                     tokens: int,
                     cost: float,
                     task_type: Optional[str] = None,
                     metadata: Optional[Dict] = None) -> None:
        """
        Log an API call for usage tracking.
        
        Args:
            provider: API provider (openai, anthropic, etc.)
            model: Model name
            prompt: The prompt sent
            response: The response received
            tokens: Token count
            cost: Estimated cost in USD
            task_type: Optional task classification
            metadata: Optional additional metadata
        """
        
        # Create log entry
        entry = {
            'timestamp': datetime.now().isoformat(),
            'provider': provider,
            'model': model,
            'prompt_hash': hashlib.sha256(prompt.encode()).hexdigest(),
            'response_hash': hashlib.sha256(response.encode()).hexdigest(),
            'prompt_length': len(prompt),
            'response_length': len(response),
            'tokens': tokens,
            'cost': cost,
            'task_type': task_type,
            'metadata': metadata or {}
        }
        
        # Append to usage log
        with open(self.usage_log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        
        # Update totals
        self.total_tokens += tokens
        self.total_cost += cost
        self.api_calls += 1
        
        # Log to console if expensive
        if cost > 0.10:
            print(f"[AUDIT] Expensive call: ${cost:.3f} for {model} ({tokens} tokens)")
    
    def log_run(self,
                experiment_name: str,
                config: Dict[str, Any],
                seed: int,
                results: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an experimental run for provenance.
        
        Args:
            experiment_name: Name of the experiment
            config: Configuration used
            seed: Random seed
            results: Optional results summary
        """
        
        run_entry = {
            'timestamp': datetime.now().isoformat(),
            'experiment': experiment_name,
            'seed': seed,
            'config': config,
            'results': results or {},
            'api_calls': self.api_calls,
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost
        }
        
        # Add to provenance
        self.provenance['runs'].append(run_entry)
        
        # Save provenance
        self.save_provenance()
        
        print(f"[AUDIT] Logged run: {experiment_name} (seed={seed}, cost=${self.total_cost:.3f})")
    
    def save_provenance(self) -> None:
        """Save provenance to file."""
        with open(self.provenance_path, 'w') as f:
            json.dump(self.provenance, f, indent=2, default=str)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of current session."""
        return {
            'api_calls': self.api_calls,
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost,
            'avg_cost_per_call': self.total_cost / max(1, self.api_calls),
            'experiments_run': len(self.provenance['runs'])
        }
    
    def reset_counters(self) -> None:
        """Reset counters for new experiment."""
        self.total_tokens = 0
        self.total_cost = 0.0
        self.api_calls = 0


# Global singleton instance
_audit_logger = None

def get_audit_logger() -> AuditLogger:
    """Get or create global audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def audit_api_call(provider: str, model: str, prompt: str, response: str,
                   tokens: int, cost: float, **kwargs) -> None:
    """Convenience function to log API call."""
    logger = get_audit_logger()
    logger.log_api_call(provider, model, prompt, response, tokens, cost, **kwargs)


def audit_run(experiment_name: str, config: Dict, seed: int, results: Dict = None) -> None:
    """Convenience function to log experimental run."""
    logger = get_audit_logger()
    logger.log_run(experiment_name, config, seed, results)


def audit_summary() -> Dict[str, Any]:
    """Get audit summary."""
    logger = get_audit_logger()
    return logger.get_summary()