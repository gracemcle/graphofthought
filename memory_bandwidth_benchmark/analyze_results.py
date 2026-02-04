#!/usr/bin/env python3
"""
Analyze benchmark results and generate penalty factor tables
for use in Roofline-based performance prediction models.

Usage:
    ./bin/bandwidth_benchmark 256 | python analyze_results.py
    
Or save results and analyze later:
    ./bin/bandwidth_benchmark 256 > results.txt
    python analyze_results.py results.txt
"""

import sys
import re
import json
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class BenchmarkResult:
    name: str
    bandwidth_gbps: float
    time_ms: float
    efficiency_pct: float


def parse_results(text: str) -> tuple[float, List[BenchmarkResult]]:
    """Parse benchmark output and return (peak_bw, results)."""
    results = []
    peak_bw = None
    
    # Find peak bandwidth
    peak_match = re.search(r'Peak.*Bandwidth:\s*([\d.]+)\s*GB/s', text)
    if peak_match:
        peak_bw = float(peak_match.group(1))
    
    # Parse result lines
    # Format: "Kernel Name                        BW (GB/s)    Time (ms)   Efficiency"
    pattern = r'^([A-Za-z0-9\s\(\)=,_-]+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%'
    
    for line in text.split('\n'):
        match = re.match(pattern, line.strip())
        if match:
            name = match.group(1).strip()
            bw = float(match.group(2))
            time = float(match.group(3))
            eff = float(match.group(4))
            results.append(BenchmarkResult(name, bw, time, eff))
    
    return peak_bw, results


def compute_penalty_factors(results: List[BenchmarkResult], 
                            baseline_name: str = "Coalesced Copy (float4)") -> Dict[str, float]:
    """Compute penalty factors relative to baseline."""
    baseline = next((r for r in results if r.name == baseline_name), None)
    if not baseline:
        baseline = results[0]  # Use first result as fallback
    
    factors = {}
    for r in results:
        penalty = baseline.bandwidth_gbps / r.bandwidth_gbps
        factor = 1.0 / penalty
        factors[r.name] = {
            'penalty': penalty,
            'factor': factor,
            'bandwidth_gbps': r.bandwidth_gbps,
            'efficiency_pct': r.efficiency_pct
        }
    
    return factors


def generate_lookup_table(factors: Dict[str, dict]) -> str:
    """Generate Python code for a penalty factor lookup table."""
    
    code = '''# Auto-generated penalty factor table
# Generated from benchmark results
# Use these factors to predict memory-bound kernel performance

PENALTY_FACTORS = {
'''
    
    # Categorize by pattern type
    categories = {
        'coalescing': [],
        'access_pattern': [],
        'optimization': [],
    }
    
    for name, data in factors.items():
        if 'stride' in name.lower():
            categories['coalescing'].append((name, data))
        elif 'random' in name.lower() or 'gather' in name.lower() or 'scatter' in name.lower():
            categories['access_pattern'].append((name, data))
        else:
            categories['optimization'].append((name, data))
    
    code += '    # Coalescing penalties\n'
    for name, data in categories['coalescing']:
        key = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('=', '_')
        code += f"    '{key}': {data['factor']:.4f},  # {data['bandwidth_gbps']:.1f} GB/s\n"
    
    code += '\n    # Access pattern penalties\n'
    for name, data in categories['access_pattern']:
        key = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        code += f"    '{key}': {data['factor']:.4f},  # {data['bandwidth_gbps']:.1f} GB/s\n"
    
    code += '\n    # Other patterns\n'
    for name, data in categories['optimization']:
        key = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('=', '_')
        code += f"    '{key}': {data['factor']:.4f},  # {data['bandwidth_gbps']:.1f} GB/s\n"
    
    code += '}\n'
    
    return code


def generate_json_output(peak_bw: float, factors: Dict[str, dict]) -> str:
    """Generate JSON output for easy consumption by other tools."""
    output = {
        'peak_bandwidth_gbps': peak_bw,
        'factors': factors
    }
    return json.dumps(output, indent=2)


def main():
    # Read input
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            text = f.read()
    else:
        text = sys.stdin.read()
    
    # Parse results
    peak_bw, results = parse_results(text)
    
    if not results:
        print("No benchmark results found in input.", file=sys.stderr)
        print("Expected format from bandwidth_benchmark output.", file=sys.stderr)
        sys.exit(1)
    
    # Compute penalty factors
    factors = compute_penalty_factors(results)
    
    # Print summary
    print("=" * 70)
    print("PENALTY FACTOR ANALYSIS")
    print("=" * 70)
    print(f"Peak Theoretical Bandwidth: {peak_bw:.2f} GB/s" if peak_bw else "Peak BW: Unknown")
    print(f"Number of benchmarks: {len(results)}")
    print()
    
    # Print table
    print(f"{'Kernel':<40} {'BW (GB/s)':>10} {'Penalty':>10} {'Factor':>10}")
    print("-" * 70)
    for name, data in factors.items():
        print(f"{name:<40} {data['bandwidth_gbps']:>10.2f} {data['penalty']:>10.2f}x {data['factor']:>10.4f}")
    print()
    
    # Generate code
    print("=" * 70)
    print("GENERATED LOOKUP TABLE (Python)")
    print("=" * 70)
    print(generate_lookup_table(factors))
    
    # Save JSON
    json_output = generate_json_output(peak_bw, factors)
    with open('penalty_factors.json', 'w') as f:
        f.write(json_output)
    print(f"\nJSON output saved to: penalty_factors.json")


if __name__ == '__main__':
    main()
