#!/usr/bin/env python3
# A/B Testing for Customer Retention Strategies

import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
import os
import random
import warnings
warnings.filterwarnings('ignore')

# Import common utilities
from utils.data_preprocessing import load_telco_data

# Create directories if they don't exist
os.makedirs('../results', exist_ok=True)
os.makedirs('../images', exist_ok=True)
os.makedirs('../docs', exist_ok=True)

def plan_ab_test(baseline_conversion_rate, minimum_detectable_effect, alpha=0.05, power=0.8):
    """
    Plan an A/B test by calculating required sample size and power
    
    Parameters:
    -----------
    baseline_conversion_rate : float
        Current conversion/retention rate (proportion between 0 and 1)
    minimum_detectable_effect : float
        Smallest effect size that is meaningful to detect (proportion between 0 and 1)
    alpha : float
        Significance level (Type I error rate)
    power : float
        Statistical power (1 - Type II error rate)
        
    Returns:
    --------
    dict
        Dictionary containing test parameters and required sample sizes
    """
    print(f"Planning A/B Test with baseline rate: {baseline_conversion_rate:.4f}")
    print(f"Minimum detectable effect: {minimum_detectable_effect:.4f}")
    
    # Calculate effect size (Cohen's h for proportions)
    effect_size = 2 * np.arcsin(np.sqrt(baseline_conversion_rate + minimum_detectable_effect)) - 2 * np.arcsin(np.sqrt(baseline_conversion_rate))
    
    # Calculate sample size
    analysis = TTestIndPower()
    sample_size = analysis.solve_power(
        effect_size=effect_size,
        power=power,
        alpha=alpha,
        ratio=1.0,
        alternative='two-sided'
    )
    
    # Round up to the nearest whole number
    sample_size = int(np.ceil(sample_size))
    
    # Calculate total sample size (both groups)
    total_sample_size = 2 * sample_size
    
    # Create results dictionary
    results = {
        'baseline_rate': baseline_conversion_rate,
        'minimum_detectable_effect': minimum_detectable_effect,
        'target_rate': baseline_conversion_rate + minimum_detectable_effect,
        'effect_size': effect_size,
        'alpha': alpha,
        'power': power,
        'sample_size_per_group': sample_size,
        'total_sample_size': total_sample_size
    }
    
    print(f"Required sample size per group: {sample_size}")
    print(f"Total sample size required: {total_sample_size}")
    
    # Plot power curve
    sample_sizes = np.linspace(10, sample_size * 2, 100)
    powers = [analysis.solve_power(
        effect_size=effect_size,
        nobs1=n,
        alpha=alpha,
        ratio=1.0,
        alternative='two-sided'
    ) for n in sample_sizes]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, powers)
    plt.axhline(y=power, color='r', linestyle='--', alpha=0.5, label=f'Target Power = {power}')
    plt.axvline(x=sample_size, color='g', linestyle='--', alpha=0.5, label=f'Required Sample Size = {sample_size}')
    plt.title('Power Analysis')
    plt.xlabel('Sample Size per Group')
    plt.ylabel('Statistical Power')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('../images/ab_test_power_curve.png')
    
    return results

def simulate_ab_test(n_per_group, control_rate, treatment_effect):
    """
    Simulate the results of an A/B test
    
    Parameters:
    -----------
    n_per_group : int
        Number of samples per test group
    control_rate : float
        Conversion/retention rate in the control group
    treatment_effect : float
        Expected effect size (increase in conversion rate) for treatment group
        
    Returns:
    --------
    tuple
        (control_results, treatment_results, test_analysis)
    """
    print(f"Simulating A/B test with {n_per_group} samples per group")
    print(f"Control rate: {control_rate:.4f}, Treatment effect: {treatment_effect:.4f}")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Control group: simulate customers with baseline conversion rate
    control_successes = np.random.binomial(1, control_rate, n_per_group)
    
    # Treatment group: simulate customers with treatment effect
    treatment_rate = control_rate + treatment_effect
    treatment_successes = np.random.binomial(1, treatment_rate, n_per_group)
    
    # Calculate results
    control_success_count = np.sum(control_successes)
    treatment_success_count = np.sum(treatment_successes)
    
    control_success_rate = control_success_count / n_per_group
    treatment_success_rate = treatment_success_count / n_per_group
    
    # Perform statistical test (Z-test for proportions)
    successes = np.array([control_success_count, treatment_success_count])
    samples = np.array([n_per_group, n_per_group])
    
    z_stat, p_value = proportions_ztest(successes, samples)
    
    # Calculate confidence intervals
    control_ci_low, control_ci_high = proportion_confint(control_success_count, n_per_group)
    treatment_ci_low, treatment_ci_high = proportion_confint(treatment_success_count, n_per_group)
    
    # Determine if result is statistically significant
    is_significant = p_value < 0.05
    
    # Calculate relative lift
    relative_lift = (treatment_success_rate - control_success_rate) / control_success_rate * 100
    
    # Compile results
    control_results = {
        'group': 'Control',
        'sample_size': n_per_group,
        'successes': control_success_count,
        'rate': control_success_rate,
        'ci_low': control_ci_low,
        'ci_high': control_ci_high
    }
    
    treatment_results = {
        'group': 'Treatment',
        'sample_size': n_per_group,
        'successes': treatment_success_count,
        'rate': treatment_success_rate,
        'ci_low': treatment_ci_low,
        'ci_high': treatment_ci_high
    }
    
    test_analysis = {
        'z_statistic': z_stat,
        'p_value': p_value,
        'is_significant': is_significant,
        'absolute_difference': treatment_success_rate - control_success_rate,
        'relative_lift': relative_lift
    }
    
    return control_results, treatment_results, test_analysis

def visualize_ab_test_results(control_results, treatment_results, test_analysis):
    """
    Visualize the results of an A/B test
    
    Parameters:
    -----------
    control_results : dict
        Results from the control group
    treatment_results : dict
        Results from the treatment group
    test_analysis : dict
        Statistical analysis of the test
    """
    # Create DataFrame for visualization
    results_df = pd.DataFrame([control_results, treatment_results])
    
    # Bar chart of conversion rates with confidence intervals
    plt.figure(figsize=(10, 6))
    
    # Create bar plot
    ax = sns.barplot(x='group', y='rate', data=results_df, palette=['blue', 'green'])
    
    # Add confidence intervals
    for i, row in results_df.iterrows():
        ax.errorbar(i, row['rate'], yerr=[[row['rate']-row['ci_low']], [row['ci_high']-row['rate']]], 
                   fmt='none', c='black', capsize=5)
    
    # Add labels and title
    plt.title('A/B Test Results: Retention Rates by Group')
    plt.ylabel('Retention Rate')
    plt.xlabel('')
    
    # Add annotations
    significance_label = "Statistically Significant" if test_analysis['is_significant'] else "Not Statistically Significant"
    p_value_format = f"p = {test_analysis['p_value']:.4f}"
    lift_format = f"{test_analysis['relative_lift']:.2f}% Lift"
    
    plt.annotate(significance_label, xy=(0.5, 0.95), xycoords='axes fraction', 
                ha='center', fontsize=12, color='darkred' if test_analysis['is_significant'] else 'darkgray')
    plt.annotate(p_value_format, xy=(0.5, 0.9), xycoords='axes fraction', ha='center', fontsize=10)
    plt.annotate(lift_format, xy=(0.5, 0.85), xycoords='axes fraction', ha='center', fontsize=10)
    
    # Show rates on bars
    for i, row in results_df.iterrows():
        ax.text(i, row['rate']/2, f"{row['rate']:.4f}", ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('../images/ab_test_results.png')
    
    # Create a more detailed visualization showing sample size and success counts
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Conversion rates
    plt.subplot(2, 2, 1)
    sns.barplot(x='group', y='rate', data=results_df, palette=['blue', 'green'])
    plt.title('Retention Rates')
    plt.ylabel('Rate')
    plt.xlabel('')
    
    # Subplot 2: Sample sizes
    plt.subplot(2, 2, 2)
    sns.barplot(x='group', y='sample_size', data=results_df, palette=['blue', 'green'])
    plt.title('Sample Sizes')
    plt.ylabel('Count')
    plt.xlabel('')
    
    # Subplot 3: Success counts
    plt.subplot(2, 2, 3)
    sns.barplot(x='group', y='successes', data=results_df, palette=['blue', 'green'])
    plt.title('Success Counts')
    plt.ylabel('Count')
    plt.xlabel('')
    
    # Subplot 4: Text summary
    plt.subplot(2, 2, 4)
    plt.axis('off')
    summary_text = (
        f"A/B Test Results Summary\n\n"
        f"Control Group Rate: {control_results['rate']:.4f}\n"
        f"Treatment Group Rate: {treatment_results['rate']:.4f}\n\n"
        f"Absolute Difference: {test_analysis['absolute_difference']:.4f}\n"
        f"Relative Lift: {test_analysis['relative_lift']:.2f}%\n\n"
        f"p-value: {test_analysis['p_value']:.4f}\n"
        f"Statistically Significant: {'Yes' if test_analysis['is_significant'] else 'No'}"
    )
    plt.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('../images/ab_test_detailed_results.png')

def define_retention_strategies():
    """
    Define different retention strategies for A/B testing based on customer segments
    
    Returns:
    --------
    list
        List of retention strategy dictionaries
    """
    # Define strategies
    strategies = [
        {
            'id': 'discount_offer',
            'name': 'Discount Offer',
            'description': 'Offer a 20% discount on monthly service for 3 months',
            'target_segment': 'Price-Sensitive Churners',
            'baseline_retention': 0.65,
            'expected_lift': 0.08,
            'cost_per_customer': 45  # Average monthly charge * 0.2 * 3 months
        },
        {
            'id': 'service_upgrade',
            'name': 'Free Service Upgrade',
            'description': 'Offer a free service upgrade (e.g., higher internet speed) for 6 months',
            'target_segment': 'High-Value At-Risk',
            'baseline_retention': 0.70,
            'expected_lift': 0.06,
            'cost_per_customer': 60  # Cost of providing the upgraded service
        },
        {
            'id': 'tech_support',
            'name': 'Free Premium Technical Support',
            'description': 'Provide 1 year of free premium technical support',
            'target_segment': 'Fiber Optic Users',
            'baseline_retention': 0.58,
            'expected_lift': 0.09,
            'cost_per_customer': 50  # Cost of providing premium support
        },
        {
            'id': 'contract_incentive',
            'name': 'Contract Incentive',
            'description': 'Offer a one-time bill credit for signing a 1-year contract',
            'target_segment': 'Month-to-Month Customers',
            'baseline_retention': 0.55,
            'expected_lift': 0.12,
            'cost_per_customer': 70  # One-time bill credit
        },
        {
            'id': 'loyalty_program',
            'name': 'Enhanced Loyalty Program',
            'description': 'Enroll customers in an enhanced loyalty program with exclusive benefits',
            'target_segment': 'Long-Term Customers',
            'baseline_retention': 0.85,
            'expected_lift': 0.03,
            'cost_per_customer': 25  # Cost of program benefits
        }
    ]
    
    return strategies

def run_ab_test_simulation_for_all_strategies():
    """
    Run A/B test simulations for all defined retention strategies
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with simulation results for all strategies
    """
    # Get strategies
    strategies = define_retention_strategies()
    
    # Store results
    all_results = []
    
    # Run simulations for each strategy
    for strategy in strategies:
        print(f"\n{'='*50}")
        print(f"Simulating A/B test for: {strategy['name']}")
        print(f"Target segment: {strategy['target_segment']}")
        print(f"{'='*50}")
        
        # Plan the test
        test_plan = plan_ab_test(
            baseline_conversion_rate=strategy['baseline_retention'],
            minimum_detectable_effect=strategy['expected_lift'],
            alpha=0.05,
            power=0.8
        )
        
        # Run the simulation
        control, treatment, analysis = simulate_ab_test(
            n_per_group=test_plan['sample_size_per_group'],
            control_rate=strategy['baseline_retention'],
            treatment_effect=strategy['expected_lift']
        )
        
        # Visualize results
        visualize_ab_test_results(control, treatment, analysis)
        
        # Save strategy-specific visualizations
        plt.figure(figsize=(10, 6))
        
        groups = [control['group'], treatment['group']]
        rates = [control['rate'], treatment['rate']]
        
        plt.bar(groups, rates, color=['blue', 'green'])
        plt.title(f"Retention Rates: {strategy['name']}")
        plt.ylabel('Retention Rate')
        plt.xlabel('Test Group')
        
        # Add text labels
        plt.text(0, control['rate']/2, f"{control['rate']:.4f}", ha='center', fontsize=12, color='white')
        plt.text(1, treatment['rate']/2, f"{treatment['rate']:.4f}", ha='center', fontsize=12, color='white')
        
        # Add lift annotation
        plt.annotate(
            f"{analysis['relative_lift']:.2f}% Lift",
            xy=(0.5, max(rates) * 1.1),
            xycoords='data',
            ha='center',
            fontsize=12,
            color='darkred' if analysis['is_significant'] else 'darkgray'
        )
        
        plt.tight_layout()
        plt.savefig(f"../images/ab_test_{strategy['id']}.png")
        
        # Calculate ROI
        avg_customer_lifetime_value = 1000  # Example value - would be calculated from actual data
        retention_improvement = analysis['absolute_difference']
        value_of_improvement = retention_improvement * avg_customer_lifetime_value
        roi = (value_of_improvement - strategy['cost_per_customer']) / strategy['cost_per_customer'] * 100
        
        # Store results
        all_results.append({
            'strategy_id': strategy['id'],
            'strategy_name': strategy['name'],
            'target_segment': strategy['target_segment'],
            'baseline_retention': strategy['baseline_retention'],
            'treatment_retention': treatment['rate'],
            'absolute_difference': analysis['absolute_difference'],
            'relative_lift': analysis['relative_lift'],
            'p_value': analysis['p_value'],
            'is_significant': analysis['is_significant'],
            'sample_size_required': test_plan['sample_size_per_group'],
            'cost_per_customer': strategy['cost_per_customer'],
            'estimated_roi': roi
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_df.to_csv('../results/ab_test_strategy_results.csv', index=False)
    
    # Create comparison visualization
    plt.figure(figsize=(12, 8))
    
    # Plot relative lift by strategy
    plt.subplot(2, 1, 1)
    bars = plt.bar(results_df['strategy_name'], results_df['relative_lift'])
    
    # Color bars based on statistical significance
    for i, is_sig in enumerate(results_df['is_significant']):
        bars[i].set_color('green' if is_sig else 'lightgray')
    
    plt.title('Relative Lift by Retention Strategy')
    plt.ylabel('Relative Lift (%)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add significance markers
    for i, (is_sig, lift) in enumerate(zip(results_df['is_significant'], results_df['relative_lift'])):
        marker = '✓' if is_sig else '✗'
        color = 'green' if is_sig else 'red'
        plt.text(i, lift + 0.5, marker, ha='center', color=color, fontsize=12)
    
    # Plot ROI by strategy
    plt.subplot(2, 1, 2)
    bars = plt.bar(results_df['strategy_name'], results_df['estimated_roi'])
    
    # Color bars based on ROI value
    for i, roi in enumerate(results_df['estimated_roi']):
        bars[i].set_color('green' if roi > 100 else 'orange' if roi > 0 else 'red')
    
    plt.title('Estimated ROI by Retention Strategy')
    plt.ylabel('ROI (%)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../images/ab_test_strategy_comparison.png')
    
    # Generate comprehensive report
    with open('../docs/ab_testing_report.md', 'w') as f:
        f.write("# A/B Testing Report for Customer Retention Strategies\n\n")
        
        f.write("## Executive Summary\n\n")
        
        # Sort strategies by ROI
        top_strategies = results_df.sort_values('estimated_roi', ascending=False)
        
        f.write("Top performing strategies by estimated ROI:\n\n")
        for i, (_, row) in enumerate(top_strategies.iterrows(), 1):
            significance = "✓" if row['is_significant'] else "✗"
            f.write(f"{i}. **{row['strategy_name']}** ({row['target_segment']}): {row['relative_lift']:.2f}% lift {significance}, {row['estimated_roi']:.1f}% ROI\n")
        
        f.write("\n## Test Results by Strategy\n\n")
        
        for _, row in results_df.iterrows():
            f.write(f"### {row['strategy_name']}\n\n")
            f.write(f"**Target Segment**: {row['target_segment']}\n\n")
            f.write(f"**Baseline Retention Rate**: {row['baseline_retention']:.4f}\n\n")
            f.write(f"**Treatment Retention Rate**: {row['treatment_retention']:.4f}\n\n")
            f.write(f"**Absolute Improvement**: {row['absolute_difference']:.4f}\n\n")
            f.write(f"**Relative Lift**: {row['relative_lift']:.2f}%\n\n")
            f.write(f"**Statistical Significance**: {'Yes' if row['is_significant'] else 'No'} (p={row['p_value']:.4f})\n\n")
            f.write(f"**Required Sample Size per Group**: {row['sample_size_required']}\n\n")
            f.write(f"**Cost per Customer**: ${row['cost_per_customer']:.2f}\n\n")
            f.write(f"**Estimated ROI**: {row['estimated_roi']:.1f}%\n\n")
            
            roi_assessment = "Excellent" if row['estimated_roi'] > 200 else "Good" if row['estimated_roi'] > 100 else "Moderate" if row['estimated_roi'] > 0 else "Poor"
            
            f.write(f"**Assessment**: {roi_assessment} return on investment")
            
            if not row['is_significant']:
                f.write(", but results are not statistically significant. Consider increasing sample size or modifying the strategy.")
            elif row['estimated_roi'] <= 0:
                f.write(". This strategy is not economically viable despite showing a statistically significant lift.")
            else:
                f.write(". This strategy is recommended for implementation.")
            
            f.write("\n\n---\n\n")
        
        f.write("## Recommendations\n\n")
        
        # Identify successful strategies
        successful_strategies = results_df[(results_df['is_significant']) & (results_df['estimated_roi'] > 0)]
        
        if len(successful_strategies) > 0:
            f.write("### Recommended for Implementation\n\n")
            
            for _, row in successful_strategies.sort_values('estimated_roi', ascending=False).iterrows():
                f.write(f"- **{row['strategy_name']}** for {row['target_segment']}: {row['relative_lift']:.2f}% lift, {row['estimated_roi']:.1f}% ROI\n")
            
            f.write("\n### Implementation Plan\n\n")
            f.write("1. **Prioritize by ROI**: Implement strategies with the highest ROI first.\n")
            f.write("2. **Segment Targeting**: Ensure proper customer segmentation to target the right customers with each strategy.\n")
            f.write("3. **Monitoring**: Continuously monitor the performance of implemented strategies through key retention metrics.\n")
            f.write("4. **Feedback Loop**: Collect customer feedback on the retention initiatives to refine strategies.\n")
        else:
            f.write("None of the tested strategies showed both statistical significance and positive ROI. Recommendations:\n\n")
            f.write("1. **Redesign Strategies**: Modify the approach to reduce costs or increase effectiveness.\n")
            f.write("2. **Increase Test Sample Size**: Some strategies may show significance with larger samples.\n")
            f.write("3. **Explore New Approaches**: Consider testing entirely different retention strategies.\n")
    
    return results_df

def main():
    """Main function to run the A/B testing simulation pipeline"""
    print("A/B Testing for Customer Retention Strategies")
    print("="*50)
    
    # Run simulations for all strategies
    results = run_ab_test_simulation_for_all_strategies()
    
    # Print summary
    print("\nResults Summary:")
    print("-" * 50)
    
    # Sort strategies by ROI
    sorted_results = results.sort_values('estimated_roi', ascending=False)
    
    for _, row in sorted_results.iterrows():
        significance = "✓" if row['is_significant'] else "✗"
        roi_color = "\033[92m" if row['estimated_roi'] > 100 else "\033[93m" if row['estimated_roi'] > 0 else "\033[91m"
        reset_color = "\033[0m"
        
        print(f"{row['strategy_name']} ({row['target_segment']})")
        print(f"  Lift: {row['relative_lift']:.2f}% {significance}")
        print(f"  ROI: {roi_color}{row['estimated_roi']:.1f}%{reset_color}")
        print("-" * 50)
    
    print("\nDetailed report saved to '../docs/ab_testing_report.md'")
    print("Visualizations saved to '../images/'")

if __name__ == "__main__":
    main()