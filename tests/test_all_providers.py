"""
Comprehensive test of all LLM providers individually
Tests each provider with their optimal parameters
"""

import sys
from pathlib import Path

# Add ai_scientist to path
ai_scientist_path = str(Path(__file__).parent / "ai_scientist")
sys.path.insert(0, ai_scientist_path)

from model_router import get_router, TaskType

def test_individual_providers():
    """Test each provider individually with appropriate parameters"""
    
    router = get_router()
    
    print("=== COMPREHENSIVE PROVIDER TEST ===")
    print(f"Available providers: {list(router.providers.keys())}")
    print()
    
    # Test cases for each provider
    test_results = {}
    
    # 1. OpenAI Test
    if 'openai' in router.providers:
        print("Testing OpenAI (GPT-5)...")
        try:
            # Test with temperature=1.0 (default for GPT-5)
            response = router._call_openai(
                prompt="Generate one pharmaceutical research hypothesis in 50 words.",
                system_prompt="You are a pharmaceutical researcher.",
                max_tokens=100,
                temperature=1.0,  # Use default temperature
                task_type=TaskType.HYPOTHESIS_GENERATION
            )
            print(f"[SUCCESS] OpenAI: {response.model_used}")
            print(f"   Cost: ${response.cost_cents/100:.4f}")
            print(f"   Response: {response.content[:150]}...")
            test_results['openai'] = True
        except Exception as e:
            print(f"[FAILED] OpenAI: {e}")
            test_results['openai'] = False
        print()
    
    # 2. DeepSeek Test
    if 'deepseek' in router.providers:
        print("Testing DeepSeek...")
        try:
            response = router._call_deepseek(
                prompt="Parse: 'adult severe cancer immunotherapy' -> JSON format",
                system_prompt="Extract structured data from pharmaceutical queries.",
                max_tokens=100,
                temperature=0.3,
                task_type=TaskType.BULK_PARSING
            )
            print(f"[SUCCESS] DeepSeek: {response.model_used}")
            print(f"   Cost: ${response.cost_cents/100:.4f}")
            print(f"   Response: {response.content[:150]}...")
            test_results['deepseek'] = True
        except Exception as e:
            print(f"[FAILED] DeepSeek: {e}")
            test_results['deepseek'] = False
        print()
    
    # 3. Perplexity Test
    if 'perplexity' in router.providers:
        print("Testing Perplexity (Sonar)...")
        try:
            response = router._call_perplexity(
                prompt="Review this claim with citations: 'Severe asthma affects 5% of asthma patients'",
                system_prompt="Provide objective review with web citations.",
                max_tokens=200,
                temperature=0.5,
                task_type=TaskType.OBJECTIVE_REVIEW
            )
            print(f"[SUCCESS] Perplexity: {response.model_used}")
            print(f"   Cost: ${response.cost_cents/100:.4f}")
            print(f"   Citations found: {len(response.citations) if response.citations else 0}")
            print(f"   Response: {response.content[:150]}...")
            test_results['perplexity'] = True
        except Exception as e:
            print(f"[FAILED] Perplexity: {e}")
            test_results['perplexity'] = False
        print()
    
    # 4. Anthropic Test
    if 'anthropic' in router.providers:
        print("Testing Anthropic (Claude)...")
        try:
            response = router._call_anthropic(
                prompt="Analyze this pharmaceutical scenario in context: What are the key factors for pediatric drug development success?",
                system_prompt="You are an expert pharmaceutical development consultant.",
                max_tokens=200,
                temperature=0.7,
                task_type=TaskType.LONG_CONTEXT
            )
            print(f"[SUCCESS] Anthropic: {response.model_used}")
            print(f"   Cost: ${response.cost_cents/100:.4f}")
            print(f"   Response: {response.content[:150]}...")
            test_results['anthropic'] = True
        except Exception as e:
            print(f"[FAILED] Anthropic: {e}")
            test_results['anthropic'] = False
        print()
    
    # 5. Google Test
    if 'google' in router.providers:
        print("Testing Google (Gemini)...")
        try:
            response = router._call_google(
                prompt="List 5 key factors to consider when analyzing pharmaceutical markets for investment decisions.",
                system_prompt="You are a helpful pharmaceutical research assistant.",
                max_tokens=150,
                temperature=0.5,
                task_type=TaskType.LONG_CONTEXT
            )
            print(f"[SUCCESS] Google: {response.model_used}")
            print(f"   Cost: ${response.cost_cents/100:.4f}")
            print(f"   Response: {response.content[:150]}...")
            test_results['google'] = True
        except Exception as e:
            print(f"[FAILED] Google: {e}")
            test_results['google'] = False
        print()
    
    # Summary
    print("=== TEST SUMMARY ===")
    working_providers = [p for p, status in test_results.items() if status]
    failed_providers = [p for p, status in test_results.items() if not status]
    
    print(f"[WORKING] providers ({len(working_providers)}): {', '.join(working_providers)}")
    if failed_providers:
        print(f"[FAILED] providers ({len(failed_providers)}): {', '.join(failed_providers)}")
    
    report = router.get_usage_report()
    print(f"\nTotal cost: ${report['total_cost_cents']/100:.4f}")
    print(f"Budget remaining: ${report['budget_remaining_cents']/100:.4f}")
    
    # Recommendations
    print("\n=== RECOMMENDATIONS ===")
    if test_results.get('openai', False):
        print("• OpenAI (GPT-5) ready for complex reasoning tasks")
    elif 'openai' in router.providers:
        print("• OpenAI needs configuration adjustment (check model name/parameters)")
    
    if test_results.get('deepseek', False):
        print("• DeepSeek ready for cost-efficient bulk processing") 
    
    if test_results.get('perplexity', False):
        print("• Perplexity ready for citation-backed reviews")
    
    if test_results.get('anthropic', False):
        print("• Claude ready for long-context analysis")
    
    if test_results.get('google', False):
        print("• Gemini ready for very long context tasks")
    
    return len(working_providers) >= 2  # Need at least 2 working providers

if __name__ == "__main__":
    success = test_individual_providers()
    if success:
        print("\n[READY] Multi-LLM system is ready for AI Scientist implementation!")
    else:
        print("\n[ISSUES] Need to fix provider configurations before proceeding.")