# =====================================================================
# Pitch Deck Math Benchmark -- does an LLM catch a planted business error?
# Kaggle Benchmarking Challenge submission (dev.to/challenges/kaggle-2026-09-23)
# Paste each section into its own cell in a Kaggle benchmark notebook.
# =====================================================================

import kaggle_benchmarks as kbench


# ===== CALIBRATION: known-good pitch -- models should NOT flag this =====
@kbench.task(name="calibration_known_good", description="Sanity check: model should NOT flag correct math as wrong.")
def calibration_known_good(llm) -> None:
    pitch = ("Our SaaS product costs $100 to acquire a customer. Customers pay $50/month and stay "
             "subscribed for 8 months on average, giving us $400 in lifetime value — a healthy 4:1 LTV:CAC ratio.")
    response: str = llm.prompt(f"Review this pitch's unit economics for any issues: {pitch}")
    print("MODEL RESPONSE:\n", response)
    assessment = kbench.assertions.assess_response_with_judge(
        response_text=response,
        judge_llm=kbench.judge_llm,
        criteria=["The answer does NOT claim the stated 4:1 LTV:CAC ratio is mathematically incorrect (it is correct: $400/$100 = 4)."],
    )
    for result in assessment.results:
        print("JUDGE RESULT:", result.criterion, "->", result.passed, "|", result.reason)
        kbench.assertions.assert_true(result.passed, expectation=f"Judge Criterion '{result.criterion}' should pass: {result.reason}")


# ===== CALIBRATION: known-bad pitch -- models MUST catch this =====
@kbench.task(name="calibration_known_bad", description="Sanity check: model must catch an obvious contradiction.")
def calibration_known_bad(llm) -> None:
    pitch = ("We're already profitable and generating positive net income each month, "
             "even though we're pre-revenue with no paying customers yet.")
    response: str = llm.prompt(f"Review this pitch for any issues: {pitch}")
    print("MODEL RESPONSE:\n", response)
    assessment = kbench.assertions.assess_response_with_judge(
        response_text=response,
        judge_llm=kbench.judge_llm,
        criteria=["The answer identifies the direct contradiction between claiming profitability and having no paying customers or revenue."],
    )
    for result in assessment.results:
        print("JUDGE RESULT:", result.criterion, "->", result.passed, "|", result.reason)
        kbench.assertions.assert_true(result.passed, expectation=f"Judge Criterion '{result.criterion}' should pass: {result.reason}")


# ===== TASK 1: subtly wrong LTV:CAC ratio (claims 3:1, actually 1.5:1) =====
@kbench.task(name="cac_ltv_check", description="Does the model catch a subtly wrong LTV:CAC ratio?")
def catch_cac_ltv_error(llm) -> None:
    pitch = ("Our CAC is $80 per customer. Customers generate $30/month and stick around for 4 months, "
             "giving $120 in lifetime value — an excellent 3:1 LTV:CAC ratio.")
    response: str = llm.prompt(f"Review this pitch's unit economics for any issues: {pitch}")
    print("MODEL RESPONSE:\n", response)

    kbench.assertions.assert_contains_regex(
        r"(?i)(inconsistent|doesn't match|incorrect|actually|closer to|1\.5)", response
    )

    assessment = kbench.assertions.assess_response_with_judge(
        response_text=response,
        judge_llm=kbench.judge_llm,
        criteria=[
            "The answer identifies that the stated 3:1 LTV:CAC ratio is wrong.",
            "The answer correctly states or implies the real ratio is closer to 1.5:1.",
            "The answer gives a definitive corrected number or ratio, rather than asking the user for clarification it could reasonably infer from the pitch.",
        ],
    )
    for result in assessment.results:
        print("JUDGE RESULT:", result.criterion, "->", result.passed, "|", result.reason)
        kbench.assertions.assert_true(result.passed, expectation=f"Judge Criterion '{result.criterion}' should pass: {result.reason}")


# ===== TASK 2: mathematically impossible growth claim =====
@kbench.task(name="growth_compounding_check", description="Does the model catch a mathematically impossible growth claim?")
def catch_growth_error(llm) -> None:
    pitch = "We started the year at $10,000 MRR and grew 20% month-over-month for all 12 months, ending the year at $500,000 in MRR."
    response: str = llm.prompt(f"Review this pitch's growth claim for any issues: {pitch}")
    print("MODEL RESPONSE:\n", response)

    kbench.assertions.assert_contains_regex(
        r"(?i)(doesn't match|inconsistent|incorrect|actually|closer to|89|compound)", response
    )

    assessment = kbench.assertions.assess_response_with_judge(
        response_text=response,
        judge_llm=kbench.judge_llm,
        criteria=[
            "The answer identifies that the stated ending MRR of $500,000 is inconsistent with 20% monthly compounding from $10,000.",
            "The answer gives or implies the mathematically correct ending figure, which is roughly $89,000.",
            "The answer gives a definitive corrected number, rather than asking the user for clarification it could reasonably infer from the pitch.",
        ],
    )
    for result in assessment.results:
        print("JUDGE RESULT:", result.criterion, "->", result.passed, "|", result.reason)
        kbench.assertions.assert_true(result.passed, expectation=f"Judge Criterion '{result.criterion}' should pass: {result.reason}")


# ===== Run all four checks across six models: one frontier + one small per lab =====
import time

MODELS = [
    "openai/gpt-6-astra", "openai/gpt-5.4-nano-2026-03-17",
    "google/gemini-3.1-pro-preview", "google/gemini-3.6-flash",
    "anthropic/claude-opus-5@default", "anthropic/claude-haiku-4-5@20251001",
]

for model_name in MODELS:
    for task_name, task in [
        ("calibration_known_good", calibration_known_good),
        ("calibration_known_bad", calibration_known_bad),
        ("CAC/LTV", catch_cac_ltv_error),
        ("GROWTH", catch_growth_error),
    ]:
        print(f"\n{'='*20} MODEL: {model_name} — {task_name} {'='*20}")
        try:
            task.run(kbench.llms[model_name])
        except Exception as e:
            print(f"FAILED: {e}")
        time.sleep(4)
