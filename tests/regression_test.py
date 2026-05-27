import pandas as pd
import ollama

# Load test cases
df = pd.read_csv("tests/test_cases.csv")

print("\n===== REGRESSION TESTING =====\n")

passed = 0
failed = 0

for index, row in df.iterrows():

    prompt = row["prompt"]

    expected_keywords = row["expected_keywords"].split(",")

    print(f"\nTesting Prompt: {prompt}")

    # Generate response
    response = ollama.chat(
        model='llama3.2',
        messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ]
    )

    answer = response['message']['content']

    # Check keywords
    test_pass = True

    for keyword in expected_keywords:

        if keyword.lower() not in answer.lower():
            test_pass = False

    # Results
    if test_pass:
        print("✅ PASSED")
        passed += 1
    else:
        print("❌ FAILED")
        failed += 1

print("\n===== FINAL RESULTS =====")

print(f"\nPassed: {passed}")
print(f"Failed: {failed}")