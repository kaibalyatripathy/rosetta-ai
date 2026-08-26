import requests
import json
import time

url = "http://127.0.0.1:8000/translate"
payload = {
    "source_code": """
function bubbleSort(arr) {
    const n = arr.length;
    for(let i = 0; i < n; i++) {
        for(let j = 0; j < n - i - 1; j++) {
            if(arr[j] > arr[j + 1]) {
                let temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
    return arr;
}
""",
    "source_lang": "javascript",
    "target_lang": "cpp",
    "algorithm_name": "unknown"
}

print("Sending request...")
response = requests.post(url, json=payload)
data = response.json()
print("Result:")
print("Pass rate:", data.get("pass_rate"))
print("Is syntax valid:", data.get("is_syntax_valid"))
print("Refactored code:")
print(data.get("target_code"))
