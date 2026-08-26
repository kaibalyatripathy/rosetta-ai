import sys
sys.path.append("E:/rosetta-ai")

from src.verification.differential_test import verify_equivalence, attach_test_driver

cpp_code = """
#include <vector>
using namespace std;

vector<int> bubbleSort(vector<int> arr) {
    int n = arr.size();
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n - i - 1; j++) {
            if(arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
    return arr;
}
"""

print(attach_test_driver(cpp_code, "cpp", [1, 5, 2]))

res = verify_equivalence(
    source_code="function bubbleSort(arr) { return arr; }",
    source_lang="javascript",
    target_code=cpp_code,
    target_lang="cpp",
    test_inputs=[[1, 5, 2]],
    algorithm_name="bubble_sort"
)

print(res)
