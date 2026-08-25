"""
Unit tests for Sandboxed Compilation & Execution across all 20 canonical algorithm fixtures.
"""

import pytest
from src.sandbox.runner import run_in_sandbox, ExecutionResult

# 20 canonical algorithm fixtures covering Python, Java, C++, and JavaScript
FIXTURES = [
    {
        "name": "factorial_recursive",
        "lang": "python",
        "expected": "120",
        "code": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)\n\nprint(factorial(5))\n"
    },
    {
        "name": "binary_search",
        "lang": "java",
        "expected": "3",
        "code": """public class Solution {
    public static int search(int[] arr, int target) {
        int left = 0, right = arr.length - 1;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (arr[mid] == target) return mid;
            if (arr[mid] < target) left = mid + 1;
            else right = mid - 1;
        }
        return -1;
    }
    public static void main(String[] args) {
        int[] arr = {1, 3, 5, 7, 9};
        System.out.println(search(arr, 7));
    }
}
"""
    },
    {
        "name": "fibonacci_iterative",
        "lang": "cpp",
        "expected": "55",
        "code": """#include <iostream>

long long fibonacci(int n) {
    if (n <= 0) return 0;
    if (n == 1) return 1;
    long long a = 0, b = 1;
    for (int i = 2; i <= n; ++i) {
        long long temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}

int main() {
    std::cout << fibonacci(10) << std::endl;
    return 0;
}
"""
    },
    {
        "name": "bubble_sort",
        "lang": "javascript",
        "expected": "1 2 5 8 9",
        "code": """function bubbleSort(arr) {
    let n = arr.length;
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                let temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
    return arr;
}

console.log(bubbleSort([5, 2, 8, 1, 9]).join(" "));
"""
    },
    {
        "name": "gcd_euclidean",
        "lang": "python",
        "expected": "6",
        "code": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n\nprint(gcd(48, 18))\n"
    },
    {
        "name": "insertion_sort",
        "lang": "cpp",
        "expected": "1 2 3 4",
        "code": """#include <iostream>
#include <vector>

void insertionSort(std::vector<int>& arr) {
    for (size_t i = 1; i < arr.size(); i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

int main() {
    std::vector<int> arr = {4, 3, 2, 1};
    insertionSort(arr);
    for (size_t i = 0; i < arr.size(); i++) {
        std::cout << arr[i] << (i + 1 == arr.size() ? "" : " ");
    }
    std::cout << std::endl;
    return 0;
}
"""
    },
    {
        "name": "is_prime",
        "lang": "java",
        "expected": "true",
        "code": """public class Solution {
    public static boolean isPrime(int n) {
        if (n <= 1) return false;
        for (int i = 2; i * i <= n; i++) {
            if (n % i == 0) return false;
        }
        return true;
    }
    public static void main(String[] args) {
        System.out.println(isPrime(29));
    }
}
"""
    },
    {
        "name": "linear_search",
        "lang": "javascript",
        "expected": "2",
        "code": """function linearSearch(arr, target) {
    for (let i = 0; i < arr.length; i++) {
        if (arr[i] === target) return i;
    }
    return -1;
}

console.log(linearSearch([10, 20, 30, 40], 30));
"""
    },
    {
        "name": "linked_list_node",
        "lang": "python",
        "expected": "1 -> 2 -> 3",
        "code": """class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

head = Node(1)
head.next = Node(2)
head.next.next = Node(3)

curr = head
res = []
while curr:
    res.append(str(curr.val))
    curr = curr.next

print(" -> ".join(res))
"""
    },
    {
        "name": "lru_cache_meta",
        "lang": "python",
        "expected": "10",
        "code": """class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache = {}

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        val = self.cache.pop(key)
        self.cache[key] = val
        return val

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.cap:
            oldest = next(iter(self.cache))
            self.cache.pop(oldest)
        self.cache[key] = value

cache = LRUCache(2)
cache.put(1, 10)
cache.put(2, 20)
print(cache.get(1))
"""
    },
    {
        "name": "matrix_multiplication",
        "lang": "cpp",
        "expected": "4 4 10 8",
        "code": """#include <iostream>
#include <vector>

int main() {
    std::vector<std::vector<int>> A = {{1, 2}, {3, 4}};
    std::vector<std::vector<int>> B = {{2, 0}, {1, 2}};
    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));

    for(int i=0; i<2; i++) {
        for(int j=0; j<2; j++) {
            for(int k=0; k<2; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    std::cout << C[0][0] << " " << C[0][1] << " " << C[1][0] << " " << C[1][1] << std::endl;
    return 0;
}
"""
    },
    {
        "name": "max_subarray_kadane",
        "lang": "java",
        "expected": "6",
        "code": """public class Solution {
    public static int maxSubArray(int[] nums) {
        int maxSoFar = nums[0];
        int currMax = nums[0];
        for (int i = 1; i < nums.length; i++) {
            currMax = Math.max(nums[i], currMax + nums[i]);
            maxSoFar = Math.max(maxSoFar, currMax);
        }
        return maxSoFar;
    }
    public static void main(String[] args) {
        int[] nums = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
        System.out.println(maxSubArray(nums));
    }
}
"""
    },
    {
        "name": "merge_sort",
        "lang": "javascript",
        "expected": "3 9 10 27 38 43 82",
        "code": """function mergeSort(arr) {
    if (arr.length <= 1) return arr;
    const mid = Math.floor(arr.length / 2);
    const left = mergeSort(arr.slice(0, mid));
    const right = mergeSort(arr.slice(mid));
    
    let result = [], i = 0, j = 0;
    while (i < left.length && j < right.length) {
        if (left[i] < right[j]) result.push(left[i++]);
        else result.push(right[j++]);
    }
    return result.concat(left.slice(i)).concat(right.slice(j));
}

console.log(mergeSort([38, 27, 43, 3, 9, 82, 10]).join(" "));
"""
    },
    {
        "name": "palindrome_check",
        "lang": "python",
        "expected": "True",
        "code": """def is_palindrome(s: str) -> bool:
    return s == s[::-1]

print(is_palindrome("racecar"))
"""
    },
    {
        "name": "power_exponentiation",
        "lang": "cpp",
        "expected": "1024",
        "code": """#include <iostream>

long long power(long long base, int exp) {
    long long res = 1;
    while (exp > 0) {
        if (exp % 2 == 1) res *= base;
        base *= base;
        exp /= 2;
    }
    return res;
}

int main() {
    std::cout << power(2, 10) << std::endl;
    return 0;
}
"""
    },
    {
        "name": "queue_array",
        "lang": "java",
        "expected": "10",
        "code": """import java.util.ArrayList;

public class Solution {
    static class Queue {
        private ArrayList<Integer> list = new ArrayList<>();
        public void enqueue(int val) { list.add(val); }
        public int dequeue() { return list.remove(0); }
    }

    public static void main(String[] args) {
        Queue q = new Queue();
        q.enqueue(10);
        q.enqueue(20);
        System.out.println(q.dequeue());
    }
}
"""
    },
    {
        "name": "quick_sort",
        "lang": "python",
        "expected": "1 5 7 8 9 10",
        "code": """def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print(" ".join(map(str, quicksort([10, 7, 8, 9, 1, 5]))))
"""
    },
    {
        "name": "reverse_string",
        "lang": "javascript",
        "expected": "olleh",
        "code": """function reverseString(str) {
    return str.split("").reverse().join("");
}

console.log(reverseString("hello"));
"""
    },
    {
        "name": "selection_sort",
        "lang": "cpp",
        "expected": "11 12 22 25 64",
        "code": """#include <iostream>
#include <vector>

void selectionSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        int minIdx = i;
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[minIdx]) minIdx = j;
        }
        std::swap(arr[i], arr[minIdx]);
    }
}

int main() {
    std::vector<int> arr = {64, 25, 12, 22, 11};
    selectionSort(arr);
    for (size_t i = 0; i < arr.size(); i++) {
        std::cout << arr[i] << (i + 1 == arr.size() ? "" : " ");
    }
    std::cout << std::endl;
    return 0;
}
"""
    },
    {
        "name": "stack_array",
        "lang": "java",
        "expected": "20",
        "code": """import java.util.Stack;

public class Solution {
    public static void main(String[] args) {
        Stack<Integer> stack = new Stack<>();
        stack.push(10);
        stack.push(20);
        System.out.println(stack.pop());
    }
}
"""
    }
]


@pytest.mark.parametrize("fixture", FIXTURES, ids=[f["name"] + "_" + f["lang"] for f in FIXTURES])
def test_fixture_execution(fixture):
    res: ExecutionResult = run_in_sandbox(fixture["code"], fixture["lang"])
    
    # Print real execution logs to stdout for verification
    print(f"\n--- FIXTURE: {fixture['name']} ({fixture['lang']}) ---")
    print(f"Compile Error: {res.compile_error}")
    if res.compile_error:
        print(f"Compile Stderr: {res.compile_stderr}")
    print(f"Exit Code: {res.exit_code}")
    print(f"Timed Out: {res.timed_out}")
    print(f"Stdout: {res.stdout.strip()}")
    print(f"Stderr: {res.stderr.strip()}")

    assert not res.compile_error, f"Compilation failed for {fixture['name']}: {res.compile_stderr}"
    assert res.exit_code == 0, f"Non-zero exit code for {fixture['name']}: {res.stderr}"
    assert not res.timed_out, f"Timed out for {fixture['name']}"
    assert res.stdout.strip() == fixture["expected"], f"Output mismatch for {fixture['name']}: expected '{fixture['expected']}', got '{res.stdout.strip()}'"
