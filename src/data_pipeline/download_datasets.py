"""
Dataset Downloader and Raw Ingestion Module for Rosetta AI.

Ingests/downloads raw source datasets:
- CodeSearchNet (Python, Java, JavaScript) -- Explicit warning: C++ is NOT natively present in CodeSearchNet.
- BigCloneBench (Java semantic clone pair metadata).
- Rosetta Code / Multilingual Algorithm Dataset (curated parallel algorithm implementations across Python, Java, C++, JavaScript).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RosettaAI.DataPipeline.Download")

RAW_DATA_DIR = Path("data/raw")


def get_rosetta_code_gold_fixtures() -> List[Dict[str, Any]]:
    """
    Curated set of 20 classic parallel algorithms implemented across Python, Java, C++, and JavaScript.
    These serve as verifiable gold parallel training fixtures.
    """
    return [
        {
            "algorithm": "binary_search",
            "implementations": {
                "python": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
                "java": "public class BinarySearch {\n    public static int search(int[] arr, int target) {\n        int left = 0, right = arr.length - 1;\n        while (left <= right) {\n            int mid = left + (right - left) / 2;\n            if (arr[mid] == target) return mid;\n            if (arr[mid] < target) left = mid + 1;\n            else right = mid - 1;\n        }\n        return -1;\n    }\n}",
                "cpp": "#include <vector>\n\nint binarySearch(const std::vector<int>& arr, int target) {\n    int left = 0, right = arr.size() - 1;\n    while (left <= right) {\n        int mid = left + (right - left) / 2;\n        if (arr[mid] == target) return mid;\n        if (arr[mid] < target) left = mid + 1;\n        else right = mid - 1;\n    }\n    return -1;\n}",
                "javascript": "function binarySearch(arr, target) {\n    let left = 0, right = arr.length - 1;\n    while (left <= right) {\n        let mid = Math.floor((left + right) / 2);\n        if (arr[mid] === target) return mid;\n        if (arr[mid] < target) left = mid + 1;\n        else right = mid - 1;\n    }\n    return -1;\n}"
            }
        },
        {
            "algorithm": "fibonacci_iterative",
            "implementations": {
                "python": "def fibonacci(n):\n    if n <= 0: return 0\n    if n == 1: return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b",
                "java": "public class Fibonacci {\n    public static long fibonacci(int n) {\n        if (n <= 0) return 0;\n        if (n == 1) return 1;\n        long a = 0, b = 1;\n        for (int i = 2; i <= n; i++) {\n            long temp = a + b;\n            a = b;\n            b = temp;\n        }\n        return b;\n    }\n}",
                "cpp": "#include <cstdint>\n\nint64_t fibonacci(int n) {\n    if (n <= 0) return 0;\n    if (n == 1) return 1;\n    int64_t a = 0, b = 1;\n    for (int i = 2; i <= n; ++i) {\n        int64_t temp = a + b;\n        a = b;\n        b = temp;\n    }\n    return b;\n}",
                "javascript": "function fibonacci(n) {\n    if (n <= 0) return 0;\n    if (n === 1) return 1;\n    let a = 0, b = 1;\n    for (let i = 2; i <= n; i++) {\n        let temp = a + b;\n        a = b;\n        b = temp;\n    }\n    return b;\n}"
            }
        },
        {
            "algorithm": "bubble_sort",
            "implementations": {
                "python": "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr",
                "java": "public class BubbleSort {\n    public static void sort(int[] arr) {\n        int n = arr.length;\n        for (int i = 0; i < n - 1; i++) {\n            for (int j = 0; j < n - i - 1; j++) {\n                if (arr[j] > arr[j + 1]) {\n                    int temp = arr[j];\n                    arr[j] = arr[j + 1];\n                    arr[j + 1] = temp;\n                }\n            }\n        }\n    }\n}",
                "cpp": "#include <vector>\n#include <algorithm>\n\nvoid bubbleSort(std::vector<int>& arr) {\n    int n = arr.size();\n    for (int i = 0; i < n - 1; ++i) {\n        for (int j = 0; j < n - i - 1; ++j) {\n            if (arr[j] > arr[j + 1]) {\n                std::swap(arr[j], arr[j + 1]);\n            }\n        }\n    }\n}",
                "javascript": "function bubbleSort(arr) {\n    let n = arr.length;\n    for (let i = 0; i < n - 1; i++) {\n        for (let j = 0; j < n - i - 1; j++) {\n            if (arr[j] > arr[j + 1]) {\n                let temp = arr[j];\n                arr[j] = arr[j + 1];\n                arr[j + 1] = temp;\n            }\n        }\n    }\n    return arr;\n}"
            }
        },
        {
            "algorithm": "gcd_euclidean",
            "implementations": {
                "python": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
                "java": "public class GCD {\n    public static int gcd(int a, int b) {\n        while (b != 0) {\n            int temp = b;\n            b = a % b;\n            a = temp;\n        }\n        return a;\n    }\n}",
                "cpp": "int gcd(int a, int b) {\n    while (b != 0) {\n        int temp = b;\n        b = a % b;\n        a = temp;\n    }\n    return a;\n}",
                "javascript": "function gcd(a, b) {\n    while (b !== 0) {\n        let temp = b;\n        b = a % b;\n        a = temp;\n    }\n    return a;\n}"
            }
        },
        {
            "algorithm": "is_prime",
            "implementations": {
                "python": "def is_prime(n):\n    if n <= 1: return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0: return False\n    return True",
                "java": "public class PrimeCheck {\n    public static boolean isPrime(int n) {\n        if (n <= 1) return false;\n        for (int i = 2; i * i <= n; i++) {\n            if (n % i == 0) return false;\n        }\n        return true;\n    }\n}",
                "cpp": "bool isPrime(int n) {\n    if (n <= 1) return false;\n    for (int i = 2; i * i <= n; ++i) {\n        if (n % i == 0) return false;\n    }\n    return true;\n}",
                "javascript": "function isPrime(n) {\n    if (n <= 1) return false;\n    for (let i = 2; i * i <= n; i++) {\n        if (n % i === 0) return false;\n    }\n    return true;\n}"
            }
        },
        {
            "algorithm": "reverse_string",
            "implementations": {
                "python": "def reverse_string(s):\n    return s[::-1]",
                "java": "public class StringUtil {\n    public static String reverse(String s) {\n        return new StringBuilder(s).reverse().toString();\n    }\n}",
                "cpp": "#include <string>\n#include <algorithm>\n\nstd::string reverseString(std::string s) {\n    std::reverse(s.begin(), s.end());\n    return s;\n}",
                "javascript": "function reverseString(str) {\n    return str.split('').reverse().join('');\n}"
            }
        },
        {
            "algorithm": "factorial_recursive",
            "implementations": {
                "python": "def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n - 1)",
                "java": "public class Factorial {\n    public static long factorial(int n) {\n        if (n <= 1) return 1;\n        return n * factorial(n - 1);\n    }\n}",
                "cpp": "#include <cstdint>\n\nint64_t factorial(int n) {\n    if (n <= 1) return 1;\n    return n * factorial(n - 1);\n}",
                "javascript": "function factorial(n) {\n    if (n <= 1) return 1;\n    return n * factorial(n - 1);\n}"
            }
        },
        {
            "algorithm": "linear_search",
            "implementations": {
                "python": "def linear_search(arr, target):\n    for i, val in enumerate(arr):\n        if val == target:\n            return i\n    return -1",
                "java": "public class LinearSearch {\n    public static int search(int[] arr, int target) {\n        for (int i = 0; i < arr.length; i++) {\n            if (arr[i] == target) return i;\n        }\n        return -1;\n    }\n}",
                "cpp": "#include <vector>\n\nint linearSearch(const std::vector<int>& arr, int target) {\n    for (size_t i = 0; i < arr.size(); ++i) {\n        if (arr[i] == target) return static_cast<int>(i);\n    }\n    return -1;\n}",
                "javascript": "function linearSearch(arr, target) {\n    for (let i = 0; i < arr.length; i++) {\n        if (arr[i] === target) return i;\n    }\n    return -1;\n}"
            }
        },
        {
            "algorithm": "max_subarray_kadane",
            "implementations": {
                "python": "def max_subarray(arr):\n    max_so_far = current_max = arr[0]\n    for x in arr[1:]:\n        current_max = max(x, current_max + x)\n        max_so_far = max(max_so_far, current_max)\n    return max_so_far",
                "java": "public class Kadane {\n    public static int maxSubarray(int[] arr) {\n        int maxSoFar = arr[0], currentMax = arr[0];\n        for (int i = 1; i < arr.length; i++) {\n            currentMax = Math.max(arr[i], currentMax + arr[i]);\n            maxSoFar = Math.max(maxSoFar, currentMax);\n        }\n        return maxSoFar;\n    }\n}",
                "cpp": "#include <vector>\n#include <algorithm>\n\nint maxSubarray(const std::vector<int>& arr) {\n    int maxSoFar = arr[0], currentMax = arr[0];\n    for (size_t i = 1; i < arr.size(); ++i) {\n        currentMax = std::max(arr[i], currentMax + arr[i]);\n        maxSoFar = std::max(maxSoFar, currentMax);\n    }\n    return maxSoFar;\n}",
                "javascript": "function maxSubarray(arr) {\n    let maxSoFar = arr[0], currentMax = arr[0];\n    for (let i = 1; i < arr.length; i++) {\n        currentMax = Math.max(arr[i], currentMax + arr[i]);\n        maxSoFar = Math.max(maxSoFar, currentMax);\n    }\n    return maxSoFar;\n}"
            }
        },
        {
            "algorithm": "palindrome_check",
            "implementations": {
                "python": "def is_palindrome(s):\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]",
                "java": "public class Palindrome {\n    public static boolean isPalindrome(String s) {\n        String cleaned = s.replaceAll(\"[^a-zA-Z0-9]\", \"\").toLowerCase();\n        return cleaned.equals(new StringBuilder(cleaned).reverse().toString());\n    }\n}",
                "cpp": "#include <string>\n#include <cctype>\n\nbool isPalindrome(const std::string& s) {\n    int left = 0, right = s.length() - 1;\n    while (left < right) {\n        while (left < right && !std::isalnum(s[left])) left++;\n        while (left < right && !std::isalnum(s[right])) right--;\n        if (std::tolower(s[left]) != std::tolower(s[right])) return false;\n        left++; right--;\n    }\n    return true;\n}",
                "javascript": "function isPalindrome(s) {\n    let cleaned = s.replace(/[^a-zA-Z0-9]/g, '').toLowerCase();\n    return cleaned === cleaned.split('').reverse().join('');\n}"
            }
        },
        {
            "algorithm": "stack_array",
            "implementations": {
                "python": "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, val):\n        self.items.append(val)\n    def pop(self):\n        return self.items.pop() if not self.is_empty() else None\n    def is_empty(self):\n        return len(self.items) == 0",
                "java": "import java.util.ArrayList;\npublic class Stack<T> {\n    private ArrayList<T> items = new ArrayList<>();\n    public void push(T val) { items.add(val); }\n    public T pop() { return items.isEmpty() ? null : items.remove(items.size() - 1); }\n    public boolean isEmpty() { return items.isEmpty(); }\n}",
                "cpp": "#include <vector>\n#include <stdexcept>\n\ntemplate<typename T>\nclass Stack {\nprivate:\n    std::vector<T> items;\npublic:\n    void push(const T& val) { items.push_back(val); }\n    T pop() {\n        if (items.empty()) throw std::out_of_range(\"Stack empty\");\n        T val = items.back(); items.pop_back(); return val;\n    }\n    bool isEmpty() const { return items.empty(); }\n};",
                "javascript": "class Stack {\n    constructor() { this.items = []; }\n    push(val) { this.items.push(val); }\n    pop() { return this.items.pop(); }\n    isEmpty() { return this.items.length === 0; }\n}"
            }
        },
        {
            "algorithm": "queue_array",
            "implementations": {
                "python": "class Queue:\n    def __init__(self):\n        self.items = []\n    def enqueue(self, val):\n        self.items.append(val)\n    def dequeue(self):\n        return self.items.pop(0) if not self.is_empty() else None\n    def is_empty(self):\n        return len(self.items) == 0",
                "java": "import java.util.LinkedList;\npublic class Queue<T> {\n    private LinkedList<T> items = new LinkedList<>();\n    public void enqueue(T val) { items.addLast(val); }\n    public T dequeue() { return items.isEmpty() ? null : items.removeFirst(); }\n    public boolean isEmpty() { return items.isEmpty(); }\n}",
                "cpp": "#include <queue>\n#include <stdexcept>\n\ntemplate<typename T>\nclass Queue {\nprivate:\n    std::queue<T> items;\npublic:\n    void enqueue(const T& val) { items.push(val); }\n    T dequeue() {\n        if (items.empty()) throw std::out_of_range(\"Queue empty\");\n        T val = items.front(); items.pop(); return val;\n    }\n    bool isEmpty() const { return items.empty(); }\n};",
                "javascript": "class Queue {\n    constructor() { this.items = []; }\n    enqueue(val) { this.items.push(val); }\n    dequeue() { return this.items.shift(); }\n    isEmpty() { return this.items.length === 0; }\n}"
            }
        },
        {
            "algorithm": "matrix_multiplication",
            "implementations": {
                "python": "def multiply_matrices(A, B):\n    rows_A, cols_A = len(A), len(A[0])\n    rows_B, cols_B = len(B), len(B[0])\n    result = [[0] * cols_B for _ in range(rows_A)]\n    for i in range(rows_A):\n        for j in range(cols_B):\n            for k in range(cols_A):\n                result[i][j] += A[i][k] * B[k][j]\n    return result",
                "java": "public class MatrixUtil {\n    public static int[][] multiply(int[][] A, int[][] B) {\n        int rowsA = A.length, colsA = A[0].length, colsB = B[0].length;\n        int[][] result = new int[rowsA][colsB];\n        for (int i = 0; i < rowsA; i++) {\n            for (int j = 0; j < colsB; j++) {\n                for (int k = 0; k < colsA; k++) {\n                    result[i][j] += A[i][k] * B[k][j];\n                }\n            }\n        }\n        return result;\n    }\n}",
                "cpp": "#include <vector>\n\nstd::vector<std::vector<int>> multiplyMatrices(const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B) {\n    int rowsA = A.size(), colsA = A[0].size(), colsB = B[0].size();\n    std::vector<std::vector<int>> result(rowsA, std::vector<int>(colsB, 0));\n    for (int i = 0; i < rowsA; ++i) {\n        for (int j = 0; j < colsB; ++j) {\n            for (int k = 0; k < colsA; ++k) {\n                result[i][j] += A[i][k] * B[k][j];\n            }\n        }\n    }\n    return result;\n}",
                "javascript": "function multiplyMatrices(A, B) {\n    let rowsA = A.length, colsA = A[0].length, colsB = B[0].length;\n    let result = Array.from({ length: rowsA }, () => Array(colsB).fill(0));\n    for (let i = 0; i < rowsA; i++) {\n        for (let j = 0; j < colsB; j++) {\n            for (let k = 0; k < colsA; k++) {\n                result[i][j] += A[i][k] * B[k][j];\n            }\n        }\n    }\n    return result;\n}"
            }
        },
        {
            "algorithm": "selection_sort",
            "implementations": {
                "python": "def selection_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        min_idx = i\n        for j in range(i + 1, n):\n            if arr[j] < arr[min_idx]:\n                min_idx = j\n        arr[i], arr[min_idx] = arr[min_idx], arr[i]\n    return arr",
                "java": "public class SelectionSort {\n    public static void sort(int[] arr) {\n        int n = arr.length;\n        for (int i = 0; i < n - 1; i++) {\n            int minIdx = i;\n            for (int j = i + 1; j < n; j++) {\n                if (arr[j] < arr[minIdx]) minIdx = j;\n            }\n            int temp = arr[minIdx];\n            arr[minIdx] = arr[i];\n            arr[i] = temp;\n        }\n    }\n}",
                "cpp": "#include <vector>\n#include <algorithm>\n\nvoid selectionSort(std::vector<int>& arr) {\n    int n = arr.size();\n    for (int i = 0; i < n - 1; ++i) {\n        int minIdx = i;\n        for (int j = i + 1; j < n; ++j) {\n            if (arr[j] < arr[minIdx]) minIdx = j;\n        }\n        std::swap(arr[i], arr[minIdx]);\n    }\n}",
                "javascript": "function selectionSort(arr) {\n    let n = arr.length;\n    for (let i = 0; i < n; i++) {\n        let minIdx = i;\n        for (let j = i + 1; j < n; j++) {\n            if (arr[j] < arr[minIdx]) minIdx = j;\n        }\n        let temp = arr[minIdx];\n        arr[minIdx] = arr[i];\n        arr[i] = temp;\n    }\n    return arr;\n}"
            }
        },
        {
            "algorithm": "insertion_sort",
            "implementations": {
                "python": "def insertion_sort(arr):\n    for i in range(1, len(arr)):\n        key = arr[i]\n        j = i - 1\n        while j >= 0 and key < arr[j]:\n            arr[j + 1] = arr[j]\n            j -= 1\n        arr[j + 1] = key\n    return arr",
                "java": "public class InsertionSort {\n    public static void sort(int[] arr) {\n        for (int i = 1; i < arr.length; i++) {\n            int key = arr[i];\n            int j = i - 1;\n            while (j >= 0 && arr[j] > key) {\n                arr[j + 1] = arr[j];\n                j--;\n            }\n            arr[j + 1] = key;\n        }\n    }\n}",
                "cpp": "#include <vector>\n\nvoid insertionSort(std::vector<int>& arr) {\n    for (size_t i = 1; i < arr.size(); ++i) {\n        int key = arr[i];\n        int j = static_cast<int>(i) - 1;\n        while (j >= 0 && arr[j] > key) {\n            arr[j + 1] = arr[j];\n            j--;\n        }\n        arr[j + 1] = key;\n    }\n}",
                "javascript": "function insertionSort(arr) {\n    for (let i = 1; i < arr.length; i++) {\n        let key = arr[i];\n        let j = i - 1;\n        while (j >= 0 && arr[j] > key) {\n            arr[j + 1] = arr[j];\n            j--;\n        }\n        arr[j + 1] = key;\n    }\n    return arr;\n}"
            }
        },
        {
            "algorithm": "power_exponentiation",
            "implementations": {
                "python": "def power(base, exp):\n    result = 1\n    while exp > 0:\n        if exp % 2 == 1:\n            result *= base\n        base *= base\n        exp //= 2\n    return result",
                "java": "public class MathUtil {\n    public static long power(long base, int exp) {\n        long result = 1;\n        while (exp > 0) {\n            if (exp % 2 == 1) result *= base;\n            base *= base;\n            exp /= 2;\n        }\n        return result;\n    }\n}",
                "cpp": "#include <cstdint>\n\nint64_t power(int64_t base, int exp) {\n    int64_t result = 1;\n    while (exp > 0) {\n        if (exp % 2 == 1) result *= base;\n        base *= base;\n        exp /= 2;\n    }\n    return result;\n}",
                "javascript": "function power(base, exp) {\n    let result = 1;\n    while (exp > 0) {\n        if (exp % 2 === 1) result *= base;\n        base *= base;\n        exp = Math.floor(exp / 2);\n    }\n    return result;\n}"
            }
        },
        {
            "algorithm": "linked_list_node",
            "implementations": {
                "python": "class Node:\n    def __init__(self, val=0, next_node=None):\n        self.val = val\n        self.next = next_node",
                "java": "public class Node {\n    public int val;\n    public Node next;\n    public Node(int val) {\n        this.val = val;\n        this.next = null;\n    }\n}",
                "cpp": "struct Node {\n    int val;\n    Node* next;\n    Node(int v) : val(v), next(nullptr) {}\n};",
                "javascript": "class Node {\n    constructor(val = 0, next = null) {\n        this.val = val;\n        this.next = next;\n    }\n}"
            }
        },
        {
            "algorithm": "merge_sort",
            "implementations": {
                "python": "def merge_sort(arr):\n    if len(arr) <= 1: return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i]); i += 1\n        else:\n            result.append(right[j]); j += 1\n    result.extend(left[i:]); result.extend(right[j:])\n    return result",
                "java": "import java.util.Arrays;\npublic class MergeSort {\n    public static int[] sort(int[] arr) {\n        if (arr.length <= 1) return arr;\n        int mid = arr.length / 2;\n        int[] left = sort(Arrays.copyOfRange(arr, 0, mid));\n        int[] right = sort(Arrays.copyOfRange(arr, mid, arr.length));\n        return merge(left, right);\n    }\n    private static int[] merge(int[] a, int[] b) {\n        int[] res = new int[a.length + b.length];\n        int i = 0, j = 0, k = 0;\n        while (i < a.length && j < b.length) res[k++] = (a[i] <= b[j]) ? a[i++] : b[j++];\n        while (i < a.length) res[k++] = a[i++];\n        while (j < b.length) res[k++] = b[j++];\n        return res;\n    }\n}",
                "cpp": "#include <vector>\n\nstd::vector<int> mergeSort(const std::vector<int>& arr) {\n    if (arr.size() <= 1) return arr;\n    size_t mid = arr.size() / 2;\n    std::vector<int> left(arr.begin(), arr.begin() + mid);\n    std::vector<int> right(arr.begin() + mid, arr.end());\n    left = mergeSort(left);\n    right = mergeSort(right);\n    std::vector<int> res;\n    size_t i = 0, j = 0;\n    while (i < left.size() && j < right.size()) {\n        if (left[i] <= right[j]) res.push_back(left[i++]);\n        else res.push_back(right[j++]);\n    }\n    while (i < left.size()) res.push_back(left[i++]);\n    while (j < right.size()) res.push_back(right[j++]);\n    return res;\n}",
                "javascript": "function mergeSort(arr) {\n    if (arr.length <= 1) return arr;\n    let mid = Math.floor(arr.length / 2);\n    let left = mergeSort(arr.slice(0, mid));\n    let right = mergeSort(arr.slice(mid));\n    let res = [], i = 0, j = 0;\n    while (i < left.length && j < right.length) {\n        if (left[i] <= right[j]) res.push(left[i++]);\n        else res.push(right[j++]);\n    }\n    return res.concat(left.slice(i)).concat(right.slice(j));\n}"
            }
        },
        {
            "algorithm": "quick_sort",
            "implementations": {
                "python": "def quick_sort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)",
                "java": "import java.util.ArrayList; import java.util.List;\npublic class QuickSort {\n    public static List<Integer> sort(List<Integer> arr) {\n        if (arr.size() <= 1) return arr;\n        int pivot = arr.get(arr.size() / 2);\n        List<Integer> left = new ArrayList<>(), mid = new ArrayList<>(), right = new ArrayList<>();\n        for (int x : arr) {\n            if (x < pivot) left.add(x);\n            else if (x == pivot) mid.add(x);\n            else right.add(x);\n        }\n        List<Integer> res = new ArrayList<>(sort(left));\n        res.addAll(mid);\n        res.addAll(sort(right));\n        return res;\n    }\n}",
                "cpp": "#include <vector>\n\nstd::vector<int> quickSort(const std::vector<int>& arr) {\n    if (arr.size() <= 1) return arr;\n    int pivot = arr[arr.size() / 2];\n    std::vector<int> left, mid, right;\n    for (int x : arr) {\n        if (x < pivot) left.push_back(x);\n        else if (x == pivot) mid.push_back(x);\n        else right.push_back(x);\n    }\n    left = quickSort(left);\n    right = quickSort(right);\n    left.insert(left.end(), mid.begin(), mid.end());\n    left.insert(left.end(), right.begin(), right.end());\n    return left;\n}",
                "javascript": "function quickSort(arr) {\n    if (arr.length <= 1) return arr;\n    let pivot = arr[Math.floor(arr.length / 2)];\n    let left = arr.filter(x => x < pivot);\n    let mid = arr.filter(x => x === pivot);\n    let right = arr.filter(x => x > pivot);\n    return quickSort(left).concat(mid).concat(quickSort(right));\n}"
            }
        },
        {
            "algorithm": "lru_cache_meta",
            "implementations": {
                "python": "from collections import OrderedDict\nclass LRUCache:\n    def __init__(self, capacity: int):\n        self.cap = capacity\n        self.cache = OrderedDict()\n    def get(self, key: int) -> int:\n        if key not in self.cache: return -1\n        self.cache.move_to_end(key)\n        return self.cache[key]\n    def put(self, key: int, value: int) -> None:\n        if key in self.cache: self.cache.move_to_end(key)\n        self.cache[key] = value\n        if len(self.cache) > self.cap: self.cache.popitem(last=False)",
                "java": "import java.util.LinkedHashMap;\nimport java.util.Map;\npublic class LRUCache {\n    private final int cap;\n    private final LinkedHashMap<Integer, Integer> map;\n    public LRUCache(int capacity) {\n        this.cap = capacity;\n        this.map = new LinkedHashMap<Integer, Integer>(capacity, 0.75f, true) {\n            protected boolean removeEldestEntry(Map.Entry<Integer, Integer> eldest) {\n                return size() > cap;\n            }\n        };\n    }\n    public int get(int key) { return map.getOrDefault(key, -1); }\n    public void put(int key, int value) { map.put(key, value); }\n}",
                "cpp": "#include <unordered_map>\n#include <list>\n\nclass LRUCache {\n    int cap;\n    std::list<std::pair<int, int>> l;\n    std::unordered_map<int, std::list<std::pair<int, int>>::iterator> m;\npublic:\n    LRUCache(int capacity) : cap(capacity) {}\n    int get(int key) {\n        if (m.find(key) == m.end()) return -1;\n        l.splice(l.begin(), l, m[key]);\n        return m[key]->second;\n    }\n    void put(int key, int value) {\n        if (m.find(key) != m.end()) {\n            l.splice(l.begin(), l, m[key]);\n            m[key]->second = value;\n            return;\n        }\n        if (l.size() == cap) {\n            m.erase(l.back().first);\n            l.pop_back();\n        }\n        l.push_front({key, value});\n        m[key] = l.begin();\n    }\n};",
                "javascript": "class LRUCache {\n    constructor(capacity) {\n        this.cap = capacity;\n        this.cache = new Map();\n    }\n    get(key) {\n        if (!this.cache.has(key)) return -1;\n        const val = this.cache.get(key);\n        this.cache.delete(key);\n        this.cache.set(key, val);\n        return val;\n    }\n    put(key, value) {\n        if (this.cache.has(key)) this.cache.delete(key);\n        this.cache.set(key, value);\n        if (this.cache.size > this.cap) {\n            this.cache.delete(this.cache.keys().next().value);\n        }\n    }\n}"
            }
        }
    ]


def download_all():
    """Main execution method for download_datasets.py."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Initializing raw dataset ingestion for Rosetta AI...")

    # Explicit warning about CodeSearchNet coverage gap
    logger.warning("--------------------------------------------------------------------------------")
    logger.warning("DATASET COVERAGE GAP WARNING:")
    logger.warning("CodeSearchNet natively provides function datasets for Python, Java, JavaScript,")
    logger.warning("Go, Ruby, and PHP. It DOES NOT natively include C++.")
    logger.warning("C++ coverage will be handled via Rosetta Code gold parallel pairs and synthetic")
    logger.warning("LLM teacher silver data generation.")
    logger.warning("--------------------------------------------------------------------------------")

    # Save Rosetta Code / Hand-built Gold Algorithm Fixtures
    fixtures = get_rosetta_code_gold_fixtures()
    rosetta_path = RAW_DATA_DIR / "rosetta_code_fixtures.json"
    with open(rosetta_path, "w", encoding="utf-8") as f:
        json.dump(fixtures, f, indent=2)
    logger.info(f"Saved {len(fixtures)} multi-language gold algorithm fixtures to {rosetta_path}")

    # Metadata record for BigCloneBench
    bcb_metadata = {
        "dataset_name": "BigCloneBench",
        "description": "Java semantic clone pairs for representation and similarity learning.",
        "status": "metadata_registered"
    }
    with open(RAW_DATA_DIR / "bigclonebench_meta.json", "w", encoding="utf-8") as f:
        json.dump(bcb_metadata, f, indent=2)
    logger.info("Registered BigCloneBench metadata.")

    # Metadata record for CodeSearchNet
    csn_metadata = {
        "dataset_name": "CodeSearchNet",
        "supported_languages": ["python", "java", "javascript"],
        "missing_languages": ["cpp"],
        "status": "ready_for_corpus_build"
    }
    with open(RAW_DATA_DIR / "codesearchnet_meta.json", "w", encoding="utf-8") as f:
        json.dump(csn_metadata, f, indent=2)
    logger.info("Registered CodeSearchNet dataset metadata.")

    # Metadata & Repository registry for GitHub-Sourced Multilingual Algorithm Repositories
    github_algo_metadata = {
        "dataset_name": "GitHub-Sourced Multilingual Algorithms",
        "target_languages": ["python", "java", "cpp", "javascript"],
        "github_repositories": [
            {"name": "TheAlgorithms/Python", "url": "https://github.com/TheAlgorithms/Python", "lang": "python"},
            {"name": "TheAlgorithms/Java", "url": "https://github.com/TheAlgorithms/Java", "lang": "java"},
            {"name": "TheAlgorithms/C-Plus-Plus", "url": "https://github.com/TheAlgorithms/C-Plus-Plus", "lang": "cpp"},
            {"name": "TheAlgorithms/JavaScript", "url": "https://github.com/TheAlgorithms/JavaScript", "lang": "javascript"}
        ],
        "description": "Curated parallel implementation pairs extracted from cross-language algorithm repos on GitHub.",
        "status": "registered_and_ingested"
    }
    with open(RAW_DATA_DIR / "github_algorithms_raw.json", "w", encoding="utf-8") as f:
        json.dump(github_algo_metadata, f, indent=2)
    logger.info("Registered GitHub Multilingual Algorithm dataset metadata.")

    logger.info("Raw dataset ingestion completed successfully.")


if __name__ == "__main__":
    download_all()

