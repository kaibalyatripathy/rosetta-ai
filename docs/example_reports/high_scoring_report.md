# Rosetta AI — Semantic Preservation Report

**Translation Pair**: `python` $\rightarrow$ `javascript`
**Algorithm Fixture**: `binary_search`
**Composite Semantic Preservation Score**: `100.0 / 100` (`EXCELLENT`)

---

## 1. Code Comparison

### Original Source Code (`python`)
```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

```

### Translated & Refactored Code (`javascript`)
```javascript
function binarySearch(arr, target) {
    let left = 0, right = arr.length - 1;
    while (left <= right) {
        let mid = Math.floor((left + right) / 2);
        if (arr[mid] === target) return mid;
        if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}

```

---

## 2. Intent & Semantic Summary
- **Intent**: Algorithmic code translation of binary_search from python to javascript.
- **Language Pair Transition**: python to javascript

---

## 3. Evaluation Signals & Breakdown

| Evaluation Dimension | Phase | Measured Signal / Status | Score Contribution | Max Points |
|:---|:---|:---|:---|:---|
| **Functional Equivalence** | Phase 10 | 4/4 inputs passed (100.0%) | `45.0` | `45.0` |
| **Semantic Risk Detection** | Phase 13 | Zero risks flagged | `25.0` | `25.0` |
| **Complexity Preservation** | Phase 14 | Source: `O(log n)` \| Target: `O(log n)` (Preserved) | `15.0` | `15.0` |
| **Round-Trip Stability** | Phase 12 | Path: `python` $\rightarrow$ `javascript` $\rightarrow$ `python` (Passed) | `15.0` | `15.0` |

---

## 4. Detailed Risk Flags & Findings
- *No semantic risks or edge-case divergences flagged.*

---

## 5. Final Quality Verdict
**Grade**: `EXCELLENT` (100.0 / 100)
**Summary**: Translation cleanly preserves functional semantics, risk boundaries, and computational complexity.
