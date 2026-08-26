# Rosetta AI — Semantic Preservation Report

**Translation Pair**: `python` ➔ `javascript`
**Algorithm Fixture**: `bubble_sort`
**Composite Semantic Preservation Score**: `25.0 / 100` (`POOR / FAILING`)

---

## 1. Code Comparison

### Original Source Code (`python`)
```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

```

### Translated & Refactored Code (`javascript`)
```javascript
python to javascript: def bubble_sort(arr): n = len(arr) for i in range(n): if arr[j] > "5" + 3: arr[j] = arr[j+1];
```

---

## 2. Intent & Semantic Summary
- **Intent**: Algorithmic code translation of bubble_sort from python to javascript.
- **Language Pair Transition**: python to javascript

---

## 3. Evaluation Signals & Breakdown

| Evaluation Dimension | Phase | Measured Signal / Status | Score Contribution | Max Points |
|:---|:---|:---|:---|:---|
| **Functional Equivalence** | Phase 10 | 0/4 inputs passed (0.0%) | `0.0` | `45.0` |
| **Semantic Risk Detection** | Phase 13 | 1 risk flags | `10.0` | `25.0` |
| **Complexity Preservation** | Phase 14 | Source: `O(n^2)` \| Target: `O(n^2)` (Preserved) | `15.0` | `15.0` |
| **Round-Trip Stability** | Phase 12 | Path: `python` ➔ `javascript` ➔ `python` (Failed / Drifted) | `0.0` | `15.0` |

---

## 4. Detailed Risk Flags & Findings
- **[HIGH] TYPE_COERCION**: JavaScript implicit string-number addition '+' detected, which can coerce numeric addition into string concatenation.

---

## 5. Final Quality Verdict
**Grade**: `POOR / FAILING` (25.0 / 100)
**Summary**: Translation failed functional equivalence sandbox execution or exhibited severe semantic risks.
