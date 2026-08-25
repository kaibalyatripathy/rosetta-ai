/**
 * Rosetta AI Web Interface Controller (Phase 16)
 */

const PRESETS = {
    binary_search: {
        lang: 'python',
        algo: 'binary_search',
        code: `def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1`
    },
    bubble_sort: {
        lang: 'python',
        algo: 'bubble_sort',
        code: `def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr`
    },
    factorial: {
        lang: 'python',
        algo: 'factorial_recursive',
        code: `def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)`
    },
    fibonacci: {
        lang: 'python',
        algo: 'fibonacci_iterative',
        code: `def fibonacci(n):
    if n <= 0: return 0
    if n == 1: return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b`
    }
};

let currentAlgorithm = "binary_search";

document.addEventListener('DOMContentLoaded', () => {
    loadPreset('binary_search');
});

function loadPreset(presetKey) {
    const p = PRESETS[presetKey];
    if (p) {
        document.getElementById('src-code').value = p.code;
        document.getElementById('src-lang').value = p.lang;
        currentAlgorithm = p.algo;
    }
}

async function handleTranslate() {
    const srcCode = document.getElementById('src-code').value.trim();
    const srcLang = document.getElementById('src-lang').value;
    const tgtLang = document.getElementById('tgt-lang').value;

    if (!srcCode) {
        alert("Please enter or paste source code snippet.");
        return;
    }
    if (srcLang === tgtLang) {
        alert("Source and Target languages must be different.");
        return;
    }

    const btnTranslate = document.getElementById('btn-translate');
    const spinner = document.getElementById('loading-spinner');
    const progress = document.getElementById('pipeline-progress');
    const resultsPanel = document.getElementById('results-panel');
    const reportSection = document.getElementById('report-section');

    btnTranslate.disabled = true;
    spinner.classList.remove('hidden');
    progress.classList.remove('hidden');
    resultsPanel.classList.add('hidden');
    reportSection.classList.add('hidden');

    try {
        const response = await fetch('/translate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                source_code: srcCode,
                source_lang: srcLang,
                target_lang: tgtLang,
                algorithm_name: currentAlgorithm
            })
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || "Translation request failed.");
        }

        const data = await response.json();
        renderResults(data);

    } catch (err) {
        alert(`Translation Error: ${err.message}`);
    } finally {
        btnTranslate.disabled = false;
        spinner.classList.add('hidden');
        progress.classList.add('hidden');
    }
}

function renderResults(data) {
    // 1. Target Code Display
    document.getElementById('tgt-code').textContent = data.target_code;

    // 2. Score & Grade
    const scoreVal = data.composite_score.toFixed(1);
    const scoreElem = document.getElementById('res-score');
    scoreElem.textContent = scoreVal;

    const gradeElem = document.getElementById('res-grade');
    gradeElem.textContent = data.quality_grade;
    gradeElem.style.borderColor = getGradeColor(data.composite_score);
    gradeElem.style.color = getGradeColor(data.composite_score);
    gradeElem.style.backgroundColor = getGradeColor(data.composite_score) + '20';

    document.getElementById('res-verdict').textContent = data.intent_summary;

    // 3. Sandbox Status
    const passText = `${data.passed_inputs} / ${data.total_inputs} Inputs Passed (${data.pass_rate.toFixed(1)}%)`;
    document.getElementById('res-sandbox-text').textContent = passText;
    document.getElementById('res-syntax-valid').textContent = data.is_syntax_valid ? "Yes" : "No";
    document.getElementById('res-equiv-score').textContent = `${data.score_equiv.toFixed(1)} / 45.0`;

    // 4. Risks Container
    const risksContainer = document.getElementById('res-risks-container');
    risksContainer.innerHTML = '';

    if (data.flagged_risks && data.flagged_risks.length > 0) {
        data.flagged_risks.forEach(r => {
            const div = document.createElement('div');
            div.className = 'risk-item';
            div.innerHTML = `<strong>[${r.severity}] ${r.category}:</strong> ${r.description}`;
            risksContainer.appendChild(div);
        });
    } else {
        risksContainer.innerHTML = '<div class="no-risks-item">✓ Zero semantic risks flagged</div>';
    }

    // 5. Complexity
    document.getElementById('res-src-comp').textContent = data.source_complexity;
    document.getElementById('res-tgt-comp').textContent = data.target_complexity;

    const compStatus = document.getElementById('res-comp-status');
    if (data.score_complexity > 0) {
        compStatus.textContent = "✓ Complexity Preserved";
        compStatus.style.color = "var(--grade-excellent)";
    } else {
        compStatus.textContent = "⚠️ Complexity Degraded";
        compStatus.style.color = "var(--grade-failing)";
    }

    // 6. Full Markdown Report
    document.getElementById('report-markdown-text').textContent = data.markdown_report;

    // Unhide panels
    document.getElementById('results-panel').classList.remove('hidden');
    document.getElementById('report-section').classList.remove('hidden');
}

function getGradeColor(score) {
    if (score >= 90.0) return "#10B981";
    if (score >= 70.0) return "#3B82F6";
    if (score >= 45.0) return "#F59E0B";
    return "#EF4444";
}

function copyTargetCode() {
    const targetCode = document.getElementById('tgt-code').textContent;
    navigator.clipboard.writeText(targetCode).then(() => {
        const btn = document.getElementById('btn-copy');
        btn.textContent = '✓ Copied!';
        setTimeout(() => { btn.textContent = '📋 Copy'; }, 2000);
    });
}

function toggleMarkdownReport() {
    const reportContent = document.getElementById('report-content');
    reportContent.classList.toggle('hidden');
}
