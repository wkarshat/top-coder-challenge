# Architecture and Design Document

## 1. Overview
Two-stage solution:
1. **Data Extraction** Analyze legacy data, discover rules via regression, rule mining, trees.
2. **Application** At runtime, dispatch request to the appropriate logic segment, ensuring per-row explainability.

## 2. Major Components
A. **Data Ingestion** Input public or private data (CSV/JSON)
B. **Rule Discovery** Fitting (linear, polynomial, piecewise, tree) + manual mining
C. **Rule Dispatcher** Apply rules by interval or type; table-driven or plugin-based
D. **Reporting** Log decisions and method used, generate visual and downloadable reports
E. **Integration API** Batch operation and API interface

## 3. Design Choices
- Interval or type-based dispatch for best legacy match
- Hybrid model: add or modify rule sets
- Modular: new rules and behaviors via Python functions
- Logging per row and with explanation

## 4. Data Flow
Input → Validation → Rule Discovery → Dispatch → Logging → Output (visuals, reports)

## 5. Technology Stack
- Python (pandas, scikit-learn, matplotlib, pwlf)
- Jupyter or Colab
- File formats CSV, JSON

# TODO / Follow-up
**Limitations**
- Manual adjustment may be needed for edge cases.
- Explainability limited to intervals.

**Follow-Up**
- Develop user-facing dashboard.
- REST API interface.
- Plugins for rule and processing changes.

**Open Questions**
- Range and types of rules
- Privacy & security requirements
- Mobile support
- Level of non-technical explainability
- Integration targets
