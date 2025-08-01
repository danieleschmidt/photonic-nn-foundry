# ADR-0003: Python Package Structure

## Status
Accepted

## Context
The photonic-nn-foundry needs a clean, maintainable Python package structure that:
- Separates core functionality from CLI interfaces
- Enables easy testing and modular development
- Supports both library usage and command-line tools
- Follows Python packaging best practices
- Facilitates future extension with plugins

## Decision
Use a standard Python package structure with:
- `src/photonic_foundry/` as the main package directory
- Separate modules for core functionality (`core.py`), transpilation (`transpiler.py`), and CLI (`cli.py`)
- `pyproject.toml` for modern Python packaging
- Entry points for command-line interfaces
- Clear separation between public APIs and internal implementation

Package structure:
```
src/
└── photonic_foundry/
    ├── __init__.py      # Public API exports
    ├── core.py          # Core acceleration classes
    ├── transpiler.py    # PyTorch to Verilog conversion
    └── cli.py           # Command-line interface
```

## Consequences

### Positive
- Clear separation of concerns between modules
- Easy to test individual components in isolation
- Standard Python packaging enables pip installation
- Modular design supports future plugin architecture
- Public API clearly defined in `__init__.py`

### Negative
- Slightly more complex than flat package structure
- Requires understanding of Python packaging conventions
- Import path changes if structure evolves

### Neutral
- Standard practice for Python libraries
- Aligns with Python Enhancement Proposal (PEP) guidelines
- Familiar structure for Python developers