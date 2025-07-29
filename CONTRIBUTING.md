# Contributing to Photonic Neural Network Foundry

We welcome contributions to the photonic-nn-foundry project! This document provides guidelines for contributing.

## 🚀 Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/photonic-nn-foundry.git
   cd photonic-nn-foundry
   ```
3. **Set up development environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements-dev.txt
   ```
4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## 🔄 Development Workflow

### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Run tests locally**:
   ```bash
   pytest
   ```

4. **Run code quality checks**:
   ```bash
   black src/ tests/
   isort src/ tests/
   flake8 src/ tests/
   mypy src/
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

### Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Adding or updating tests
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

## 🧪 Testing

- Write tests for new functionality in the `tests/` directory
- Ensure all tests pass: `pytest`
- Maintain test coverage above 80%
- Include both unit and integration tests

## 📝 Documentation

- Update docstrings for new functions/classes
- Add examples to demonstrate usage
- Update README.md if needed
- Follow [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)

## 🐛 Reporting Issues

When reporting issues, please include:

- Python version and OS
- Steps to reproduce
- Expected vs. actual behavior
- Relevant error messages/logs
- Minimal code example if applicable

## 📋 Code Style

We use:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Pre-commit hooks will enforce these automatically.

## 🔬 Photonic Computing Guidelines

When contributing photonic-specific features:

- Reference relevant academic papers
- Include physical parameter units (nm, pJ, ps)
- Consider fabrication constraints
- Test with realistic device parameters
- Document assumptions about photonic components

## 🤝 Community

- Be respectful and inclusive
- Help others learn photonic computing concepts
- Share knowledge and best practices
- Follow our [Code of Conduct](CODE_OF_CONDUCT.md)

## 📞 Getting Help

- Open an issue for bugs or feature requests
- Start discussions for architectural questions
- Join our community channels (links in README)

Thank you for contributing to the future of photonic computing! 🌟