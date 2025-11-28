# Contributing to Heart Disease Prediction

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the Heart Disease Prediction project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please be respectful and constructive in all interactions.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, dependency versions)
- **Error messages or logs**

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:

- **Clear use case** for the enhancement
- **Detailed description** of the proposed functionality
- **Why this enhancement would be useful** to most users

### Code Contributions

#### Areas for Contribution

- **Model improvements**: New algorithms, hyperparameter tuning, ensemble methods
- **Feature engineering**: Creating new features, feature selection
- **Documentation**: Improving README, adding tutorials, code comments
- **Testing**: Adding unit tests, integration tests
- **Visualization**: New plots, dashboards, interpretability tools
- **Deployment**: API development, containerization, cloud deployment
- **Bug fixes**: Resolving issues, improving error handling

## Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Create a new branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use meaningful variable and function names
- Maximum line length: 100 characters (for code), 80 for docstrings
- Use type hints where appropriate

### Documentation

- All functions should have docstrings following Google style:
  ```python
  def function_name(param1: str, param2: int) -> bool:
      """
      Brief description of what the function does.
      
      Args:
          param1 (str): Description of param1
          param2 (int): Description of param2
          
      Returns:
          bool: Description of return value
          
      Raises:
          ValueError: When and why this error is raised
      """
      pass
  ```

- Update README.md if you add new features or change usage
- Add comments for complex logic

### Code Organization

- Keep functions small and focused (single responsibility)
- Avoid code duplication
- Use meaningful module and file names
- Place utility functions in appropriate modules

### Testing

- Test your changes before submitting
- Ensure existing functionality still works
- Add tests for new features (if applicable)

## Pull Request Process

1. **Update documentation** for any changed functionality
2. **Update the README.md** with details of changes if needed
3. **Ensure your code follows** the coding standards above
4. **Test your changes** thoroughly
5. **Commit with clear messages**:
   ```bash
   git commit -m "Add feature: clear description of what was added"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request** with:
   - Clear title describing the change
   - Detailed description of what was changed and why
   - Reference to any related issues
   - Screenshots (if UI changes)

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
Describe how you tested your changes

## Checklist
- [ ] Code follows project style guidelines
- [ ] Documentation has been updated
- [ ] Changes have been tested
- [ ] All existing tests pass
```

## Questions?

If you have questions about contributing, feel free to:
- Open an issue with the "question" label
- Reach out to the maintainers

## Recognition

Contributors will be acknowledged in the project README. Thank you for helping improve this project!

---

**Happy Contributing!** 🎉
