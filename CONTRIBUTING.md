# Contributing to Distributed Training Lab

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## Development Setup

1. **Fork and clone:**
   ```bash
   git clone https://github.com/yourusername/distributed-training-lab.git
   cd distributed-training-lab
   ```

2. **Install in development mode:**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Run tests:**
   ```bash
   make test
   ```

## Contribution Guidelines

### Code Style

- Follow PEP 8
- Use type hints where possible
- Write docstrings for all public functions/classes
- Keep functions focused and small
- Use meaningful variable names

### Formatting

We use `black` for code formatting:

```bash
make format
```

### Linting

We use `flake8` for linting:

```bash
make lint
```

### Type Checking

We use `mypy` for type checking:

```bash
make type-check
```

### Testing

- Write tests for new features
- Ensure all tests pass: `make test`
- Aim for good test coverage

### Documentation

- Update README.md for user-facing changes
- Add docstrings for new functions/classes
- Update architecture docs if needed
- Add examples for new features

## Pull Request Process

1. **Create a branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes:**
   - Write code
   - Add tests
   - Update documentation

3. **Run checks:**
   ```bash
   make check  # Runs lint, type-check, and test
   ```

4. **Commit:**
   ```bash
   git commit -m "Add feature: description"
   ```

5. **Push and create PR:**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Wait for review:**
   - Address feedback
   - Update PR as needed

## Areas for Contribution

### Features

- Additional distributed strategies (DeepSpeed, ZeRO, etc.)
- Pipeline parallelism support
- Tensor parallelism support
- Better profiling tools
- Visualization utilities

### Improvements

- Performance optimizations
- Better error messages
- More comprehensive tests
- Documentation improvements
- Example notebooks

### Bug Fixes

- Report bugs via GitHub issues
- Include reproduction steps
- Fix and test thoroughly

## Testing Guidelines

### Unit Tests

- Test individual components
- Mock external dependencies
- Use fixtures for common setup

### Integration Tests

- Test full training workflows
- Test distributed setups (if possible)
- Test checkpoint save/load

### Performance Tests

- Benchmark changes
- Compare before/after
- Document performance impact

## Documentation Guidelines

### Code Documentation

- Use Google-style docstrings
- Include Args, Returns, Raises sections
- Add examples for complex functions

### User Documentation

- Keep README up to date
- Add examples for new features
- Update troubleshooting guide

### Architecture Documentation

- Update ARCHITECTURE.md for design changes
- Document new components
- Explain design decisions

## Questions?

- Open a GitHub issue for questions
- Check existing issues first
- Be specific about what you need help with

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing! 🎉

