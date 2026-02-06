---
name: ultralytics-contribute-code
description: Contribute code to the Ultralytics YOLO open-source project. Use when the user wants to submit bug fixes, features, or improvements following Ultralytics coding standards and contribution guidelines.
license: AGPL-3.0
metadata:
  author: Burhan-Q
  version: "1.0"
  ultralytics-version: ">=8.0.0"
---

# Contribute Code to Ultralytics

## When to use this skill

Use this skill when you need to:
- Submit bug fixes to Ultralytics YOLO
- Add new features or enhancements
- Improve documentation or code quality
- Follow Ultralytics contribution guidelines

## Prerequisites

- Python ≥3.8 with PyTorch ≥1.8 installed
- Git installed and configured
- GitHub account
- Familiarity with pull request workflow

## Contribution Workflow

### 1. Fork and Clone

```bash
# Fork the repository on GitHub (click "Fork" button)
# Then clone your fork
git clone https://github.com/YOUR-USERNAME/ultralytics.git
cd ultralytics

# Add upstream remote
git remote add upstream https://github.com/ultralytics/ultralytics.git
```

### 2. Create a Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch with descriptive name
git checkout -b fix-issue-123  # for bug fixes
# or
git checkout -b add-feature-xyz  # for features
```

### 3. Set Up Development Environment

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### 4. Make Your Changes

Follow these guidelines:

#### Important Guidelines

- Limit changes to target a specific, narrow purpose.
    - Good Example: Removed unused variable in `utils/__init__.py`. Lines changed (+0/-1)
        - Reason: Single purpose and simple change.
    - Bad Example: Simplified code. Lines changed (+1,789/-1541)
        - Reason: Unclear purpose and broad scope.
- Favor is given for pull requests that simplify code, remove redundant code, and/or remove code; without detriment to performance or functionality.
    - Good Example: Combined two functions performing the same task. Lines changed (+43/-57); all tests pass no regressions.
        - Reason: Adds simplification without causing issues.
    - Bad Example: Added new functionality for detection models with simplification. Lines changed (+330/-541); segmentation task broken, regressions for other model tasks.
        - Reason: Changes break other functionality and result in performance regression.
- Ensure all code changes or additions are clear and easy to read.
    - Good Example: `even_numbers = [num for num in range(5) if num % 2 == 0]`
        - Reason: Clear names and builtin function use
    - Bad Example: `e = [float(u) for u in "12345" if str(int(u) / 2).rsplit(".", 1)[-1] == "0"]`
        - Reason: Highly complex, numerous type changes, and difficult to read logic
- Always optimize for code execution speed, but assess the trade off with added lines, complexity, and/or reduced readability.
    - Good Example: `if number in (1, 2, 3, 3, 5, 1, 2, 2):` changed to `if number in {1, 2, 3, 5}:`
        - Reason: Easy to read, better performance, more concise.
    - Bad Example: `result = expensive_function()` changed to `subprocess.run(expensive_function)`
        - Reason: Added complexity by using `subprocess`
- Where possible, minimize the amount of changes or additions (fewest lines of code added or modified). Removing unneeded code of any amount is welcome, given all other guidelines are followed.
    - Good Example: Lines changed (+10/-33)
        - Reason: Removed more code than added, assume compliance with all other guidelines
    - Bad Example: Lines changed (+100/-3)
        - Reason: Added much more code than removed, assume compliance with all other guidelines
- If a code pattern or design is changed, ensure consistency across all instances.
    - Good Example: All instances of `or` converted to `|` in all files
        Reason: Pattern applied across all instances in entire codebase
    - Bad Example: All instances of `or` converted to `|` in a single file
        - Reason: Pattern change applicable to entire codebase, change limited in scope
- Always execute `ruff format` and `ruff check --fix` on all modified files before opening pull request.
- Ensure ALL tests pass with zero regressions.

#### Code Style

**PEP 8 Compliance:**
- Use 4 spaces for indentation (not tabs)
- Maximum line length: 120 characters
- Use meaningful variable and function names
- Add type hints where appropriate

**Example:**

```python
def process_image(img: np.ndarray, size: int = 640) -> torch.Tensor:
    """
    Process image for YOLO inference.
    
    Args:
        img: Input image as numpy array
        size: Target size for resizing
        
    Returns:
        Preprocessed image tensor
    """
    # Implementation
    return tensor
```

#### Docstring Format

Use Google-style docstrings:

```python
def train_model(data: str, epochs: int = 100, batch: int = 16) -> dict:
    """
    Train a YOLO model.
    
    Args:
        data (str): Path to dataset YAML configuration file.
        epochs (int): Number of training epochs. Default is 100.
        batch (int): Batch size for training. Default is 16.
        
    Returns:
        (dict): Training metrics including mAP, loss, etc.
        
    Raises:
        FileNotFoundError: If data file doesn't exist.
        ValueError: If epochs or batch are invalid.
        
    Examples:
        >>> metrics = train_model("coco8.yaml", epochs=50, batch=32)
        >>> print(f"mAP: {metrics['mAP']}")
    """
    # Implementation
    return metrics
```

#### Import Organization

```python
# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import cv2
import numpy as np
import torch
import torch.nn as nn

# Local imports
from ultralytics.utils import LOGGER
from ultralytics.nn.modules import Conv, C2f
```

#### Logging

Use the LOGGER instead of print():

```python
from ultralytics.utils import LOGGER

# Good
LOGGER.info("Training started")
LOGGER.warning("Low confidence detections")
LOGGER.error("Failed to load model")

# Bad
print("Training started")  # Don't use print()
```

### 5. Test Your Changes

```bash
# Run relevant tests
pytest tests/test_engine.py -v

# Run specific test
pytest tests/test_engine.py::test_train -v

# Run all tests (can be slow)
pytest tests/

# Skip slow tests during development
pytest -m "not slow"
```

#### Format and Lint Your Code

```bash
# Format code with ruff
ruff format .

# Check for linting issues
ruff check .

# Auto-fix linting issues where possible
ruff check . --fix
```

#### Add Tests for New Features

```python
# tests/test_my_feature.py
import pytest
from ultralytics import YOLO


def test_my_new_feature():
    """Test my new feature."""
    model = YOLO("yolo26n.pt")
    result = model.my_new_feature(param="value")
    assert result is not None
    assert isinstance(result, dict)


@pytest.mark.slow
def test_my_feature_comprehensive():
    """Comprehensive test for my feature."""
    # More extensive testing
    pass
```

### 6. Commit Your Changes

```bash
# Stage changes
git add file1.py file2.py

# Commit with descriptive message
git commit -m "Fix: Resolve memory leak in dataloader (issue #123)"
# or
git commit -m "Add: Support for custom augmentation pipeline"
# or
git commit -m "Update: Improve training documentation"
```

**Commit Message Format:**
- **Fix:** Bug fixes
- **Add:** New features
- **Update:** Documentation or minor improvements
- **Refactor:** Code restructuring without functionality changes
- **Test:** Adding or updating tests

### 7. Push and Create Pull Request

```bash
# Push to your fork
git push origin fix-issue-123
```

On GitHub:
1. Navigate to your fork
2. Click "Compare & pull request"
3. Fill out the PR template with:
   - Clear title describing the change
   - Description of what was changed and why
   - Link to related issue (if applicable)
   - Screenshots (for UI changes)
   - Checklist completion

**PR Template Example:**

```markdown
## Description
Fixes memory leak in dataloader that occurred when using large batch sizes.

## Changes
- Added proper cleanup in `__del__` method
- Fixed shared memory handling
- Added unit test to prevent regression

## Related Issue
Closes #123

## Checklist
- [x] I have read the [Contributing Guide](https://docs.ultralytics.com/help/contributing/)
- [x] I have added tests that prove my fix is effective
- [x] I have updated documentation if needed
- [x] I have run `ruff format .` and `ruff check .` locally
- [x] I have run `pytest` locally and all tests pass
- [x] My code follows the project's style guidelines
```

### 8. Respond to Review Feedback

- Address all reviewer comments
- Make requested changes in new commits
- Push updates to the same branch
- Be respectful and professional

```bash
# Make requested changes
git add modified_files.py
git commit -m "Address review feedback: improve error handling"
git push origin fix-issue-123
```

## Code Guidelines

### Performance Considerations

```python
# Use numpy/torch operations instead of loops
# Bad
result = []
for item in items:
    result.append(item * 2)

# Good
result = np.array(items) * 2
```

### Error Handling

```python
# Provide informative error messages
# Bad
try:
    model.load(path)
except:
    pass

# Good
from pathlib import Path

try:
    model.load(path)
except FileNotFoundError:
    LOGGER.error(f"Model file not found: {path}")
    raise
except Exception as e:
    LOGGER.error(f"Failed to load model: {e}")
    raise
```

### Type Hints

```python
from typing import Optional, Union, List
import numpy as np
import torch

def predict(
    source: Union[str, Path, np.ndarray],
    conf: float = 0.25,
    iou: float = 0.7,
    device: Optional[Union[int, str]] = None
) -> List[torch.Tensor]:
    """Run prediction."""
    pass
```

## Testing Requirements

1. **Unit Tests:**
   - Test individual functions/methods
   - Use pytest fixtures
   - Mock external dependencies

2. **Integration Tests:**
   - Test complete workflows
   - Mark as `@pytest.mark.slow` if time-consuming

3. **Test Coverage:**
   - Aim for >80% coverage on new code
   - Run `pytest --cov=ultralytics` to check

## Documentation Standards

### Code Documentation

```python
def complex_function(
    param1: str,
    param2: int = 10,
    param3: Optional[dict] = None
) -> tuple:
    """
    Brief one-line description.
    
    More detailed description explaining what the function does,
    its purpose, and any important implementation details.
    
    Args:
        param1 (str): Description of param1.
        param2 (int): Description of param2. Defaults to 10.
        param3 (dict, optional): Description of param3.
        
    Returns:
        (tuple): Description of return value.
            - element1 (type): Description
            - element2 (type): Description
            
    Examples:
        >>> result = complex_function("test", param2=20)
        >>> print(result)
        (element1, element2)
        
    Raises:
        ValueError: If param1 is empty.
        TypeError: If param2 is not an integer.
        
    Note:
        This function requires XYZ to be installed.
        
    See Also:
        related_function(): Related functionality.
    """
    # Implementation
    pass
```

### Updating Existing Documentation

If your changes affect user-facing features:

1. **Update relevant `.md` files** in `docs/en/`
2. **Add code examples** showing new functionality
3. **Update API reference** if adding/modifying public methods
4. **Include screenshots** for UI changes

**Example Documentation Update:**

````markdown
## New Feature: Custom Augmentation

You can now apply custom augmentation pipelines during training:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.train(
    data="coco8.yaml",
    augment_pipeline="custom_aug.yaml",
    epochs=50
)
```

See [Augmentation Guide](augmentation.md) for details.