# Error Model

A preliminary code release of a dynamic error model for satellite departures.

## Overview

Observation-minus-background (O–B) departures often deviate from a Gaussian distribution in cloudy conditions, which violates the core Gaussianity assumption in a data assimilation system. Major weather centers (e.g. ECMWF) have studied this challenge with all-sky infrared and microwave channels, developing methods to “Gaussianize” the errors.

This code applies a similar idea to the SEVIRI 0.6 µm channel (visible reflectance). By quantifying cloud amount using both synthetic and observed data to develope the error model. Although reflectance values are naturally bounded (and thus exhibit their own non-Gaussian behavior), I use the 0.6 µm channel here as a clear, illustrative example of the cloud-dependent approach.  

## Files

See **CONTENTS.md** for a complete list and descriptions.

## Getting Started

1. **Clone or download** this repository.  
2. **Install dependencies**:
   ```bash
   pip install numpy pandas matplotlib scipy
3. **Explore the example notebook:**
   ```bash
   jupyter notebook run_error_model.ipynb

## Acknowledgments
- Refactored from my original code with the help of GitHub Copilot.

- Contributions and feedback are warmly welcome!

## License

© 2025 Sandy Chkeir. Released under the MIT License. See the LICENSE file for details.

## Citation

If you use this code in your work, please cite:

> Chkeir, S. (2025). Non-Gaussian-errors: A cloud-dependent error model (Version 1.0) [Source code]. GitHub. https://github.com/Sandy887/Satellite_DA/Non-Gaussian-errors

### BibTex:
```bibtex
@software{error_model_utils_2025,
  author    = {Chkeir, S.},
  title     = {Non-Gaussian-errors: A cloud-dependent error model},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/Sandy887/Satellite_DA/Non-Gaussian-errors},
  note      = {Source code}
}