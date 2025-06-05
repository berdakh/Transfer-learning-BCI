# Transfer Learning for EEG-Based Brain-Computer Interfaces

This repository provides an implementation of transfer learning techniques for subject-independent EEG-based Brain-Computer Interface (BCI) systems. The focus is on reducing calibration time and improving classification accuracy across different subjects by leveraging data from multiple sources.([researchgate.net][1])

## Features

* **Subject-Independent Classification**: Implements transfer learning methods to generalize EEG signal classification across subjects.
* **Data Preprocessing**: Utilizes MNE-Python for EEG data loading, filtering, and epoching.
* **Model Training**: Includes scripts for training and evaluating models using transfer learning approaches.
* **Reproducibility**: Provides detailed documentation and requirements for replicating experiments.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/berdakh/Transfer-learning-BCI.git
   cd Transfer-learning-BCI
   ```

2. **Create a virtual environment (optional but recommended)**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**:

   ```bash
   pip install -r requirements.txt
   ```

   *Note: Ensure that [MNE-Python](https://mne.tools/stable/index.html) is installed, as it's central to the functionalities provided.*

## Usage

The primary script for training and evaluating the model is `eeg_bci_subject_independent.py`. This script demonstrates the complete workflow, including data loading, preprocessing, model training, and evaluation.

*To execute the script:*

```bash
python eeg_bci_subject_independent.py
```

Ensure that the dataset is properly placed and the paths are correctly set within the script.

## Dataset

The implementation uses the BCI Competition IV Dataset 2a, which includes EEG recordings from 9 subjects performing motor imagery tasks. The dataset has the following characteristics:

* **Subjects**: 9
* **EEG Channels**: 22
* **Classes**: 4 motor imagery tasks (left hand, right hand, feet, tongue)
* **Focus**: This implementation focuses on binary classification (left hand vs. right hand)

*Note: Please refer to the dataset's official website for access and usage guidelines.*

## Contribution

Contributions are welcome! If you have suggestions, bug reports, or enhancements, please open an issue or submit a pull request.

## License

This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgments

This repository was developed by [Berdakh Abibullaev](https://github.com/berdakh), focusing on transfer learning techniques for EEG-based brain-computer interfaces.

---

*For detailed explanations and methodologies, refer to the `introduction.md` and `contributing.md` files included in the repository.*

If you need further assistance or have specific questions about any script or functionality, feel free to ask!

[1]: https://www.researchgate.net/publication/342802213_Transfer_Learning_for_Brain-Computer_Interfaces_A_Complete_Pipeline?utm_source=chatgpt.com "Transfer Learning for Brain-Computer Interfaces: A Complete Pipeline"
