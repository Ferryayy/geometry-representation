# G-IR: Geometric Image Representation for Learning (Official Implementation)

This is the official implementation for the paper **G-IR: Geometric Image Representation for Learning**.

In this work, we propose an innovative geometric image representation (G-IR). Based on the theories of Optimal Transport and Quasiconformal Mapping, our method transforms the pixel intensity representation of an image into an intrinsic, reversible geometric shape representation known as the Beltrami Coefficient. This representation captures the fine-grained structure of the image and preserves structural continuity and fidelity in generative tasks, demonstrating superior performance in applications like image reconstruction and interpolation.

![Pipeline](need to add) *Figure: Overview of the G-IR generation and reconstruction pipeline.*

## Introduction

Traditional image representations (e.g., pixel intensity) often fail to explicitly capture the global and intrinsic structure of an image Our G-IR framework addresses this by:

1.  **Image to Measured Mesh**: Treating the input image's pixel intensities as a "mass" distribution.
2.  **Optimal Transport (OT)**: Computing an optimal transport map to "flatten" this non-uniform intensity distribution, resulting in an intensity-aware OT mesh
3.  **G-IR Representation**: The deformation from a standard grid to the OT mesh is uniquely represented by a quasiconformal map, whose core is a complex-valued function—the Beltrami Coefficient (BC). This BC field (which we call a μ-Image) is our proposed G-IR.
4.  **Image Reconstruction**: The G-IR contains all the geometric information needed to decode back to the original image. By solving the quasiconformal map and reversing the initial transformation, the original image can be reconstructed with high fidelity.

This repository contains the code for each core module of the G-IR framework, as well as the autoencoder (AE) models used to reproduce the experiments in the paper.

## Environment Setup

1.  **Clone the Repository**
    ```bash
    git clone xx
    cd xx
    ```

2.  **Create a Conda Environment (Recommended)**
    We recommend using Conda to manage dependencies.

    ```bash
    conda create -n g-ir python=3.9
    conda activate g-ir
    ```

3.  **Install Dependencies**
    * **C++ (for OT module)**:
        Our Optimal Transport (OT) solver is partially implemented in C++. Please ensure you have a compatible C++ compiler (e.g., GCC/G++). You will need to compile the C++ module in the `c_modules/OT` directory according to your environment.
        ```bash
        cd c_modules/OT
        # Compile based on your system, example below:
        g++ -shared -o ot_solver.so ot_solver.cpp -fPIC
        cd ../..
        ```
        *Please add more detailed C++ compilation instructions here.*

    * **Python Dependencies**:
        The majority of the framework is implemented in PyTorch. Experiments were conducted on a single NVIDIA RTX 4090 GPU.

        ```bash
        pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
        pip install -r requirements.txt
        ```
        Your `requirements.txt` file might include:
        ```text
        numpy
        matplotlib
        scipy
        
        ```

## Code Modules

This project contains the following core modules:

### 1. **OT (C++)**
* **Path**: `c_modules/OT/`
* **Description**: This is the core Optimal Transport solver. It takes a standard triangular mesh and a target measure defined by the image intensities to compute the initial OT map, outputting the deformed vertex positions (`V_OT`). This is the starting point of the entire pipeline, providing the basis for the subsequent geometric representation.

### 2. **OT Optimize (Python)**
* **Path**: `python_modules/ot_optimize.py`
* **Description**: Due to numerical errors from discrete computation, the OT mesh generated directly from the C++ module may have minor artifacts that can affect reconstruction quality. This module refines and optimizes the OT mesh by minimizing an energy functional `E_g(V)`. The energy functional balances image reconstruction fidelity with deformation field regularization to ensure geometric stability and accuracy.

### 3. **QC (Python)**
* **Path**: `python_modules/qc_mapping.py`
* **Description**: This module handles two-way transformations:
    * **Encoding**: Computes the deformation from the standard mesh to the optimized OT mesh (`V*_OT`) and represents it as the Beltrami Coefficient `μ`. This `μ` is our final G-IR.
    * **Decoding**: Takes a Beltrami Coefficient `μ` as input and uses a linear Beltrami solver to compute the inverse quasiconformal map, thereby reconstructing a mesh (`V_QC`).

### 4. **QC Optimize (Python)**
* **Path**: `python_modules/qc_optimize.py`
* **Description**: Similar to OT optimization, the reconstructed mesh `V_QC` can also accumulate errors from numerical computations. This module optimizes the reconstructed QC mesh by minimizing another energy functional `E_μ(V)`, which forces its resulting Beltrami coefficient to match the target `μ` as closely as possible, thus maximizing reconstruction fidelity.

### 5. **Pixel-IR AE (Python)**
* **Path**: `models/pixel_ae.py`
* **Description**: This is a standard U-Net-based Autoencoder that serves as the baseline model for our experiments. It is trained and evaluated directly on raw image pixels to be compared against our G-IR AE.

### 6. **G-IR AE (Python)**
* **Path**: `models/gir_ae.py`
* **Description**: This model shares the same U-Net architecture as the Pixel-IR AE but operates on our G-IR (`μ`-Image) instead of raw pixels[cite: 303]. It is used to validate the effectiveness of G-IR as a latent space representation, especially for interpolation tasks where preserving structural continuity is critical.
