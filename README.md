# Schrodinger-NumericalSolvers

Numerical solutions of the **1D Schrödinger equation** in a finite quantum well using **Finite Difference (FD)** and **Numerov** methods.

---

## 📌 Project Overview

This project explores the **numerical solution of the time-independent Schrödinger equation** for a **1D finite quantum well**.  
Two classical numerical methods are implemented and compared:

- **Finite Difference Method (FD)**  
- **Numerov Method**  

The program computes:  
- **Bound states (eigenenergies)** inside the potential well  
- **Even and odd eigenfunctions**  
- **Comparison with theoretical values**  

It also provides **visualizations** of the normalized wavefunctions obtained by both methods, making it a practical tool for studying numerical quantum mechanics.

---

## 🔬 Key Features

- Clean and modular Python implementation  
- Bisection algorithm for eigenvalue calculation  
- Normalized eigenfunction plotting (up to 6 bound states)  
- Comparison of FD vs Numerov accuracy and efficiency  

---

## 📊 Results

- Eigenenergies are computed and compared against theoretical values.  
- Six wavefunctions are plotted:
  - 3 obtained with the **Finite Difference method** (even and odd states)  
  - 3 obtained with the **Numerov method**  

This allows a direct visual comparison between methods.

---

## 🛠️ Requirements

- Python 3.8+  
- NumPy  
- Matplotlib  

You can install dependencies with:

```bash
pip install numpy matplotlib
