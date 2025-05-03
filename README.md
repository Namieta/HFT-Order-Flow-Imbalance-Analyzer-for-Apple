# HFT-OFIAnalyzer

A comprehensive framework for analyzing Order Flow Imbalance across multiple price levels in high-frequency trading using Principal Component Analysis. Implements price impact models that integrate entire limit order book information, showing significant improvement over single-level approaches.

## Overview

This project implements advanced Order Flow Imbalance (OFI) analysis techniques for high-frequency trading data. It extends beyond traditional single-level OFI approaches by incorporating information from multiple price levels in the limit order book.

The implementation is based on the methodology described in:

> Cont, R., Cucuringu, M., & Zhang, C. (2023). Cross-impact of order flow imbalance in equity markets. *Quantitative Finance*.

Key features:
- Basic single-level OFI calculation and price impact modeling
- Multi-level OFI analysis across configurable depth of the order book
- PCA-based integration of information from all order book levels
- Cross-asset OFI analysis for market interconnectedness
- Visualization tools for OFI correlation structure
- Parallel processing for efficient handling of large datasets

## Methodology

The analysis follows these main steps:

1. **Data Processing**: Loads and preprocesses high-frequency order book data
2. **Basic OFI Calculation**: Computes OFI at the best bid/ask level following standard approaches
3. **Multi-level OFI**: Extends analysis to deeper levels in the order book
4. **Integrated OFI**: Uses Principal Component Analysis (PCA) to combine information from all levels
5. **Price Impact Modeling**: Estimates models relating OFI to price returns
6. **Cross-Asset Analysis**: Examines relationships between assets (for multi-asset datasets)

The methodology implements equations (1), (4), (6), (7), and (8) from Cont et al. (2023), with particular focus on the integrated OFI approach using PCA.

## Results

Our analysis shows:
- First principal component explains 52-62% of variance in multi-level OFI
- Integrated OFI approach improves R-squared by 31x over best-level approach
- Strong correlation structure between mid to deep levels of the order book
- Best level (top of book) carries unique information with weak correlation to deeper levels
- 
## If you use this code in your research, please cite:
@article{cont2023cross,
  title={Cross-impact of order flow imbalance in equity markets},
  author={Cont, Rama and Cucuringu, Mihai and Zhang, Chao},
  journal={Quantitative Finance},
  year={2023}
}

@misc{HFT-OFIAnalyzer,
  author = {Your Name},
  title = {HFT-OFIAnalyzer: Multi-level Order Flow Imbalance Analysis},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Namieta/HFT-OFIAnalyzer}
}

## Installation

```bash
git clone https://github.com/yourusername/HFT-OFIAnalyzer.git
cd HFT-OFIAnalyzer
pip install -r requirements.txt

