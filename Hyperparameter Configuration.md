\##  Hyperparameter Configuration



All experiments were conducted under consistent settings to ensure fair comparison and reproducibility.  

The table below summarizes the full configuration used during contrastive pretraining on \*\*NTU-RGB+D 60 (Cross-Subject)\*\*.





| \*\*Parameter\*\* | \*\*Value\*\* |

|----------------|-----------|

| Dataset / Protocol | NTU60 X-Sub |

| Batch size | 64 |

| Epochs | 451 |

| Optimizer | SGD |

| Learning rate | 0.001 |

| Momentum | 0.9 |

| Weight decay | 1.0e-4 |

| LR schedule | \[100, 160] |

| Temperature | 0.07 |

| Queue size | 16,384 |

| MoCo momentum | 0.999 |

| Capacity | student \\| middle \\| teacher |

| \*\*Joint selector\*\* | \*\*SCL\*\* |

| Top-K joints | student: \[6]; middle: \[12]; teacher: \[25] |

| Clip length | 64 |

| Num joints | 25 |

| Num persons | 2 |

| Seed | \[42] |



Each experiment was repeated \*\*three times\*\* with independent random seed \*(42)\* to ensure statistical stability.  






