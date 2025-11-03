# TMS-CL

Hierarchical Joint Contrastive Learning with Knowledge
Distillation for self-supervised 3D Skeleton-based Action
Recognition

This repository provides preprocessing, configuration, and evaluation scripts used in our paper  
*"Hierarchical Joint Contrastive Learning with Knowledge
Distillation for self-supervised 3D Skeleton-based Action
Recognition".  
The full model training code will be released upon paper acceptance.  



##  Overview

Hierarchical Joint Contrastive Learning is a **three-branch contrastive learning framework** designed for unsupervised 3D skeleton-based action recognition.  
It employs hierarchical knowledge distillation (Teacher → Middle → Student) and joint-subset selection to achieve strong **accuracy–efficiency trade-offs** across datasets such as NTU-RGB+D 60/120 and PKU-MMD.

Each branch operates at a different joint granularity:
Teacher: 25 joints (full skeleton)
Middle: 12 joints
Student: 6 joints


