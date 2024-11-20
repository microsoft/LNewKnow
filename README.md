# Learning New Knowledge

Welcome to the official repository for our work  **"Uncovering the Inequality in LLM's New Knowledge Learning Across Different Languages"** . This study investigates how LLM learns and transfers new knowledge in multilingual settings, highlighting disparities that may arise during this process.

## 🌟 Overview

As LLM's influence grows globally, it is crucial to examine how it interacts with diverse languages. This study addresses key questions about fairness and efficiency in multilingual AI systems:

### Research Questions:

1️⃣ **Equally Efficiently?**
Can LLM learn new knowledge with equal efficiency across different languages?

2️⃣ **Equally Transferable?**
Is the newly learned knowledge equally transferable between languages?

3️⃣ **Equally Prioritized?**
When conflicting knowledge arises across languages, does LLM prioritize one language over another?

4️⃣ **Equally Resistant?**
Does LLM demonstrate equal resistance to errors while learning new knowledge in different languages?

Through these questions, we aim to reveal hidden disparities in multilingual AI systems and suggest improvements for fostering fairness in AI development.

## 📦 **Setup**

To get started, clone the repository and install the necessary dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 **How to Use**

Below is a detailed explanation of how to run the experiments for each research question, including relevant code snippets and expected outcomes. Each script contains comments to guide you through the process.

---

### **To Answer Q1 & Q2: Learning and Transferability**

#### **In-context Learning**

Evaluate how LLM learns in one language and transfers knowledge to another.

```
cd src/1.ic_learning # Navigate to the directory for in-context learning
python completion_evaluation.py # Run the script to evaluate knowledge transfer
python parse_results.py # Parse the raw results
python visualization_matrix.py # Visualize the results
python visualization_inequality.py # Plot inequalty trends
```

#### **Finetuning**

Evaluate performance when LLM is finetuned in one language and tested in another.

```
cd ../src/2.ft_learning # Navigate to the directory for finetuning
python evaluation_completion_main.py # Finetune and evaluate
python parse_results.py # Parse the raw results
python visualization_matrix.py # Visualize the results
python visualization_inequality.py # Plot inequalty trends
```

---

### **To Answer Q3: Handling Conflicting Knowledge**

#### **In-context Learning**

Evaluate LLM's behavior when presented with conflicting knowledge across languages.

```
cd ../src/3.ic_conflict # Navigate to the directory for in-context conflict evaluation
python completion_evaluation.py # Run the conflict evaluation
python test_stats.py # Parse the raw results
python aggregation.py # Aggregate the outcomes
python visualization_aggregation.py # Visualize the outcomes
```

#### **Finetuning**

```
cd ../src/4.ft_conflict # Navigate to the directory for finetuning conflict evaluation
python completion_evaluation.py # Run the conflict evaluation
python test_stats.py # Parse the raw results
python aggregation.py # Aggregate the outcomes
python visualization_aggregation.py # Visualize the outcomes
```

---

### **To Answer Q4: Error Resistance**

#### **In-context Learning**

Evaluate LLM's resistance to errors during knowledge acquisition.

```
cd ../src/5.ic_resist # Navigate to the resistance evaluation directory
python completion_evaluation.py # Run the resistance test
python test_stats.py # Parse the raw results
python plot_resist.py # Visualize the results
```

#### **Finetuning**

```
cd ../src/6.ft_resist # Navigate to the resistance evaluation directory
python resist.py # Run the resistance test
python resist_stats.py # Parse the raw results
python fig_plot.py # Visualize the results
```

---

## 🔍 Notes

* Each script contains detailed comments to explain its functionality.
* Results are saved in the `results/` directory by default.
* Visualization outputs include heatmaps, line plots, and inequality metrics.


## 📜 **License**

This project is licensed under the license found in the [LICENSE](./LICENSE.txt) file in the root directory of this source tree.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

## ™️ **Trademarks**

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft’s Trademark &amp; Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.

## 🔗 **Contact**

For inquiries or contributions, feel free to reach out or open an issue in this repository. Let’s work together to bridge the gaps in multilingual AI!
