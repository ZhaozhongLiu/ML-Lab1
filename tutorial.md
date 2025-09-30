# 机器学习作业完成指南

本教程将详细介绍如何从零开始完成机器学习作业，包括环境配置、代码编写、数据可视化、LaTeX 文档编写等全过程。

## 1. 环境配置

### 1.1 Python 环境配置

1. 创建虚拟环境：
```bash
python -m venv .venv
```

2. 激活虚拟环境：
```bash
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

3. 安装必要的包：
```bash
pip install numpy matplotlib torch pandas scipy scikit-learn
```

## 2. 代码编写和运行

### 2.1 作业代码结构

建议将作业代码分为以下几个部分：

1. 工具函数文件（如 `hw1_utils.py`）：存放通用函数
2. 主要实现文件（如 `hw1.py`）：存放具体问题的实现
3. 测试文件：用于验证实现的正确性

### 2.2 代码编写建议

1. Problem 1-4 的实现：
   - 按照作业要求实现每个函数
   - 添加适当的注释说明实现思路
   - 使用 NumPy 进行矩阵运算
   - 使用 sklearn 进行数据预处理

2. Problem 5（文本生成）的实现：
   - 使用 PyTorch 构建文本生成模型
   - 设置合适的超参数（n=4, embedding_dim=10）
   - 实现温度参数控制采样多样性
   - 保存生成的样本用于分析

### 2.3 代码运行示例

```python
# 运行具体问题的代码
python hw1.py

# 运行文本生成
python run_text_generation.py
```

## 3. 数据可视化

### 3.1 使用 Matplotlib 绘图

1. 基本绘图模板：
```python
import matplotlib.pyplot as plt
import numpy as np

def plot_results(x, y, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
```

2. 散点图示例：
```python
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.colorbar()
plt.savefig('scatter.png')
plt.close()
```

3. 绘图注意事项：
   - 设置合适的图像大小
   - 添加清晰的标题和轴标签
   - 使用网格增加可读性
   - 选择合适的颜色方案
   - 保存为高质量图片（DPI >= 300）

## 4. LaTeX 文档编写

### 4.1 文档结构

1. 基本结构：
```latex
\\documentclass[11pt]{article}
\\usepackage{amsmath}
\\usepackage{amssymb}
\\usepackage{amsthm}
\\usepackage{graphicx}
\\usepackage{float}

\\begin{document}
% 内容
\\end{document}
```

2. 分节结构：
```latex
\\section*{Problem 1}
\\noindent\\textbf{(a)} 解答...
\\noindent\\textbf{(b)} 解答...
```

### 4.2 插入图片

1. 基本图片插入：
```latex
\\begin{figure}[H]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{图片路径}
    \\caption{图片描述}
    \\label{fig:标签}
\\end{figure}
```

2. 并排插入多张图片：
```latex
\\begin{figure}[H]
    \\centering
    \\begin{minipage}{0.48\\textwidth}
        \\includegraphics[width=\\textwidth]{图片1}
    \\end{minipage}
    \\begin{minipage}{0.48\\textwidth}
        \\includegraphics[width=\\textwidth]{图片2}
    \\end{minipage}
    \\caption{图片描述}
\\end{figure}
```

### 4.3 数学公式

1. 行内公式：`$公式$`
2. 独立公式：
```latex
\\[
公式
\\]
```
3. 带编号的公式：
```latex
\\begin{equation}
公式
\\end{equation}
```
4. 多行对齐公式：
```latex
\\begin{align*}
公式1 \\\\
公式2 \\\\
公式3
\\end{align*}
```

### 4.4 编写代码结果

1. 文本生成结果展示：
```latex
\\begin{itemize}
\\item ``生成的文本1...''
\\item ``生成的文本2...''
\\end{itemize}
```

2. 实验结果展示：
```latex
\\begin{tabular}{|c|c|c|}
\\hline
参数 & 结果 & 说明 \\\\
\\hline
值1 & 结果1 & 说明1 \\\\
\\hline
\\end{tabular}
```

## 5. 编译和生成 PDF

### 5.1 编译命令

1. 基本编译：
```bash
pdflatex hw1.tex
```

2. 带参考文献的编译：
```bash
pdflatex hw1.tex
bibtex hw1
pdflatex hw1.tex
pdflatex hw1.tex
```

3. 持续编译（遇到错误不停止）：
```bash
pdflatex -interaction=nonstopmode hw1.tex
```

### 5.2 常见问题解决

1. 图片路径问题：
   - 使用相对路径
   - 确保图片文件存在
   - 检查图片文件名大小写

2. 数学公式错误：
   - 检查数学模式是否正确配对
   - 检查特殊字符是否正确转义
   - 使用 `\\` 而不是 `\` 作为换行

3. 编译错误：
   - 检查 LaTeX 包是否安装
   - 检查语法错误
   - 查看 .log 文件定位问题

## 6. 版本控制和备份

1. Git 基本操作：
```bash
git init
git add .
git commit -m "Initial commit"
git push origin main
```

2. 定期备份：
   - 使用 Git 进行版本控制
   - 保存重要的中间结果
   - 备份原始数据和图片文件

## 7. 作业提交检查清单

1. 代码部分：
   - [ ] 所有函数都已实现并测试
   - [ ] 代码有适当的注释
   - [ ] 移除调试用的打印语句

2. 图表部分：
   - [ ] 所有需要的图表都已生成
   - [ ] 图表清晰可读
   - [ ] 图表已正确插入 LaTeX 文档

3. LaTeX 文档：
   - [ ] 所有问题都已回答
   - [ ] 数学公式格式正确
   - [ ] 参考文献已正确引用
   - [ ] PDF 文件已成功生成

4. 最终检查：
   - [ ] 文件命名符合要求
   - [ ] 所有文件都已包含
   - [ ] PDF 文件可以正常打开
   - [ ] 内容符合作业要求

## 8. 补充说明

1. 工作效率提示：
   - 编写代码时使用 VS Code 的自动补全功能
   - 使用 LaTeX Workshop 插件实时预览
   - 保持良好的文件组织结构
   - 定期保存和备份工作

2. 注意事项：
   - 仔细阅读作业要求
   - 注意代码的可重用性
   - 保持文档的整洁和专业性
   - 遵守学术诚信要求
