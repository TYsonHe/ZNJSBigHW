# 进度日志

## 2026-06-08
- 启动报告选题与规划工作。
- 用户指定：使用 brainstorming 与 plan-with-files 流程；所有文件必须保存在 `final_report` 中；不参考其它目录。
- 检查 `final_report`，当前为空。
- 创建 `task_plan.md`、`findings.md`、`progress.md` 三个规划文件。
- 初步形成三个候选方向：
  1. 路径规划/TSP：蚁群算法 vs 遗传算法。
  2. 连续函数优化：粒子群优化 vs 差分进化。
  3. 特征选择：二进制粒子群 vs 遗传算法。
- 初步推荐方向 A：路径规划/TSP 中的蚁群算法与遗传算法对比。
- 用户确认选择方案 A。
- 更新规划：阶段 1 完成，阶段 2 报告设计确认开始。
- 用户确认实验 demo 采用方案 3：轻量仿真 + 小型真实数据。
- 用户指出批判性结论不应只分析劣势，还要分析算法优势；后续报告结论需平衡呈现优势、适用边界与局限。
- 按 TDD 要求创建 `test_experiment.py`，先确认因 `experiment.py` 缺失而失败，再实现实验代码。
- `experiment.py` 已生成，标准库 unittest 共 5 项测试通过。
- 正式运行实验，生成 `results/summary.csv`、`results/convergence.svg`、`results/routes.svg`。
- 生成报告正文 `report.md`，包含摘要、引言、问题建模、算法机制、实验结果、讨论、个人贡献、结论、参考文献与 AI 使用反思。
- 用户要求先生成 docx 版本；因本机无 pandoc 与 python-docx，使用 Python 标准库生成基础 Word OOXML 文件。
- 创建并运行 `test_make_docx.py`，确认 docx 包含必要结构与正文内容。
- 已生成 `面向物流配送路径规划的蚁群算法与遗传算法比较研究.docx`。
- 用户安装 `python-docx` 后，重写 Word 生成流程：设置 A4 页面、页边距、中文字体、标题样式、页码、实验结果表格和三张内嵌图。
- `pandoc` 在当前终端仍不可用；本次正式 Word 版采用 `python-docx` 生成。
- 生成真实 PNG 图表：`results/summary_chart.png`、`results/convergence_chart.png`、`results/routes_chart.png`。
- 润色 `report.md` 中实验设计和结果解释段落，强化优势、适用边界与局限的平衡分析。
- 验证新版 docx：63 个段落、1 个表格、3 张内嵌图片，文件大小 129458 字节。
- 按用户要求扩充第 2 节“问题建模与评价框架”和第 3 节“蚁群算法与遗传算法的机制比较”。
- 删除正式报告正文中的文件路径、工具名和代码文件说明，改为正常学术报告表述。
- 对全文进行自然化润色，减少机械性表达，保留 AI 使用说明中的诚实声明但弱化生硬工具痕迹。
- 重新生成 Word 文档，验证结果：67 个段落、1 个表格、3 张内嵌图片，文件大小 130736 字节。
- 测试通过：实验测试 5 项通过，DOCX 测试 3 项通过。
