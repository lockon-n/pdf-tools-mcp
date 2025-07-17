# PDF Tools MCP Server

一个基于 FastMCP 的 PDF 读取和操作工具服务器，支持从 PDF 文件的指定页面范围提取文本内容。

## 功能特性

- 📄 读取 PDF 文件指定页面范围的内容
- 🔢 支持起始和结束页面参数（包含范围）
- 🛡️ 自动处理无效页码（负数、超出范围等）
- 📊 获取 PDF 文件的基本信息
- 💾 支持从字节数据读取 PDF
- 🔗 合并多个 PDF 文件
- ✂️ 提取 PDF 的特定页面

## 安装

### 从 PyPI 安装

```bash
pip install pdf-tools-mcp
```

### 从源码安装

```bash
git clone https://github.com/yourusername/pdf-tools-mcp.git
cd pdf-tools-mcp
pip install -e .
```

## 使用方法

### 作为命令行工具

```bash
# 基本使用
pdf-tools-mcp

# 指定工作目录
pdf-tools-mcp --workspace_path /path/to/workspace
```

### 作为 Python 包

```python
from pdf_tools_mcp import read_pdf_pages, get_pdf_info, merge_pdfs, extract_pdf_pages

# 读取 PDF 页面
result = await read_pdf_pages("document.pdf", 1, 5)

# 获取 PDF 信息
info = await get_pdf_info("document.pdf")

# 合并 PDF 文件
result = await merge_pdfs(["file1.pdf", "file2.pdf"], "merged.pdf")

# 提取特定页面
result = await extract_pdf_pages("source.pdf", [1, 3, 5], "extracted.pdf")
```

## 主要工具函数

### 1. read_pdf_pages
读取 PDF 文件指定页面范围的内容

**参数:**
- `pdf_file_path` (str): PDF 文件路径
- `start_page` (int, 默认 1): 起始页码
- `end_page` (int, 默认 1): 结束页码

**示例:**
```python
# 读取第 1-5 页
result = await read_pdf_pages("document.pdf", 1, 5)

# 读取第 10 页
result = await read_pdf_pages("document.pdf", 10, 10)
```

### 2. get_pdf_info
获取 PDF 文件的基本信息

**参数:**
- `pdf_file_path` (str): PDF 文件路径

**返回信息:**
- 总页数
- 标题
- 作者
- 创建者
- 创建日期

### 3. merge_pdfs
合并多个 PDF 文件

**参数:**
- `pdf_paths` (List[str]): 要合并的 PDF 文件路径列表
- `output_path` (str): 合并后的输出文件路径

### 4. extract_pdf_pages
从 PDF 中提取特定页面

**参数:**
- `source_path` (str): 源 PDF 文件路径
- `page_numbers` (List[int]): 要提取的页码列表（从 1 开始）
- `output_path` (str): 输出文件路径

## 错误处理

工具自动处理以下情况：
- 负数页码：自动调整为第 1 页
- 超出 PDF 总页数的页码：自动调整为最后一页
- 起始页大于结束页：自动交换
- 文件未找到：返回相应错误信息
- 权限不足：返回相应错误信息

## 使用示例

```python
# 获取 PDF 信息
info = await get_pdf_info("sample.pdf")
print(info)

# 读取前 3 页
content = await read_pdf_pages("sample.pdf", 1, 3)
print(content)

# 读取最后一页（假设 PDF 有 10 页）
content = await read_pdf_pages("sample.pdf", 10, 10)
print(content)

# 合并多个 PDF
result = await merge_pdfs(["part1.pdf", "part2.pdf", "part3.pdf"], "complete.pdf")
print(result)

# 提取特定页面
result = await extract_pdf_pages("source.pdf", [1, 3, 5, 7], "selected.pdf")
print(result)
```

## 注意事项

- 页面范围使用包含区间，即起始页和结束页都包含在内
- 如果指定页面没有文本内容，将被跳过
- 返回结果会显示 PDF 总页数和实际提取的页面范围
- 支持各种语言的 PDF 文档
- 建议一次读取的页面数不超过 50 页，以避免性能问题

## 开发

### 构建

```bash
uv build
```

### 发布到 PyPI

```bash
uv publish
```

### 本地开发

```bash
# 安装开发依赖
pip install -e .

# 运行测试
python -m pytest

# 运行服务器
python -m pdf_tools_mcp.server
```

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 更新日志

### 0.1.0
- 初始版本
- 支持 PDF 文本提取
- 支持 PDF 信息获取
- 支持 PDF 合并
- 支持页面提取
