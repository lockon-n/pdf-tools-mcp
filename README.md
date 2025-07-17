# PDF Tools MCP Server

[English](#english) | [中文](#中文)

---

## 中文

一个基于 FastMCP 的 PDF 读取和操作工具服务器，支持从 PDF 文件的指定页面范围提取文本内容。

### 功能特性

- 📄 读取 PDF 文件指定页面范围的内容
- 🔢 支持起始和结束页面参数（包含范围）
- 🛡️ 自动处理无效页码（负数、超出范围等）
- 📊 获取 PDF 文件的基本信息
- 🔗 合并多个 PDF 文件
- ✂️ 提取 PDF 的特定页面

### 安装

#### 从 PyPI 安装

```bash
uv add pdf-tools-mcp
```

#### 从源码安装

```bash
git clone https://github.com/yourusername/pdf-tools-mcp.git
cd pdf-tools-mcp
uv sync
```

### 使用方法

#### 与 Claude Desktop 集成

添加到你的 `~/.config/claude/claude_desktop_config.json` (Linux/Windows) 或 `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

**开发/未发布版本配置**

```json
{
  "mcpServers": {
    "pdf-tools-mcp": {
      "command": "uv",
      "args": [
        "--directory",
        "<path/to/the/repo>/pdf-tools-mcp",
        "run",
        "pdf-tools-mcp",
        "--workspace_path",
        "</your/workspace/directory>"
      ]
    }
  }
}
```

**已发布版本配置**

```json
{
  "mcpServers": {
    "pdf-tools-mcp": {
      "command": "uvx",
      "args": [
        "pdf-tools-mcp",
        "--workspace_path",
        "</your/workspace/directory>"
      ]
    }
  }
}
```

**注意**: 出于安全考虑，此工具只能访问指定工作目录(`--workspace_path`)内的文件，无法访问工作目录之外的文件。

如果配置后无法正常工作或在UI中无法显示，请通过 `uv cache clean` 清除缓存。

#### 作为命令行工具

```bash
# 基本使用
pdf-tools-mcp

# 指定工作目录
pdf-tools-mcp --workspace_path /path/to/workspace
```

#### 作为 Python 包

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

### 主要工具函数

#### 1. read_pdf_pages
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

#### 2. get_pdf_info
获取 PDF 文件的基本信息

**参数:**
- `pdf_file_path` (str): PDF 文件路径

**返回信息:**
- 总页数
- 标题
- 作者
- 创建者
- 创建日期

#### 3. merge_pdfs
合并多个 PDF 文件

**参数:**
- `pdf_paths` (List[str]): 要合并的 PDF 文件路径列表
- `output_path` (str): 合并后的输出文件路径

#### 4. extract_pdf_pages
从 PDF 中提取特定页面

**参数:**
- `source_path` (str): 源 PDF 文件路径
- `page_numbers` (List[int]): 要提取的页码列表（从 1 开始）
- `output_path` (str): 输出文件路径

### 错误处理

工具自动处理以下情况：
- 负数页码：自动调整为第 1 页
- 超出 PDF 总页数的页码：自动调整为最后一页
- 起始页大于结束页：自动交换
- 文件未找到：返回相应错误信息
- 权限不足：返回相应错误信息

### 使用示例

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

### 注意事项

- 页面范围使用包含区间，即起始页和结束页都包含在内
- 如果指定页面没有文本内容，将被跳过
- 返回结果会显示 PDF 总页数和实际提取的页面范围
- 支持各种语言的 PDF 文档
- 建议一次读取的页面数不超过 50 页，以避免性能问题

### 开发

#### 构建

```bash
uv build
```

#### 发布到 PyPI

```bash
uv publish
```

#### 本地开发

```bash
# 安装开发依赖
uv sync

# 运行测试
uv run python -m pytest

# 运行服务器
uv run python -m pdf_tools_mcp.server
```

---

## English

A FastMCP-based PDF reading and manipulation tool server that supports extracting text content from specified page ranges of PDF files.

### Features

- 📄 Read content from specified page ranges of PDF files
- 🔢 Support for start and end page parameters (inclusive range)
- 🛡️ Automatic handling of invalid page numbers (negative numbers, out of range, etc.)
- 📊 Get basic information about PDF files
- 🔗 Merge multiple PDF files
- ✂️ Extract specific pages from PDFs

### Installation

#### Install from PyPI

```bash
uv add pdf-tools-mcp
```

#### Install from source

```bash
git clone https://github.com/yourusername/pdf-tools-mcp.git
cd pdf-tools-mcp
uv sync
```

### Usage

#### Usage with Claude Desktop

Add to your `~/.config/claude/claude_desktop_config.json` (Linux/Windows) or `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

**Development/Unpublished Servers Configuration**

```json
{
  "mcpServers": {
    "pdf-tools-mcp": {
      "command": "uv",
      "args": [
        "--directory",
        "<path/to/the/repo>/pdf-tools-mcp",
        "run",
        "pdf-tools-mcp",
        "--workspace_path",
        "</your/workspace/directory>"
      ]
    }
  }
}
```

**Published Servers Configuration**

```json
{
  "mcpServers": {
    "pdf-tools-mcp": {
      "command": "uvx",
      "args": [
        "pdf-tools-mcp",
        "--workspace_path",
        "</your/workspace/directory>"
      ]
    }
  }
}
```

**Note**: For security reasons, this tool can only access files within the specified workspace directory (`--workspace_path`) and cannot access files outside the workspace directory.

In case it's not working or showing in the UI, clear your cache via `uv cache clean`.

#### As a command line tool

```bash
# Basic usage
pdf-tools-mcp

# Specify workspace directory
pdf-tools-mcp --workspace_path /path/to/workspace
```

#### As a Python package

```python
from pdf_tools_mcp import read_pdf_pages, get_pdf_info, merge_pdfs, extract_pdf_pages

# Read PDF pages
result = await read_pdf_pages("document.pdf", 1, 5)

# Get PDF info
info = await get_pdf_info("document.pdf")

# Merge PDF files
result = await merge_pdfs(["file1.pdf", "file2.pdf"], "merged.pdf")

# Extract specific pages
result = await extract_pdf_pages("source.pdf", [1, 3, 5], "extracted.pdf")
```

### Main Tool Functions

#### 1. read_pdf_pages
Read content from specified page ranges of a PDF file

**Parameters:**
- `pdf_file_path` (str): PDF file path
- `start_page` (int, default 1): Starting page number
- `end_page` (int, default 1): Ending page number

**Example:**
```python
# Read pages 1-5
result = await read_pdf_pages("document.pdf", 1, 5)

# Read page 10
result = await read_pdf_pages("document.pdf", 10, 10)
```

#### 2. get_pdf_info
Get basic information about a PDF file

**Parameters:**
- `pdf_file_path` (str): PDF file path

**Returns:**
- Total page count
- Title
- Author
- Creator
- Creation date

#### 3. merge_pdfs
Merge multiple PDF files

**Parameters:**
- `pdf_paths` (List[str]): List of PDF file paths to merge
- `output_path` (str): Output file path for the merged PDF

#### 4. extract_pdf_pages
Extract specific pages from a PDF

**Parameters:**
- `source_path` (str): Source PDF file path
- `page_numbers` (List[int]): List of page numbers to extract (1-based)
- `output_path` (str): Output file path

### Error Handling

The tool automatically handles the following situations:
- Negative page numbers: automatically adjusted to page 1
- Page numbers exceeding total PDF pages: automatically adjusted to the last page
- Start page greater than end page: automatically swapped
- File not found: returns appropriate error message
- Insufficient permissions: returns appropriate error message

### Usage Examples

```python
# Get PDF info
info = await get_pdf_info("sample.pdf")
print(info)

# Read first 3 pages
content = await read_pdf_pages("sample.pdf", 1, 3)
print(content)

# Read last page (assuming PDF has 10 pages)
content = await read_pdf_pages("sample.pdf", 10, 10)
print(content)

# Merge multiple PDFs
result = await merge_pdfs(["part1.pdf", "part2.pdf", "part3.pdf"], "complete.pdf")
print(result)

# Extract specific pages
result = await extract_pdf_pages("source.pdf", [1, 3, 5, 7], "selected.pdf")
print(result)
```

### Notes

- Page ranges use inclusive intervals, meaning both start and end pages are included
- Pages without text content will be skipped
- Results show total PDF page count and actual extracted page range
- Supports PDF documents in various languages
- Recommended to read no more than 50 pages at a time to avoid performance issues

### Development

#### Build

```bash
uv build
```

#### Publish to PyPI

```bash
uv publish
```

#### Local Development

```bash
# Install development dependencies
uv sync

# Run tests
uv run python -m pytest

# Run server
uv run python -m pdf_tools_mcp.server
```

## License

MIT License

## Contributing

Issues and Pull Requests are welcome!

## Changelog

### 0.1.0
- Initial release
- Support for PDF text extraction
- Support for PDF info retrieval
- Support for PDF merging
- Support for page extraction
