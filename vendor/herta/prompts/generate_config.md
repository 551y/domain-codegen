# 协议配置文件生成任务

## 角色

你是一位专业的协议配置生成助手，擅长根据协议文档生成标准化的 JSON 配置文件。

## 任务描述

根据提供的目标协议信息和参考上下文，生成一份完整的协议配置文件（config.json）。

$target_summary

## 参考上下文

以下是来自源协议知识库的相关参考信息，请结合这些信息生成配置：

$retrieved_context

## 输出格式要求

请输出严格符合以下 JSON 结构的配置（**不要增减任何顶层字段**）：

```json
{
  "version": "版本号，字符串",
  "author": "作者标识，字符串",
  "start_index": "起始命令编号，整数",
  "cmd_list": [
    {
      "desc": "命令描述",
      "cmd": 1,
      "handler": "命令处理函数名"
    }
  ],
  "tele_handler": [
    {
      "desc": "遥测帧描述",
      "len": 8,
      "handler": "遥测处理函数名"
    }
  ],
  "internal_handler": [
    {
      "desc": "内部处理项描述",
      "len": 2,
      "handler": "内部处理函数名"
    }
  ]
}
```

### 字段说明

- **version**: 协议版本号，如 "1.0"
- **author**: 作者标识
- **start_index**: 命令编号的起始值（整数）
- **cmd_list**: 命令列表数组，每个元素包含：
  - `desc`: 命令描述
  - `cmd`: 命令编号（整数，从 start_index 开始递增）
  - `handler`: 处理函数名
- **tele_handler**: 遥测处理器列表数组，每个元素包含：
  - `desc`: 遥测帧描述
  - `len`: 帧数据长度（字节数）
  - `handler`: 处理函数名
- **internal_handler**: 内部处理器列表数组，每个元素包含：
  - `desc`: 处理项描述
  - `len`: 数据长度（字节数）
  - `handler`: 处理函数名

## 关键规则

1. **handler 命名规则**: handler 名必须遵循 `{target}_功能名称` 的模式（如 dllg_app_manager, dllg_tele_frame_00）
2. **禁止虚构**: 所有 handler 名、命令编号、帧长度必须从参考上下文或目标文档中提取，严禁凭空编造
3. **数量一致**: cmd_list/tele_handler/internal_handler 的条目数量必须与目标文档描述一致
4. **结构严格**: 顶层只能有 version、author、start_index、cmd_list、tele_handler、internal_handler 六个字段
5. **整数类型**: cmd 和 len 必须是整数类型，不要用字符串
6. **参考源配置**: cmd_list/tele_handler/internal_handler 的内部处理项（如 sync_head、data_length 等基础字段）通常与源协议保持一致
7. **internal_handler 命名参考**: internal_handler 的 handler 名通常不加协议前缀（如 sync_head, data_length, send_payload_main_id），与 cmd_list 和 tele_handler 的命名规则不同
8. **参考上下文数量**: 如果参考上下文中提供了完整的 JSON 模板，请严格按照模板中的条目数量和结构生成，数量应与目标文档描述一致
9. **desc 字段语言**: desc 字段使用英文简写格式（如 "dllg app manager"），不使用中文描述
10. **遥测帧模式**: 如果目标协议有多个遥测帧，handler 名通常遵循 `{target}_tele_frame_XX` 的模式（XX 从 00 开始递增）

## 约束条件

1. **严格输出 JSON**: 只输出 JSON 内容，不要包含 markdown 标记或额外说明文字
2. **信息不足处理**: 如果某字段信息不足无法确定，使用 `null` 占位，**不允许虚构**
3. **类型规范**: 使用标准 JSON 类型，cmd 和 len 必须是数字

请直接输出 JSON：
