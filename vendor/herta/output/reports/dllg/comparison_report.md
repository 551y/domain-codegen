# Herta 对比评测报告 - dllg

生成时间: 2026-10-19 18:04:29

## 总体评分

**整体相似度: 56.90%**

### 各维度得分

| 维度 | 得分 | 权重 |
|------|------|------|
| Config 拓扑结构 | 80.00% | 20% |
| Config 命令表 | 24.52% | 30% |
| Config 遥测表 | 100.00% | 30% |
| Config 内部处理 | 100.00% | 20% |
| **Config 总分** | **73.36%** | - |
| Handler 代码 | 48.03% | 65% |
| **整体加权** | **56.90%** | - |

---

## Config 详细分析

### 1. 拓扑结构相似度

**得分: 80.00%**

✓ 拓扑结构检查通过

### 2. 命令表相似度

**得分: 24.52%**

- 生成命令数: 15
- 真实命令数: 6

**命令对比详情:**

| 命令 | Handler 匹配 | Desc 相似度 | 得分 |
|------|--------------|-------------|------|
| 1 | ✗ | 50% | 25.0 |
| 2 | ✗ | 48% | 24.1 |
| 3 | ✗ | 80% | 40.0 |
| 6 | ✗ | 63% | 31.6 |
| 7 | ✗ | 53% | 26.5 |

**问题:**
- 命令数量不一致: 生成 15 vs 真实 6
- 缺失命令: 5

### 3. 遥测表相似度

**得分: 100.00%**

- 生成遥测项: 28
- 真实遥测项: 28

### 4. 内部处理表相似度

**得分: 100.00%**

- 生成内部处理项: 11
- 真实内部处理项: 11

---

## Handler 代码详细分析

**Handler 相似度: 48.03%**

### 函数匹配结果

| 真实函数 | 生成函数 | 同名 | AST 相似度 | TED | 语义 | 状态 |
|----------|----------|------|------------|-----|------|------|
| dllg_get_power_mode | dllg_ac_eng_param_collect | ✗ | - | - | - | ✗ 未匹配 |
| dllg_get_cpu_usage | dllg_ap_set_task | ✗ | - | - | - | ✗ 未匹配 |
| dllg_get_gpu_usage | dllg_ac_bus_control | ✗ | - | - | - | ✗ 未匹配 |
| dllg_get_mem_usage | dllg_calc_checksum | ✗ | - | - | - | ✗ 未匹配 |
| dllg_get_emmc_usage | dllg_switch_bus | ✗ | - | - | - | ✗ 未匹配 |
| dllg_get_storage_usage | dllg_ap_app_manager | ✗ | - | - | - | ✗ 未匹配 |
| check_file_exist | dllg_ap_upload_manager | ✗ | - | - | - | ✗ 未匹配 |
| dllg_tele_frame_00 | dllg_ap_tele_frame_01 | ✗ | 58.6% | 67% | 39% | ✓ 已匹配 |
| dllg_tele_frame_01 | dllg_ap_tele_frame_02 | ✗ | 59.3% | 73% | 28% | ✓ 已匹配 |
| dllg_tele_frame_02 | dllg_ap_tele_frame_04 | ✗ | 47.6% | 56% | 28% | ✓ 已匹配 |
| dllg_tele_frame_03 | dllg_ap_tele_frame_15 | ✗ | 52.1% | 62% | 28% | ✓ 已匹配 |
| dllg_tele_frame_04 | dllg_ap_tele_frame_05 | ✗ | 61.5% | 74% | 31% | ✓ 已匹配 |
| dllg_tele_frame_05 | dllg_ac_send_response | ✗ | - | - | - | ✗ 未匹配 |
| dllg_tele_frame_06 | dllg_ap_tele_frame_07 | ✗ | 51.5% | 61% | 29% | ✓ 已匹配 |
| dllg_tele_frame_07 | dllg_ac_tele_frame_01 | ✗ | 54.9% | 68% | 26% | ✓ 已匹配 |
| dllg_tele_frame_08 | dllg_ap_tele_frame_09 | ✗ | 35.3% | 39% | 27% | ✓ 已匹配 |
| dllg_tele_frame_09 | dllg_handler_init | ✗ | 33.4% | 37% | 25% | ✓ 已匹配 |
| dllg_tele_frame_10 | dllg_ap_tele_frame_17 | ✗ | 50.2% | 58% | 33% | ✓ 已匹配 |
| dllg_tele_frame_11 | dllg_ap_tele_frame_03 | ✗ | 51.9% | 60% | 33% | ✓ 已匹配 |
| dllg_tele_frame_12 | dllg_ap_tele_frame_14 | ✗ | 46.8% | 55% | 27% | ✓ 已匹配 |
| dllg_tele_frame_13 | dllg_ap_tele_frame_12 | ✗ | 46.9% | 55% | 28% | ✓ 已匹配 |
| dllg_tele_frame_14 | dllg_ap_tele_frame_13 | ✗ | 43.2% | 50% | 28% | ✓ 已匹配 |
| dllg_tele_frame_15 | dllg_ap_tele_frame_25 | ✗ | 46.3% | 52% | 32% | ✓ 已匹配 |
| dllg_tele_frame_16 | dllg_ac_tele_frame_02 | ✗ | 54.9% | 65% | 30% | ✓ 已匹配 |
| dllg_tele_frame_17 | dllg_ap_tele_frame_10 | ✗ | 48.6% | 58% | 27% | ✓ 已匹配 |
| dllg_tele_frame_18 | dllg_ap_tele_frame_24 | ✗ | 48.6% | 55% | 32% | ✓ 已匹配 |
| dllg_tele_frame_19 | dllg_ap_tele_frame_22 | ✗ | 48.2% | 54% | 34% | ✓ 已匹配 |
| dllg_tele_frame_20 | dllg_ap_tele_frame_23 | ✗ | 53.2% | 65% | 26% | ✓ 已匹配 |
| dllg_tele_frame_21 | dllg_ap_tele_frame_21 | ✗ | 43.2% | 50% | 28% | ✓ 已匹配 |
| dllg_tele_frame_22 | dllg_ap_tele_frame_20 | ✗ | 47.0% | 54% | 32% | ✓ 已匹配 |
| dllg_tele_frame_23 | dllg_ap_tele_frame_06 | ✗ | 48.2% | 57% | 27% | ✓ 已匹配 |
| dllg_tele_frame_24 | dllg_ap_tele_frame_11 | ✗ | 53.2% | 65% | 26% | ✓ 已匹配 |
| dllg_tele_frame_25 | dllg_ap_tele_frame_16 | ✗ | 53.2% | 65% | 26% | ✓ 已匹配 |
| dllg_tele_frame_26 | dllg_ap_tele_frame_19 | ✗ | 53.2% | 65% | 26% | ✓ 已匹配 |
| dllg_tele_frame_27 | dllg_ap_tele_frame_08 | ✗ | 53.2% | 65% | 26% | ✓ 已匹配 |

### 未匹配的生成函数

以下生成函数在真实代码中没有对应:

- `dllg_get_system_time`
- `dllg_send_can_frame`
- `dllg_ap_bus_control`
- `dllg_ap_eng_param_collect`
- `dllg_ap_data_trans_manager`
- `dllg_ap_data_clear`
- `dllg_ap_set_power_mode`
- `dllg_ap_orbit_param`
- `dllg_ac_bus_reset`
- `dllg_ac_func_manager`
- `dllg_ac_backup_sys_manager`
- `dllg_ap_tele_frame_18`
- `dllg_ap_tele_frame_26`
- `dllg_ap_tele_frame_27`
- `dllg_ap_tele_frame_28`
- `dllg_ac_tele_frame_03`
- `dllg_ap_send_response`
- `dllg_can_frame_dispatch`
- `dllg_check_bus_timeout`
