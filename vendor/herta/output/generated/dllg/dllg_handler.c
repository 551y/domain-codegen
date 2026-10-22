/*
 * Handler 实现文件
 * 协议: dllg (MCMI星座卫星AI处理机CAN总线通信协议)
 * 生成时间: 2023-10-27
 * 注意: 部分实现包含 TODO 标记，需要根据实际情况补充
 */

/* ==================== 头文件包含 ==================== */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>

/* TODO: 根据实际项目结构调整头文件路径 */
#include "protocol_common.h"
#include "dllg_config.h"

/* ==================== 宏定义 ==================== */
/* 系统标识定义 */
#define SYS_ID_BROADCAST    0x26
#define SYS_ID_AP_RX        0xC8
#define SYS_ID_AP_TX        0xC9
#define SYS_ID_AC_RX        0xCA
#define SYS_ID_AC_TX        0xCB

/* AP指令字定义 (ID20~ID13) */
#define AP_CMD_BUS_CTRL         0x01
#define AP_CMD_ENG_PARAM        0x02
#define AP_CMD_APP_MANAGER      0x03
#define AP_CMD_UPLOAD_MANAGER   0x04
#define AP_CMD_DATA_TRANS       0x06
#define AP_CMD_DATA_CLEAR       0x07
#define AP_CMD_SET_POWER_MODE   0x08
#define AP_CMD_SET_TASK         0x09
#define AP_CMD_ORBIT_1          0x0A
#define AP_CMD_ORBIT_2          0x0B
#define AP_CMD_ORBIT_3          0x0C
#define AP_CMD_ORBIT_4          0x0D
#define AP_CMD_ORBIT_5          0x0E
#define AP_CMD_ORBIT_6          0x0F
#define AP_CMD_ORBIT_7          0x10

/* AC指令字定义 (ID20~ID13) */
#define AC_CMD_BUS_CTRL         0x01
#define AC_CMD_ENG_PARAM        0x02
#define AC_CMD_BUS_RESET        0x03
#define AC_CMD_FUNC_MANAGER     0x04
#define AC_CMD_BACKUP_SYS       0x05

/* 应答帧类型码 */
#define RESP_CMD_TYPE           0xA0

/* 总线控制字 */
#define BUS_CTRL_AUTO_A         0x00
#define BUS_CTRL_AUTO_B         0xAA
#define BUS_CTRL_FORCE_A        0xA0
#define BUS_CTRL_FORCE_B        0x0A

/* 应答状态码 */
#define RESP_STATUS_OK          0x00
#define RESP_STATUS_RECV_ERR    0xA0
#define RESP_STATUS_OK_NO_EXEC  0x0A

/* 状态码定义 */
#define HANDLER_OK      0
#define HANDLER_ERR     -1
#define HANDLER_EPARAM  -2
#define HANDLER_ECMD    -3

/* 缓冲区大小 */
#define CAN_DATA_LEN    8
#define MAX_RESPONSE_LEN 2048

/* 时间基准: 2000年1月1日12时00分00秒的Unix时间戳 */
#define UTC_BASE_TIME   946728000LL

/* ==================== 数据结构定义 ==================== */

typedef struct {
    uint8_t id;
    uint8_t version;
    uint8_t install_status;
    uint8_t run_status;
    uint8_t data_status;
    uint8_t pkg_status;
    uint8_t update_pkg_status;
    uint8_t reserved[8]; /* APP处理结果等 */
} app_status_t;

typedef struct {
    uint32_t system_time;
    uint8_t can_channel;
    uint8_t can_reset_count;
    uint8_t can_error_count;
    uint8_t poll_recv_count;
    
    uint8_t cmd_recv_count;
    uint8_t cmd_exec_count;
    uint8_t time_broadcast_count;
    uint16_t ais_frame_count;
    uint16_t orbit_cmd_count;
    
    uint8_t send_to_ac_count;
    uint8_t recv_from_ac_count;
    uint16_t module_temp;
    uint8_t power_mode;
    uint8_t cpu_usage;
    uint8_t mem_usage;
    uint8_t storage_usage;
    
    uint8_t ext_storage_usage;
    uint8_t data_trans_status;
    uint24_t recv_comm_count;
    uint24_t send_comm_count;
    
    uint8_t upload_channel;
    uint8_t upload_status;
    uint24_t upload_count;
    uint16_t upload_error_count;
    
    uint8_t ais_file_count;
    uint8_t mul_l0_file_count;
    uint8_t mul_l2_file_count;
    uint8_t ir_file_count;
    uint8_t task_id;
    
    uint8_t sw_ver_x;
    uint8_t sw_ver_y;
    uint16_t log_file_count;
    uint8_t app_group_id;
    
    app_status_t apps[5];
    
    /* AIS 数据 */
    uint8_t ais_work_status;
    uint8_t ais_img_need;
    uint8_t ais_img_res;
    uint8_t ais_down_need;
    uint32_t ais_mmsi;
    int32_t ais_lon;
    int32_t ais_lat;
    uint16_t ais_speed;
    uint16_t ais_course;
    uint16_t ais_ship_type;
    uint8_t ais_nav_status;
    uint32_t ais_time_s;
    uint16_t ais_time_ms;
    
    /* 火点数据 */
    uint16_t fire_area;
    int32_t fire_lon;
    int32_t fire_lat;
    int32_t fire_alt;
    uint16_t fire_xmin;
    uint16_t fire_ymin;
    uint16_t fire_xmax;
    uint16_t fire_ymax;
    
    /* 任务规划 */
    uint32_t cam_on_s;
    uint16_t cam_on_ms;
    uint32_t cam_off_s;
    uint16_t cam_off_ms;
    
    /* 姿态 */
    double roll;
    double pitch;
    double yaw;
    
    /* 总线状态 */
    uint8_t bus_mode; /* 0:自主, 1:强制 */
    uint8_t active_bus; /* 0:A, 1:B */
    uint32_t last_recv_time;
    
    /* 错误统计 */
    uint8_t error_cmd_count;
    uint8_t last_error_cmd_id;
} ap_remote_status_t;

typedef struct {
    uint8_t can_channel;
    uint8_t can_reset_count;
    uint8_t can_error_count;
    uint8_t poll_recv_count;
    uint8_t cmd_recv_count;
    uint8_t cmd_exec_count;
    uint8_t send_to_ap_count;
    uint8_t recv_from_ap_count;
    
    uint8_t wdg_status;
    uint8_t work_status;
    uint8_t ap_reset_count;
    uint8_t sw_ver_x;
    uint8_t sw_ver_y;
    
    uint8_t bus_mode;
    uint8_t active_bus;
    uint32_t last_recv_time;
    
    uint8_t error_cmd_count;
    uint8_t last_error_cmd_id;
} ac_remote_status_t;

/* ==================== 静态变量 ==================== */
static ap_remote_status_t g_ap_status;
static ac_remote_status_t g_ac_status;
static uint8_t g_tx_buffer[MAX_RESPONSE_LEN];

/* ==================== 辅助函数声明 ==================== */
static uint8_t dllg_calc_checksum(const uint8_t *data, uint32_t len);
static int dllg_switch_bus(uint8_t target_bus, uint8_t mode, uint8_t is_ap);
static uint32_t dllg_get_system_time(void);
static int dllg_send_can_frame(uint32_t id, const uint8_t *data, uint8_t len);

/* ==================== 命令处理函数声明 ==================== */
static int dllg_ap_bus_control(const uint8_t *data);
static int dllg_ap_eng_param_collect(void);
static int dllg_ap_app_manager(const uint8_t *data);
static int dllg_ap_upload_manager(const uint8_t *data);
static int dllg_ap_data_trans_manager(const uint8_t *data);
static int dllg_ap_data_clear(const uint8_t *data);
static int dllg_ap_set_power_mode(const uint8_t *data);
static int dllg_ap_set_task(const uint8_t *data);
static int dllg_ap_orbit_param(uint8_t cmd_id, const uint8_t *data);

static int dllg_ac_bus_control(const uint8_t *data);
static int dllg_ac_eng_param_collect(void);
static int dllg_ac_bus_reset(const uint8_t *data);
static int dllg_ac_func_manager(const uint8_t *data);
static int dllg_ac_backup_sys_manager(const uint8_t *data);

/* ==================== 遥测组包函数声明 ==================== */
static int dllg_ap_tele_frame_01(uint8_t *send_data);
static int dllg_ap_tele_frame_02(uint8_t *send_data);
static int dllg_ap_tele_frame_03(uint8_t *send_data);
static int dllg_ap_tele_frame_04(uint8_t *send_data);
static int dllg_ap_tele_frame_05(uint8_t *send_data);
static int dllg_ap_tele_frame_06(uint8_t *send_data);
static int dllg_ap_tele_frame_07(uint8_t *send_data);
static int dllg_ap_tele_frame_08(uint8_t *send_data);
static int dllg_ap_tele_frame_09(uint8_t *send_data);
static int dllg_ap_tele_frame_10(uint8_t *send_data);
static int dllg_ap_tele_frame_11(uint8_t *send_data);
static int dllg_ap_tele_frame_12(uint8_t *send_data);
static int dllg_ap_tele_frame_13(uint8_t *send_data);
static int dllg_ap_tele_frame_14(uint8_t *send_data);
static int dllg_ap_tele_frame_15(uint8_t *send_data);
static int dllg_ap_tele_frame_16(uint8_t *send_data);
static int dllg_ap_tele_frame_17(uint8_t *send_data);
static int dllg_ap_tele_frame_18(uint8_t *send_data);
static int dllg_ap_tele_frame_19(uint8_t *send_data);
static int dllg_ap_tele_frame_20(uint8_t *send_data);
static int dllg_ap_tele_frame_21(uint8_t *send_data);
static int dllg_ap_tele_frame_22(uint8_t *send_data);
static int dllg_ap_tele_frame_23(uint8_t *send_data);
static int dllg_ap_tele_frame_24(uint8_t *send_data);
static int dllg_ap_tele_frame_25(uint8_t *send_data);
static int dllg_ap_tele_frame_26(uint8_t *send_data);
static int dllg_ap_tele_frame_27(uint8_t *send_data);
static int dllg_ap_tele_frame_28(uint8_t *send_data);

static int dllg_ac_tele_frame_01(uint8_t *send_data);
static int dllg_ac_tele_frame_02(uint8_t *send_data);
static int dllg_ac_tele_frame_03(uint8_t *send_data);

/* ==================== 辅助函数实现 ==================== */

/*
 * 计算8位累加和校验码
 */
static uint8_t dllg_calc_checksum(const uint8_t *data, uint32_t len)
{
    uint8_t sum = 0;
    for (uint32_t i = 0; i < len; i++) {
        sum += data[i];
    }
    return sum;
}

/*
 * 执行总线切换逻辑
 */
static int dllg_switch_bus(uint8_t target_bus, uint8_t mode, uint8_t is_ap)
{
    if (is_ap) {
        g_ap_status.bus_mode = mode;
        g_ap_status.active_bus = target_bus;
        g_ap_status.last_recv_time = dllg_get_system_time();
        g_ap_status.can_channel = target_bus;
    } else {
        g_ac_status.bus_mode = mode;
        g_ac_status.active_bus = target_bus;
        g_ac_status.last_recv_time = dllg_get_system_time();
        g_ac_status.can_channel = target_bus;
    }
    /* TODO: 调用底层硬件驱动切换CAN总线通道 */
    return HANDLER_OK;
}

/*
 * 获取当前UTC时间(秒)
 */
static uint32_t dllg_get_system_time(void)
{
    /* TODO: 替换为硬件RTC或系统真实时间获取 */
    time_t t = time(NULL);
    return (uint32_t)t - UTC_BASE_TIME;
}

/*
 * CAN帧发送接口
 */
static int dllg_send_can_frame(uint32_t id, const uint8_t *data, uint8_t len)
{
    /* TODO: 调用底层CAN驱动发送函数 */
    return HANDLER_OK;
}

/* ==================== AP命令处理函数实现 ==================== */

static int dllg_ap_bus_control(const uint8_t *data)
{
    uint8_t ctrl = data[2];
    uint8_t target_bus = 0;
    uint8_t mode = 0;
    
    switch (ctrl) {
        case BUS_CTRL_AUTO_A:
            target_bus = 0; mode = 0; break;
        case BUS_CTRL_AUTO_B:
            target_bus = 1; mode = 0; break;
        case BUS_CTRL_FORCE_A:
            target_bus = 0; mode = 1; break;
        case BUS_CTRL_FORCE_B:
            target_bus = 1; mode = 1; break;
        default:
            return HANDLER_EPARAM;
    }
    return dllg_switch_bus(target_bus, mode, 1);
}

static int dllg_ap_eng_param_collect(void)
{
    uint8_t buf[CAN_DATA_LEN];
    uint32_t id;
    
    /* 构造AP发送ID: ID28~21=C9H, ID20~13=01H~1CH, ID12~6同ID28~22, ID5~0=00H */
    for (int i = 1; i <= 28; i++) {
        id = (SYS_ID_AP_TX << 21) | (i << 13) | ((SYS_ID_AP_TX & 0x7F) << 6);
        
        switch (i) {
            case 1: dllg_ap_tele_frame_01(buf); break;
            case 2: dllg_ap_tele_frame_02(buf); break;
            case 3: dllg_ap_tele_frame_03(buf); break;
            case 4: dllg_ap_tele_frame_04(buf); break;
            case 5: dllg_ap_tele_frame_05(buf); break;
            case 6: dllg_ap_tele_frame_06(buf); break;
            case 7: dllg_ap_tele_frame_07(buf); break;
            case 8: dllg_ap_tele_frame_08(buf); break;
            case 9: dllg_ap_tele_frame_09(buf); break;
            case 10: dllg_ap_tele_frame_10(buf); break;
            case 11: dllg_ap_tele_frame_11(buf); break;
            case 12: dllg_ap_tele_frame_12(buf); break;
            case 13: dllg_ap_tele_frame_13(buf); break;
            case 14: dllg_ap_tele_frame_14(buf); break;
            case 15: dllg_ap_tele_frame_15(buf); break;
            case 16: dllg_ap_tele_frame_16(buf); break;
            case 17: dllg_ap_tele_frame_17(buf); break;
            case 18: dllg_ap_tele_frame_18(buf); break;
            case 19: dllg_ap_tele_frame_19(buf); break;
            case 20: dllg_ap_tele_frame_20(buf); break;
            case 21: dllg_ap_tele_frame_21(buf); break;
            case 22: dllg_ap_tele_frame_22(buf); break;
            case 23: dllg_ap_tele_frame_23(buf); break;
            case 24: dllg_ap_tele_frame_24(buf); break;
            case 25: dllg_ap_tele_frame_25(buf); break;
            case 26: dllg_ap_tele_frame_26(buf); break;
            case 27: dllg_ap_tele_frame_27(buf); break;
            case 28: dllg_ap_tele_frame_28(buf); break;
        }
        
        /* 计算校验和并填充最后一字节 */
        buf[7] = dllg_calc_checksum(buf, 7);
        dllg_send_can_frame(id, buf, CAN_DATA_LEN);
    }
    return HANDLER_OK;
}

static int dllg_ap_app_manager(const uint8_t *data)
{
    uint8_t app_id = data[3];
    if (app_id < 1 || app_id > 255) return HANDLER_EPARAM;
    
    /* TODO: 通过DDS或内部总线转发给对应APP */
    return HANDLER_OK;
}

static int dllg_ap_upload_manager(const uint8_t *data)
{
    uint8_t channel = data[3];
    if (channel > 0x01) return HANDLER_EPARAM;
    
    /* TODO: 执行上注管理逻辑 */
    return HANDLER_OK;
}

static int dllg_ap_data_trans_manager(const uint8_t *data)
{
    /* TODO: 执行数传管理逻辑 */
    return HANDLER_OK;
}

static int dllg_ap_data_clear(const uint8_t *data)
{
    /* TODO: 执行数据清除逻辑 */
    return HANDLER_OK;
}

static int dllg_ap_set_power_mode(const uint8_t *data)
{
    /* TODO: 设置模组功耗模式 */
    return HANDLER_OK;
}

static int dllg_ap_set_task(const uint8_t *data)
{
    uint8_t task_id = data[2];
    if (task_id != 0x00 && task_id != 0x01 && task_id != 0x03) return HANDLER_EPARAM;
    
    g_ap_status.task_id = task_id;
    return HANDLER_OK;
}

static int dllg_ap_orbit_param(uint8_t cmd_id, const uint8_t *data)
{
    /* TODO: 存储轨道参数 */
    return HANDLER_OK;
}

/* ==================== AC命令处理函数实现 ==================== */

static int dllg_ac_bus_control(const uint8_t *data)
{
    uint8_t ctrl = data[2];
    uint8_t target_bus = 0;
    uint8_t mode = 0;
    
    switch (ctrl) {
        case BUS_CTRL_AUTO_A:
            target_bus = 0; mode = 0; break;
        case BUS_CTRL_AUTO_B:
            target_bus = 1; mode = 0; break;
        case BUS_CTRL_FORCE_A:
            target_bus = 0; mode = 1; break;
        case BUS_CTRL_FORCE_B:
            target_bus = 1; mode = 1; break;
        default:
            return HANDLER_EPARAM;
    }
    return dllg_switch_bus(target_bus, mode, 0);
}

static int dllg_ac_eng_param_collect(void)
{
    uint8_t buf[CAN_DATA_LEN];
    uint32_t id;
    
    for (int i = 1; i <= 3; i++) {
        id = (SYS_ID_AC_TX << 21) | (i << 13) | ((SYS_ID_AC_TX & 0x7F) << 6);
        
        switch (i) {
            case 1: dllg_ac_tele_frame_01(buf); break;
            case 2: dllg_ac_tele_frame_02(buf); break;
            case 3: dllg_ac_tele_frame_03(buf); break;
        }
        
        buf[7] = dllg_calc_checksum(buf, 7);
        dllg_send_can_frame(id, buf, CAN_DATA_LEN);
    }
    return HANDLER_OK;
}

static int dllg_ac_bus_reset(const uint8_t *data)
{
    /* TODO: 执行总线复位逻辑 */
    return HANDLER_OK;
}

static int dllg_ac_func_manager(const uint8_t *data)
{
    /* TODO: 执行功能管理逻辑 */
    return HANDLER_OK;
}

static int dllg_ac_backup_sys_manager(const uint8_t *data)
{
    /* TODO: 执行备份系统管理逻辑 */
    return HANDLER_OK;
}

/* ==================== AP遥测组包函数实现 ==================== */

static int dllg_ap_tele_frame_01(uint8_t *send_data)
{
    uint32_t time = dllg_get_system_time();
    send_data[0] = (time >> 24) & 0xFF;
    send_data[1] = (time >> 16) & 0xFF;
    send_data[2] = (time >> 8) & 0xFF;
    send_data[3] = time & 0xFF;
    send_data[4] = g_ap_status.can_channel;
    send_data[5] = g_ap_status.can_reset_count;
    send_data[6] = g_ap_status.can_error_count;
    send_data[7] = g_ap_status.poll_recv_count;
    return HANDLER_OK;
}

static int dllg_ap_tele_frame_02(uint8_t *send_data)
{
    send_data[0] = g_ap_status.cmd_recv_count;
    send_data[1] = g_ap_status.cmd_exec_count;
    send_data[2] = g_ap_status.time_broadcast_count;
    send_data[3] = 0xAA; /* 预留 */
    send_data[4] = (g_ap_status.ais_frame_count >> 8) & 0xFF;
    send_data[5] = g_ap_status.ais_frame_count & 0xFF;
    send_data[6] = (g_ap_status.orbit_cmd_count >> 8) & 0xFF;
    send_data[7] = g_ap_status.orbit_cmd_count & 0xFF;
    return HANDLER_OK;
}

static int dllg_ap_tele_frame_03(uint8_t *send_data)
{
    send_data[0] = g_ap_status.send_to_ac_count;
    send_data[1] = g_ap_status.recv_from_ac_count;
    send_data[2] = (g_ap_status.module_temp >> 8) & 0xFF;
    send_data[3] = g_ap_status.module_temp & 0xFF;
    send_data[4] = g_ap_status.power_mode;
    send_data[5] = g_ap_status.cpu_usage;
    send_data[6] = g_ap_status.mem_usage;
    send_data[7] = g_ap_status.storage_usage;
    return HANDLER_OK;
}

static int dllg_ap_tele_frame_04(uint8_t *send_data)
{
    send_data[0] = g_ap_status.ext_storage_usage;
    send_data[1] = g_ap_status.data_trans_status;
    send_data[2] = (g_ap_status.recv_comm_count >> 16) & 0xFF;
    send_data[3] = (g_ap_status.recv_comm_count >> 8) & 0xFF;
    send_data[4] = g_ap_status.recv_comm_count & 0xFF;
    send_data[5] = 0xAA; /* 预留 */
    send_data[6] = (g_ap_status.send_comm_count >> 8) & 0xFF;
    send_data[7] = g_ap_status.send_comm_count & 0xFF;
    return HANDLER_OK;
}

static int dllg_ap_tele_frame_05(uint8_t *send_data)
{
    send_data[0] = g_ap_status.upload_channel;
    send_data[1] = g_ap_status.upload_status;
    send_data[2] = (g_ap_status.upload_count >> 16) & 0xFF;
    send_data[3] = (g_ap_status.upload_count >> 8) & 0xFF;
    send_data[4] = g_ap_status.upload_count & 0xFF;
    send_data[5] = (g_ap_status.upload_error_count >> 8) & 0xFF;
    send_data[6] = g_ap_status.upload_error_count & 0xFF;
    send_data[7] = 0xAA; /* 预留 */
    return HANDLER_OK;
}

static int dllg_ap_tele_frame_06(uint8_t *send_data)
{
    send_data[0] = 0xAA; send_data[1] = 0xAA; send_data[2] = 0xAA; /* 预留 */
    send_data[3] = g_ap_status.ais_file_count;
    send_data[4] = g_ap_status.mul_l0_file_count;
    send_data[5] = g_ap_status.mul_l2_file_count;
    send_data[6] = g_ap_status.ir_file_count;
    send_data[7] = g_ap_status.task_id;
    return HANDLER_OK;
}

static int dllg_ap_tele_frame_07(uint8_t *send_data)
{
    send_data[0] = g_ap_status.sw_ver_x;
    send_data[1] = g_ap_status.sw_ver_y;
    send_data[2] = (g_ap_status.log_file_count >> 8) & 0xFF;
    send_data[3] = g_ap_status.log_file_count & 0xFF;
    send_data[4] = 0xAA; send_data[5] = 0xAA; send_data[6] = 0xAA; /* 保留 */
    send_data[7] = g_ap_status.app_group_id;
    return HANDLER_OK;
}

static int dllg_ap_tele_frame_08(uint8_t *send_data)
{
    send_data[0] = g_ap_status.apps[0].id;
    send_data[1] = g_ap_status.apps[0].version;
    send_data[2] = g_ap_status.apps[0].install_status;
    send_data[3] = g_ap_status.apps[0].run_status;
    send_data[4] = g_ap_status.apps[0].data_status;
    send_data[5] = g_ap_status.apps[0].pkg_status;
    send_data[6] = g_ap_status.apps[0].update_pkg_status;
    send_data[7] = 0xAA;
    return HANDLER_OK;
}

static int dllg_ap_tele_frame_09(uint8_t *send_data)
{
    send_data[0] = 0xAA; send_data[1] = 0xAA; send_data[2] = 0xAA; /* 预留 */
    memcpy(&send_data[3], &g_ap_status.apps[0].reserved[0], 5);
    return HANDLER_OK;
}

static int dllg_ap_tele_frame_10(uint8_t *send_data)
{
    memcpy(&send_data[0], &g_ap_status.apps[0].reserved[5], 3);
    send_data[3] = 0xAA; send_data[4] = 0xAA; send_data[5] = 0xAA;
    send_data[6] = 0xAA; send_data[7] = 0xAA;
    return HANDLER_OK;
}

static int dllg_ap_tele_frame_11(uint8_t *send_data)
{
    send_data[0] = g_ap_status.apps[1].id;
    send_data[1] = g_ap_status.apps[1].version;
    send_data[2] = g_ap_status.apps[1].install_status;
    send_data[3] = g_ap_status.apps[1].run_status;
    send_data[4] = g_ap_status.apps[1].data_status;
    send_data[5] = g_ap_status.apps[1].pkg_status;
    send_data[6] = g_ap_status.apps[1].update_pkg_status;
    send_data[7] = 0xAA;
    return HANDLER_OK;
}

static int dllg_ap_tele_frame_12(uint8_t *send_data)
{
    send_data[0] = g_ap_status.ais_work_status;
    send_data[1] = g_ap_status.ais_img_need;
    send_data[2] = g_ap_status.ais_img_res;
    send_data[3] = g_ap_status.ais_down_need;
    send_data[4] = (g_ap_status.ais_mmsi >> 24) & 0xFF;
    send_data[5] = (g_ap_status.ais_mmsi >> 16) & 0xFF;
    send_data[6] = (g_ap_status.ais_mmsi >> 8) & 0xFF;
    send_data[7] = g_ap_status.ais_mmsi & 0xFF;
    return HANDLER_OK;
}

static int dllg_ap_tele_frame_13(uint8_t *send_data)
{
    send_data[0] = (g_ap_status.ais_lon >> 24) & 0xFF;
    send_data[1] = (g_ap_status.ais_lon >> 16) & 0xFF;
    send_data[2] = (g_ap_status.ais_lon >> 8) & 0xFF;
    send_data[3] = g_ap_status.ais_lon & 0xFF;
    send_data[4] = (g_ap_status.ais_lat >> 24) & 0xFF;
    send_data[5] = (g_ap_status.ais_lat >> 16) & 0xFF;
    send_data[6] = (g_ap_status.ais_lat >> 8) & 0xFF;
    send_data[7] = g_ap_status.ais_lat & 0xFF;
    return HANDLER_OK;
}

static int dllg_ap_tele_frame_14(uint8_t *send_data)
{
    send_data[0] = (g_ap_status.ais_speed >> 8) & 0xFF;
    send_data[1] = g_ap_status.ais_speed & 0xFF;
    send_data[2] = (g_ap_status.ais_course >> 8) & 0xFF;
    send_data[3] = g_ap_status.ais_course & 0xFF;
    send_data[4] = (g_ap_status.ais_ship_type >> 8) & 0xFF;
    send_data[5] = g_ap_status.ais_ship_type & 0xFF;
    send_data[6] = g_ap_status.ais_nav_status;
    send_data[7] = 0xAA;
    return HANDLER_OK;
}

static int dllg_ap_tele_frame_15(uint8_t *send_data)
{
    send_data[0] = (g_ap_status.ais_time_s >> 24) & 0xFF;
    send_data[1] = (g_ap_status.ais_time_s >> 16) & 0xFF;
    send_data[2] = (g_ap_status.ais_time_s >> 8) & 0xFF;
    send_data[3] = g_ap_status.ais_time_s & 0xFF;
    send_data[4] = (g_ap_status.ais_time_ms >> 8) & 0xFF;
    send_data[5] = g_ap_status.ais_time_ms & 0xFF;
    send_data[6] = 0xAA; send_data[7] = 0xAA;
    return HANDLER_OK;
}

static int dllg_ap_tele_frame_16(uint8_t *send_data)
{
    send_data[0] = g_ap_status.apps[2].id;
    send_data[1] = g_ap_status.apps[2].version;
    send_data[2] = g_ap_status.apps[2].install_status;
    send_data[3] = g_ap_status.apps[2].run_status;
    send_data[4] = g_ap_status.apps[2].data_status;
    send_data[5] = g_ap_status.apps[2].pkg_status;
    send_data[6] = g_ap_status.apps[2].update_pkg_status;
    send_data[7] = 0xAA;
    return HANDLER_OK;
}

static int dllg_ap_tele_frame_17(uint8_t *send_data)
{
    send_data[0] = g_ap_status.apps[2].reserved[0]; /* 云检测工作状态 */
    send_data[1] = g_ap_status.apps[2].reserved[1]; /* 是否读取遥感图像 */
    send_data[2] = g_ap_status.apps[2].reserved[2]; /* 云量占比 */
    send_data[3] = g_ap_status.apps[2].reserved[3]; /* 是否保留当前图像 */
    send_data[4] = 0xAA; send_data[5] = 0xAA; send_data[6] = 0xAA; send_data[7] = 0xAA;
    return HANDLER_OK;
}

static int dllg_ap_tele_frame_18(uint8_t *send_data)
{
    memset(send_data, 0xAA, 8);
    return HANDLER_OK;
}

static int dllg_ap_tele_frame_19(uint8_t *send_data)
{
    send_data[0] = g_ap_status.apps[3].id;
    send_data[1] = g_ap_status.apps[3].version;
    send_data[2] = g_ap_status.apps[3].install_status;
    send_data[3] = g_ap_status.apps[3].run_status;
    send_data[4] = g_ap_status.apps[3].data_status;
    send_data[5] = g_ap_status.apps[3].pkg_status;
    send_data[6] = g_ap_status.apps[3].update_pkg_status;
    send_data[7] = 0xAA;
    return HANDLER_OK;
}

static int dllg_ap_tele_frame_20(uint8_t *send_data)
{
    send_data[0] = g_ap_status.apps[3].reserved[0]; /* 火点检测工作状态 */
    send_data[1] = g_ap_status.apps[3].reserved[1]; /* 是否读取遥感图像 */
    send_data[2] = (g_ap_status.fire_area >> 8) & 0xFF;
    send_data[3] = g_ap_status.fire_area & 0xFF;
    send_data[4] = (g_ap_status.fire_lon >> 24) & 0xFF;
    send_data[5] = (g_ap_status.fire_lon >> 16) & 0xFF;
    send_data[6] = (g_ap_status.fire_lon >> 8) & 0xFF;
    send_data[7] = g_ap_status.fire_lon & 0xFF;
    return HANDLER_OK;
}

static int dllg_ap_tele_frame_21(uint8_t *send_data)
{
    send_data[0] = (g_ap_status.fire_lat >> 24) & 0xFF;
    send_data[1] = (g_ap_status.fire_lat >> 16) & 0xFF;
    send_data[2] = (g_ap_status.fire_lat >> 8) & 0xFF;
    send_data[3] = g_ap_status.fire_lat & 0xFF;
    send_data[4] = (g_ap_status.fire_alt >> 24) & 0xFF;
    send_data[5] = (g_ap_status.fire_alt >> 16) & 0xFF;
    send_data[6] = (g_ap_status.fire_alt >> 8) & 0xFF;
    send_data[7] = g_ap_status.fire_alt & 0xFF;
    return HANDLER_OK;
}

static int dllg_ap_tele_frame_22(uint8_t *send_data)
{
    send_data[0] = (g_ap_status.fire_xmin >> 8) & 0xFF;
    send_data[1] = g_ap_status.fire_xmin & 0xFF;
    send_data[2] = (g_ap_status.fire_ymin >> 8) & 0xFF;
    send_data[3] = g_ap_status.fire_ymin & 0xFF;
    send_data[4] = (g_ap_status.fire_xmax >> 8) & 0xFF;
    send_data[5] = g_ap_status.fire_xmax & 0xFF;
    send_data[6] = (g_ap_status.fire_ymax >> 8) & 0xFF;
    send_data[7] = g_ap_status.fire_ymax & 0xFF;
    return HANDLER_OK;
}

static int dllg_ap_tele_frame_23(uint8_t *send_data)
{
    send_data[0] = g_ap_status.apps[4].id;
    send_data[1] = g_ap_status.apps[4].version;
    send_data[2] = g_ap_status.apps[4].install_status;
    send_data[3] = g_ap_status.apps[4].run_status;
    send_data[4] = g_ap_status.apps[4].data_status;
    send_data[5] = g_ap_status.apps[4].pkg_status;
    send_data[6] = g_ap_status.apps[4].update_pkg_status;
    send_data[7] = 0xAA;
    return HANDLER_OK;
}

static int dllg_ap_tele_frame_24(uint8_t *send_data)
{
    send_data[0] = g_ap_status.apps[4].reserved[0]; /* 任务规划工作状态 */
    send_data[1] = (g_ap_status.cam_on_s >> 24) & 0xFF;
    send_data[2] = (g_ap_status.cam_on_s >> 16) & 0xFF;
    send_data[3] = (g_ap_status.cam_on_s >> 8) & 0xFF;
    send_data[4] = g_ap_status.cam_on_s & 0xFF;
    send_data[5] = (g_ap_status.cam_on_ms >> 8) & 0xFF;
    send_data[6] = g_ap_status.cam_on_ms & 0xFF;
    send_data[7] = 0xAA;
    return HANDLER_OK;
}

static int dllg_ap_tele_frame_25(uint8_t *send_data)
{
    send_data[0] = (g_ap_status.cam_off_s >> 24) & 0xFF;
    send_data[1] = (g_ap_status.cam_off_s >> 16) & 0xFF;
    send_data[2] = (g_ap_status.cam_off_s >> 8) & 0xFF;
    send_data[3] = g_ap_status.cam_off_s & 0xFF;
    send_data[4] = (g_ap_status.cam_off_ms >> 8) & 0xFF;
    send_data[5] = g_ap_status.cam_off_ms & 0xFF;
    send_data[6] = 0xAA; send_data[7] = 0xAA;
    return HANDLER_OK;
}

static int dllg_ap_tele_frame_26(uint8_t *send_data)
{
    memcpy(send_data, &g_ap_status.roll, 8);
    return HANDLER_OK;
}

static int dllg_ap_tele_frame_27(uint8_t *send_data)
{
    memcpy(send_data, &g_ap_status.pitch, 8);
    return HANDLER_OK;
}

static int dllg_ap_tele_frame_28(uint8_t *send_data)
{
    memcpy(send_data, &g_ap_status.yaw, 8);
    return HANDLER_OK;
}

/* ==================== AC遥测组包函数实现 ==================== */

static int dllg_ac_tele_frame_01(uint8_t *send_data)
{
    send_data[0] = g_ac_status.can_channel;
    send_data[1] = g_ac_status.can_reset_count;
    send_data[2] = g_ac_status.can_error_count;
    send_data[3] = g_ac_status.poll_recv_count;
    send_data[4] = g_ac_status.cmd_recv_count;
    send_data[5] = g_ac_status.cmd_exec_count;
    send_data[6] = g_ac_status.send_to_ap_count;
    send_data[7] = g_ac_status.recv_from_ap_count;
    return HANDLER_OK;
}

static int dllg_ac_tele_frame_02(uint8_t *send_data)
{
    send_data[0] = g_ac_status.wdg_status;
    send_data[1] = g_ac_status.work_status;
    send_data[2] = g_ac_status.ap_reset_count;
    send_data[3] = g_ac_status.sw_ver_x;
    send_data[4] = g_ac_status.sw_ver_y;
    send_data[5] = 0xAA; send_data[6] = 0xAA; send_data[7] = 0xAA;
    return HANDLER_OK;
}

static int dllg_ac_tele_frame_03(uint8_t *send_data)
{
    memset(send_data, 0xAA, 8);
    return HANDLER_OK;
}

/* ==================== 应答帧发送函数 ==================== */

static int dllg_ap_send_response(uint8_t cmd_id, uint8_t status)
{
    uint8_t buf[CAN_DATA_LEN];
    uint32_t id = (SYS_ID_AP_TX << 21) | (RESP_CMD_TYPE << 13) | ((SYS_ID_AP_TX & 0x7F) << 6);
    
    uint8_t bus_status = 0;
    if (g_ap_status.bus_mode == 0) { /* 自主 */
        bus_status = (g_ap_status.active_bus == 0) ? 0xA0 : 0x0A;
    } else { /* 强制 */
        bus_status = (g_ap_status.active_bus == 0) ? 0xA5 : 0x5A;
    }
    
    buf[0] = bus_status;
    buf[1] = g_ap_status.cmd_recv_count;
    buf[2] = cmd_id;
    buf[3] = status;
    buf[4] = g_ap_status.error_cmd_count;
    buf[5] = (status != RESP_STATUS_OK) ? cmd_id : 0x00;
    buf[6] = 0xAA;
    buf[7] = 0xAA;
    
    return dllg_send_can_frame(id, buf, CAN_DATA_LEN);
}

static int dllg_ac_send_response(uint8_t cmd_id, uint8_t status)
{
    uint8_t buf[CAN_DATA_LEN];
    uint32_t id = (SYS_ID_AC_TX << 21) | (RESP_CMD_TYPE << 13) | ((SYS_ID_AC_TX & 0x7F) << 6);
    
    uint8_t bus_status = 0;
    if (g_ac_status.bus_mode == 0) {
        bus_status = (g_ac_status.active_bus == 0) ? 0xA0 : 0x0A;
    } else {
        bus_status = (g_ac_status.active_bus == 0) ? 0xA5 : 0x5A;
    }
    
    buf[0] = bus_status;
    buf[1] = g_ac_status.cmd_recv_count;
    buf[2] = cmd_id;
    buf[3] = status;
    buf[4] = g_ac_status.error_cmd_count;
    buf[5] = (status != RESP_STATUS_OK) ? cmd_id : 0x00;
    buf[6] = 0xAA;
    buf[7] = 0xAA;
    
    return dllg_send_can_frame(id, buf, CAN_DATA_LEN);
}

/* ==================== 主入口函数 ==================== */

/*
 * dllg_can_frame_dispatch
 * 职责: CAN帧主入口，解析仲裁域ID，分发到AP/AC/广播处理
 * 参数: id - 29位扩展帧ID, data - 数据域指针, len - 数据长度
 * 返回: 0成功，<0错误码
 */
int32_t dllg_can_frame_dispatch(uint32_t id, uint8_t* data, uint8_t len)
{
    int ret = HANDLER_OK;
    
    if (data == NULL || len != CAN_DATA_LEN) {
        return HANDLER_EPARAM;
    }
    
    /* 解析ID */
    uint8_t sys_id = (id >> 21) & 0xFF;
    uint8_t cmd_id = (id >> 13) & 0xFF;
    
    /* 更新接收时间，用于总线超时判断 */
    uint32_t now = dllg_get_system_time();
    g_ap_status.last_recv_time = now;
    g_ac_status.last_recv_time = now;
    
    switch (sys_id) {
        case SYS_ID_BROADCAST:
            if (cmd_id == 0x01) { /* UTC时间同步广播帧 */
                /* 更新系统时间 */
                uint32_t utc_time = ((uint32_t)data[0] << 24) | ((uint32_t)data[1] << 16) |
                                    ((uint32_t)data[2] << 8) | data[3];
                /* TODO: 同步本地RTC时间 */
                g_ap_status.time_broadcast_count++;
            }
            break;
            
        case SYS_ID_AP_RX:
            g_ap_status.cmd_recv_count++;
            switch (cmd_id) {
                case AP_CMD_BUS_CTRL:
                    ret = dllg_ap_bus_control(data);
                    break;
                case AP_CMD_ENG_PARAM:
                    ret = dllg_ap_eng_param_collect();
                    break;
                case AP_CMD_APP_MANAGER:
                    ret = dllg_ap_app_manager(data);
                    break;
                case AP_CMD_UPLOAD_MANAGER:
                    ret = dllg_ap_upload_manager(data);
                    break;
                case AP_CMD_DATA_TRANS:
                    ret = dllg_ap_data_trans_manager(data);
                    break;
                case AP_CMD_DATA_CLEAR:
                    ret = dllg_ap_data_clear(data);
                    break;
                case AP_CMD_SET_POWER_MODE:
                    ret = dllg_ap_set_power_mode(data);
                    break;
                case AP_CMD_SET_TASK:
                    ret = dllg_ap_set_task(data);
                    break;
                case AP_CMD_ORBIT_1:
                case AP_CMD_ORBIT_2:
                case AP_CMD_ORBIT_3:
                case AP_CMD_ORBIT_4:
                case AP_CMD_ORBIT_5:
                case AP_CMD_ORBIT_6:
                case AP_CMD_ORBIT_7:
                    ret = dllg_ap_orbit_param(cmd_id, data);
                    g_ap_status.orbit_cmd_count++;
                    /* 轨道参数指令不回应答 */
                    return HANDLER_OK;
                default:
                    ret = HANDLER_ECMD;
                    break;
            }
            
            /* 发送应答帧 */
            if (ret == HANDLER_OK) {
                g_ap_status.cmd_exec_count++;
                dllg_ap_send_response(cmd_id, RESP_STATUS_OK);
            } else {
                g_ap_status.error_cmd_count++;
                g_ap_status.last_error_cmd_id = cmd_id;
                dllg_ap_send_response(cmd_id, (ret == HANDLER_EPARAM) ? RESP_STATUS_RECV_ERR : RESP_STATUS_OK_NO_EXEC);
            }
            break;
            
        case SYS_ID_AC_RX:
            g_ac_status.cmd_recv_count++;
            switch (cmd_id) {
                case AC_CMD_BUS_CTRL:
                    ret = dllg_ac_bus_control(data);
                    break;
                case AC_CMD_ENG_PARAM:
                    ret = dllg_ac_eng_param_collect();
                    break;
                case AC_CMD_BUS_RESET:
                    ret = dllg_ac_bus_reset(data);
                    break;
                case AC_CMD_FUNC_MANAGER:
                    ret = dllg_ac_func_manager(data);
                    break;
                case AC_CMD_BACKUP_SYS:
                    ret = dllg_ac_backup_sys_manager(data);
                    break;
                default:
                    ret = HANDLER_ECMD;
                    break;
            }
            
            /* 发送应答帧 */
            if (ret == HANDLER_OK) {
                g_ac_status.cmd_exec_count++;
                dllg_ac_send_response(cmd_id, RESP_STATUS_OK);
            } else {
                g_ac_status.error_cmd_count++;
                g_ac_status.last_error_cmd_id = cmd_id;
                dllg_ac_send_response(cmd_id, (ret == HANDLER_EPARAM) ? RESP_STATUS_RECV_ERR : RESP_STATUS_OK_NO_EXEC);
            }
            break;
            
        default:
            /* 忽略非本机ID */
            break;
    }
    
    return ret;
}

/* ==================== 初始化函数 ==================== */

/*
 * Handler 模块初始化
 */
int32_t dllg_handler_init(void)
{
    memset(&g_ap_status, 0, sizeof(ap_remote_status_t));
    memset(&g_ac_status, 0, sizeof(ac_remote_status_t));
    
    /* 默认总线A，自主模式 */
    g_ap_status.active_bus = 0;
    g_ap_status.bus_mode = 0;
    g_ap_status.can_channel = 0;
    g_ap_status.app_group_id = 1;
    
    g_ac_status.active_bus = 0;
    g_ac_status.bus_mode = 0;
    g_ac_status.can_channel = 0;
    
    /* TODO: 初始化硬件接口、DDS通信等 */
    
    return HANDLER_OK;
}

/*
 * 总线超时检查函数(需在定时器或主循环中周期调用)
 */
void dllg_check_bus_timeout(void)
{
    uint32_t now = dllg_get_system_time();
    
    /* AP自主模式超时检查 */
    if (g_ap_status.bus_mode == 0) {
        if ((now - g_ap_status.last_recv_time) >= 6) {
            uint8_t next_bus = (g_ap_status.active_bus == 0) ? 1 : 0;
            dllg_switch_bus(next_bus, 0, 1);
        }
    }
    
    /* AC自主模式超时检查 */
    if (g_ac_status.bus_mode == 0) {
        if ((now - g_ac_status.last_recv_time) >= 6) {
            uint8_t next_bus = (g_ac_status.active_bus == 0) ? 1 : 0;
            dllg_switch_bus(next_bus, 0, 0);
        }
    }
}