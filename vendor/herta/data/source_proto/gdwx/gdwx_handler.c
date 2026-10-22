#include <stdio.h>
#include <stdint.h>
#include <sys/vfs.h>
#include <sys/statfs.h>
#include <arpa/inet.h>
#include "cmd_utils.h"
#include "get_info.h"
#include "handler.h"
#include "file_util.h"
#include "frame_process.h"
#include "log.h"
#include "fpgaAdcoFunc.h"

// 获取模组功耗
static int gdwx_get_power_mode() {
    FILE *fp;
    char line[256];
    int mode = -1;

    fp = popen("sudo nvpmodel -q 2>/dev/null", "r");
    if (!fp) {
        perror("无法执行 nvpmodel 命令");
        return -1;
    }

    // 先找到包含"Power Mode"的行
    while (fgets(line, sizeof(line), fp)) {
        if (strstr(line, "Power Mode")) {
            // 继续读取下一行获取数值
            if (fgets(line, sizeof(line), fp)) {
                // 去除末尾换行符
                line[strcspn(line, "\n")] = '\0';
                // 转换为整数
                char *endptr;
                long val = strtol(line, &endptr, 10);
                if (endptr != line && val >= 0 && val <= 9) {
                    mode = (int)val;
                }
            }
            break;
        }
    }

    pclose(fp);

    if (mode == -1) {
        fprintf(stderr, "未找到有效的功耗模式\n");
    }

    // log_debug("功耗模式数值: %d\n", mode);
    return mode;
}

// 获取cpu占有率
static float gdwx_get_cpu_usage() {
    FILE *fp;
    char output[128];
    float usage = -1.0;

    // 使用两次迭代并指定间隔时间，确保计算差值
    fp = popen("top -bn2 -d 0.5 | grep '^%Cpu' | tail -n1 | awk '{print 100 - $8}'", "r");
    if (!fp) {
        log_err("无法执行命令");
        return -1.0;
    }

    if (fgets(output, sizeof(output), fp)) {
        // 去除换行符
        output[strcspn(output, "\n")] = '\0';
        char *endptr;
        float val = strtof(output, &endptr);
        if (endptr != output && *endptr == '\0' && val >= 0.0 && val <= 100.0) {
            usage = val;
        } else {
            log_err("无效的数值: %s", output);
        }
    } else {
        log_err("无法读取命令输出");
    }

    int rc = pclose(fp);
    if (rc != 0) {
        log_err("命令执行失败，返回码：%d", rc);
        usage = -1.0;
    }

    // log_debug("mode: %f\n", usage);

    return usage;
}

// 获取gpu占有率
static float gdwx_get_gpu_usage() {
    FILE *fp;
    char path[] = "/sys/devices/gpu.0/load";  // 路径可能因设备不同而调整
    char output[256];
    float usage = -1.0;

    fp = fopen(path, "r");
    if (!fp) {
        log_err("无法打开GPU负载文件");
        return -1.0;
    }

    if (fgets(output, sizeof(output), fp)) {
        // 格式通常是整数（如 "123456"），需转换为百分比
        long load = atol(output);
        usage = (load / 1000.0);  // 根据设备文档调整转换公式
        usage = usage > 100 ? 100 : usage;  // 限制最大值为100%
    }

    fclose(fp);

    // log_debug("GPU使用率: %f\n", usage);
    return (usage >= 0 && usage <= 100) ? usage : -1.0;
}

// 获取内存占有率
static float gdwx_get_mem_usage() {
    FILE *fp;
    char output[256];
    float usage = -1.0;

    // 使用 free 命令获取内存信息
    fp = popen("free -m | awk '/Mem/ {print $3/$2 * 100}'", "r");
    if (!fp) {
        log_err("获取内存占用失败");
        return -1.0;
    }

    // 直接读取计算后的百分比
    if (fgets(output, sizeof(output), fp)) {
        usage = atof(output);
    }

    pclose(fp);
    // log_debug("mode: %f\n", usage);
    return (usage >= 0 && usage <= 100) ? usage : -1.0;
}

// 获取emmc存储占有率
static float gdwx_get_emmc_usage() {
    FILE *fp;
    char output[256];
    float usage = -1.0;

    // 使用 df 命令获取根分区（默认eMMC）的存储使用率
    fp = popen("df -P / | awk 'NR==2 {gsub(/%/, \"\"); print $5}'", "r");
    if (!fp) {
        log_err("获取eMMC存储占用失败");
        return -1.0;
    }

    // 读取并解析百分比
    if (fgets(output, sizeof(output), fp)) {
        usage = atof(output);
    } else {
        fprintf(stderr, "无法解析eMMC存储占用或设备未挂载\n");
    }

    pclose(fp);
    // log_debug("mode: %f\n", usage);
    return (usage >= 0 && usage <= 100) ? usage : -1.0;
}

// 获取外部存储占有率
static float gdwx_get_storage_usage() {
    FILE *fp;
    char output[256];
    float usage = -1.0;

    // 使用 df 命令获取指定挂载点的存储使用率
    fp = popen("df -P /media/data | awk 'NR==2 {gsub(/%/, \"\"); print $5}'", "r");
    if (!fp) {
        log_err("获取存储占用失败");
        return -1.0;
    }

    // 读取并解析百分比
    if (fgets(output, sizeof(output), fp)) {
        usage = atof(output);
    } else {
        fprintf(stderr, "无法解析存储占用或路径未挂载\n");
    }

    pclose(fp);
    // log_debug("mode: %f\n", usage);
    return (usage >= 0 && usage <= 100) ? usage : -1.0;
}

// gdwx 工程参数遥测回复
int32_t gdwx_tele_frame_00(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    // engineer_frame_00
    send_data[0] = 0x22;
    send_data[1] = 0xA3;
    send_data[2] = 0x1F;
    send_data[3] = 0xD8;
    send_data[4] = 0xF0;
    send_data[5] = 0xFF;
    send_data[6] = 0xF0;
    send_data[7] = 0x65;

    (*send_len) = 8;
    return RET_OK;
}

int32_t gdwx_tele_frame_01(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    // 系统时间 UTC累计时间，无符号十六进制整型时间起点为xxxx年x月x日0时0分0秒
    time_t rawtime = time(NULL);  // 获取当前UTC时间戳
    uint32_t utc_timestamp = (uint32_t)rawtime; // 强转为32位（注意2038年溢出问题）
    utc_timestamp = utc_timestamp - 1735660800;
    //utc_timestamp = utc_timestamp - 1735660798;
    // 按大端序存储到4个字节（兼容网络协议）
    send_data[8] = (utc_timestamp >> 24) & 0xFF; // 最高位字节
    send_data[9] = (utc_timestamp >> 16) & 0xFF;
    send_data[10] = (utc_timestamp >> 8)  & 0xFF;
    send_data[11] = utc_timestamp & 0xFF;        // 最低位字节
    send_data[12] = rcs->cur_working_can + 1 ; // CAN总线工作通道
    send_data[13] = rcs->can_reset_count & 0xff; // CAN总线复位计数
    send_data[14] = rcs->indirect_instruction_error_count & 0xff; // CAN总线接收错误帧计数
    send_data[15] = (rcs->fast_polling_recv_count+1) & 0xff; // 轮询接收计数
    (*send_len) = 8;
    return RET_OK;
}

int32_t gdwx_tele_frame_02(char *send_data, int32_t *send_len)
{
    // 18 19 22 23 有错误
    remote_ctl_status_t *rcs = get_rcs();
    send_data[16] = rcs->indirect_instruction_recv_count & 0xff; // 指令接收计数
    send_data[17] = rcs->indirect_instruction_execute_count & 0xff; // 指令执行计数
    send_data[18] = rcs->pose_broadcast_recv_count & 0xff; // 时间广播接收计数，已删除
    send_data[19] = rcs->pose_broadcast_recv_count & 0xff; // 姿控广播接收计数
    // RS422接收帧计数 
    send_data[20] = (rcs->satellite_work_rs422_recv_count >> 8) & 0xff; 
    send_data[21] = rcs->satellite_work_rs422_recv_count & 0xff;
    // RS422发送帧计数，已删除
    send_data[22] = 0xAA; 
    send_data[23] = 0xAA;   
    (*send_len) = 8;
    return RET_OK;
}

int32_t gdwx_tele_frame_03(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[24] = rcs->recv_mcu; // 向MCU发送帧计数 待补充
    send_data[25] = rcs->send_mcu; // 从MCU接收帧计数 待补充
    send_data[26] = (char)gdwx_get_power_mode(); // 模组功耗模式
    send_data[27] = gdwx_get_cpu_usage() ; // 模组CPU占有率 待补充
    send_data[28] = gdwx_get_gpu_usage(); // 模组GPU占有率 待补充
    send_data[29] = gdwx_get_mem_usage(); // 模组内存占用率
    send_data[30] = gdwx_get_emmc_usage();  // 模组存储占用率
    send_data[31] = gdwx_get_storage_usage();  // 外部存储占有率 
    (*send_len) = 8;
    return RET_OK;
}

int32_t gdwx_tele_frame_04(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[32] = (rcs->gdwx_temperature_and_flowth[0] >> 8) & 0xff; // 模组温度
    send_data[33] = rcs->gdwx_temperature_and_flowth[0] & 0xff; 
    send_data[34] = (rcs->gdwx_temperature_and_flowth[1] >> 8) & 0xff; // 温度采集点1
    send_data[35] = rcs->gdwx_temperature_and_flowth[1] & 0xff; 
    send_data[36] = (rcs->gdwx_temperature_and_flowth[2] >> 8) & 0xff; // 温度采集点2
    send_data[37] = rcs->gdwx_temperature_and_flowth[2] & 0xff; 
    send_data[38] = (rcs->gdwx_temperature_and_flowth[3] >> 8) & 0xff; // 温度采集点3
    send_data[39] = rcs->gdwx_temperature_and_flowth[3] & 0xff; 
    (*send_len) = 8;
    return RET_OK;
}

int32_t gdwx_tele_frame_05(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[40] = (rcs->gdwx_temperature_and_flowth[4] >> 8) & 0xff; // 温度采集点4
    send_data[41] = rcs->gdwx_temperature_and_flowth[4] & 0xff;  
    send_data[42] = (rcs->gdwx_temperature_and_flowth[5] >> 8) & 0xff; // 温度采集点5
    send_data[43] = rcs->gdwx_temperature_and_flowth[5] & 0xff;  
    send_data[44] = (rcs->gdwx_temperature_and_flowth[6] >> 8) & 0xff; // 流速
    send_data[45] = rcs->gdwx_temperature_and_flowth[6] & 0xff; 
    send_data[46] = rcs->data_trans_status; // 数传状态
    send_data[47] = 0xAA; // 从通信机接收数据帧计数
    (*send_len) = 8;
    return RET_OK;
}

int32_t gdwx_tele_frame_06(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[48] = 0xAA; 
    send_data[49] = 0xAA;   
    send_data[50] = 0xAA;  // 向通信机发送数据帧计数
    send_data[51] = 0xAA;  
    send_data[52] = 0xAA;  
    send_data[53] = rcs->telemetry_data.upload_channel; // 上注通道
    send_data[54] = rcs->telemetry_data.upload_status; // 上注状态
    send_data[55] = (rcs->telemetry_data.upload_count >> 16) & 0xff; // 接收上注帧计数
    (*send_len) = 8;
    return RET_OK;
}

int32_t gdwx_tele_frame_07(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[56] = (rcs->telemetry_data.upload_count >> 8) & 0xff; 
    send_data[57] = rcs->telemetry_data.upload_count & 0xff; 
    send_data[58] = (rcs->gdwx_upload_lost_count >> 16) & 0xff; ; // 上注帧错误计数
    send_data[59] = (rcs->gdwx_upload_lost_count >> 8) & 0xff; ; 
    send_data[60] = (rcs->gdwx_upload_lost_count) & 0xff; ;  
    send_data[61] = 0xAA; // 上注帧错误序号
    send_data[62] = 0xAA;  
    send_data[63] = 0xAA;
    (*send_len) = 8;
    return RET_OK;
}

int32_t gdwx_tele_frame_08(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[64] = (get_file_count(REMOTE_CTL_LOG_DIR) >> 8) & 0xff;  // 日志文件计数
    send_data[65] = get_file_count(REMOTE_CTL_LOG_DIR) & 0xff; 
    send_data[66] = 0xAA;  // 对地图像文件计数
    send_data[67] = 0xAA; // 对天图像文件计数 
    send_data[68] = 0; // 平台软件版本号 X
    send_data[69] = 0; // 平台软件版本号 Y
    send_data[70] = 1; // APP组号
    send_data[71] = rcs->telemetry_data.app_id; // 第一个APP ID
    (*send_len) = 8;
    return RET_OK;
}

int32_t gdwx_tele_frame_09(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[72] = rcs->telemetry_data.app_version; // 第一个APP 版本号
    send_data[73] = rcs->telemetry_data.app_install_status; // 第一个APP安装状态
    send_data[74] = rcs->telemetry_data.app_run_status; // 第一个APP运行状态
    send_data[75] = rcs->telemetry_data.app_data_status; //第一个APP数据状态
    send_data[76] = rcs->telemetry_data.app_pkg_status; // 第一个APP安装包状态
    send_data[77] = rcs->telemetry_data.app_update_pkg_status; // 第一个APP更新包状态
    send_data[78] = rcs->gdwx_dds_app_01[2]; 
    send_data[79] = rcs->gdwx_dds_app_01[3];
    (*send_len) = 8;
    return RET_OK;
}

int32_t gdwx_tele_frame_10(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[80] = rcs->gdwx_dds_app_01[4]; 
    send_data[81] = rcs->gdwx_dds_app_01[5]; 
    send_data[82] = rcs->gdwx_dds_app_01[6]; 
    send_data[83] = rcs->gdwx_dds_app_01[7]; 
    send_data[84] = rcs->gdwx_dds_app_01[8]; 
    send_data[85] = rcs->gdwx_dds_app_01[9]; 
    send_data[86] = rcs->gdwx_dds_app_01[10]; 
    send_data[87] = rcs->gdwx_dds_app_01[11];
    (*send_len) = 8;
    return RET_OK;
}

int32_t gdwx_tele_frame_11(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[88] = rcs->gdwx_dds_app_01[12]; 
    send_data[89] = rcs->gdwx_dds_app_01[13]; 
    send_data[90] = rcs->gdwx_dds_app_01[14]; 
    send_data[91] = rcs->gdwx_dds_app_01[15]; 
    send_data[92] = rcs->gdwx_dds_app_01[16]; 
    send_data[93] = rcs->gdwx_dds_app_01[17]; 
    send_data[94] = rcs->gdwx_dds_app_01[18]; 
    send_data[95] = rcs->gdwx_dds_app_01[19];
    (*send_len) = 8;
    return RET_OK;
}

int32_t gdwx_tele_frame_12(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[96] = rcs->gdwx_dds_app_01[20];  
    send_data[97] = rcs->gdwx_dds_app_01[21];  
    send_data[98] = rcs->gdwx_dds_app_01[22];  
    send_data[99] = rcs->gdwx_dds_app_01[23];  
    send_data[100] = rcs->gdwx_dds_app_01[24]; 
    send_data[101] = rcs->gdwx_dds_app_01[25]; 
    send_data[102] = rcs->gdwx_dds_app_01[26]; 
    send_data[103] = rcs->gdwx_dds_app_01[27]; 
    (*send_len) = 8;
    return RET_OK;
}

int32_t gdwx_tele_frame_13(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[104] = rcs->gdwx_dds_app_01[28]; 
    send_data[105] = rcs->gdwx_dds_app_01[29];
    send_data[106] = rcs->gdwx_dds_app_01[30];
    send_data[107] = rcs->gdwx_dds_app_01[31];
    send_data[108] = rcs->gdwx_dds_app_01[32];
    send_data[109] = rcs->gdwx_dds_app_01[33];
    send_data[110] = rcs->gdwx_dds_app_01[34];
    send_data[111] = rcs->gdwx_dds_app_01[35];
    (*send_len) = 8;
    return RET_OK;
}

int32_t gdwx_tele_frame_14(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[112] = rcs->gdwx_dds_app_01[36]; 
    send_data[113] = rcs->gdwx_dds_app_01[37];
    send_data[114] = rcs->gdwx_dds_app_01[38];
    send_data[115] = rcs->gdwx_dds_app_01[39];
    send_data[116] = rcs->gdwx_dds_app_01[40];
    send_data[117] = rcs->gdwx_dds_app_01[41];
    send_data[118] = rcs->gdwx_dds_app_01[42];
    send_data[119] = rcs->gdwx_dds_app_01[43];
    (*send_len) = 8;
    return RET_OK;
}

int32_t gdwx_tele_frame_15(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[120] = rcs->gdwx_dds_app_01[44];
    send_data[121] = rcs->gdwx_dds_app_01[45];
    send_data[122] = rcs->gdwx_dds_app_01[46];
    send_data[123] = rcs->gdwx_dds_app_01[47];
    send_data[124] = rcs->gdwx_dds_app_01[48];
    send_data[125] = rcs->gdwx_dds_app_01[49];
    send_data[126] = rcs->gdwx_dds_app_01[50];
    send_data[127] = rcs->gdwx_dds_app_01[51];
    (*send_len) = 8;
    return RET_OK;
}

int32_t gdwx_tele_frame_16(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[128] = rcs->telemetry_data.app2_info[0]; // 第二个app ID
    send_data[129] = rcs->telemetry_data.app2_info[1];
    send_data[130] = rcs->telemetry_data.app2_info[2];
    send_data[131] = rcs->telemetry_data.app2_info[3];
    send_data[132] = rcs->telemetry_data.app2_info[4]; 
    send_data[133] = rcs->telemetry_data.app2_info[5];
    send_data[134] = rcs->telemetry_data.app2_info[6];
    send_data[135] = 0xAA;
    (*send_len) = 8;
    return RET_OK;
}

int32_t gdwx_tele_frame_17(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[136] = rcs->telemetry_data.app3_info[0]; // 第三个app ID
    send_data[137] = rcs->telemetry_data.app3_info[1];
    send_data[138] = rcs->telemetry_data.app3_info[2];
    send_data[139] = rcs->telemetry_data.app3_info[3];
    send_data[140] = rcs->telemetry_data.app3_info[4]; 
    send_data[141] = rcs->telemetry_data.app3_info[5];
    send_data[142] = rcs->telemetry_data.app3_info[6];
    send_data[143] = 0xAA;
    (*send_len) = 8;
    return RET_OK;
}

int32_t gdwx_tele_frame_18(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[144] = rcs->telemetry_data.app4_info[0]; // 第四个 APP ID
    send_data[145] = rcs->telemetry_data.app4_info[1];
    send_data[146] = rcs->telemetry_data.app4_info[2];
    send_data[147] = rcs->telemetry_data.app4_info[3];
    send_data[148] = rcs->telemetry_data.app4_info[4]; 
    send_data[149] = rcs->telemetry_data.app4_info[5];
    send_data[150] = rcs->telemetry_data.app4_info[6];
    send_data[151] = 0xAA;
    (*send_len) = 8;
    return RET_OK;
}

int32_t gdwx_tele_frame_19(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[152] = rcs->telemetry_data.app5_info[0]; // 第五个app ID
    send_data[153] = rcs->telemetry_data.app5_info[1];
    send_data[154] = rcs->telemetry_data.app5_info[2];
    send_data[155] = rcs->telemetry_data.app5_info[3]; 
    send_data[156] = rcs->telemetry_data.app5_info[4]; 
    send_data[157] = rcs->telemetry_data.app5_info[5];
    send_data[158] = rcs->telemetry_data.app5_info[6];
    send_data[159] = 0xAA;
    (*send_len) = 8;
    return RET_OK;
}

int32_t gdwx_tele_frame_20(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[160] = rcs->telemetry_data.app6_info[0];
    send_data[161] = rcs->telemetry_data.app6_info[1];
    send_data[162] = rcs->telemetry_data.app6_info[2]; 
    send_data[163] = rcs->telemetry_data.app6_info[3];
    send_data[164] = rcs->telemetry_data.app6_info[4];
    send_data[165] = rcs->telemetry_data.app6_info[5];
    send_data[166] = rcs->telemetry_data.app6_info[6];
    send_data[167] = 0xAA;
    (*send_len) = 8;
    return RET_OK;
}

