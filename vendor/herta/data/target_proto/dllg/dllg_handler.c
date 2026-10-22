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
static int dllg_get_power_mode() {
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
static float dllg_get_cpu_usage() {
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
static float dllg_get_gpu_usage() {
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
static float dllg_get_mem_usage() {
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
static float dllg_get_emmc_usage() {
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
static float dllg_get_storage_usage() {
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

int check_file_exist(const char *path) {
    if (path == NULL) {
        return 0;  // 安全处理空指针
    }
    // 使用F_OK模式检查文件是否存在
    if (access(path, F_OK) == 0) {
        return 1;  // 文件存在
    }
    return 0;      // 文件不存在或检查失败
}

// dllg 工程参数遥测回复
int32_t dllg_tele_frame_00(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    time_t rawtime = time(NULL);  // 获取当前UTC时间戳
    uint32_t utc_timestamp = (uint32_t)rawtime; // 强转为32位（注意2038年溢出问题）
    // 按大端序存储到4个字节（兼容网络协议）
    send_data[0] = (utc_timestamp >> 24) & 0xFF; // 最高位字节
    send_data[1] = (utc_timestamp >> 16) & 0xFF;
    send_data[2] = (utc_timestamp >> 8)  & 0xFF;
    send_data[3] = utc_timestamp & 0xFF;        // 最低位字节
    send_data[4] = rcs->cur_working_can ; // CAN总线工作通道
    send_data[5] = rcs->can_reset_count & 0xff; // CAN总线复位计数
    send_data[6] = rcs->indirect_instruction_error_count & 0xff; // CAN总线接收错误帧计数
    send_data[7] = (rcs->fast_polling_recv_count+1) & 0xff; // 轮询接收计数
    (*send_len) = 8;
    return RET_OK;
}

int32_t dllg_tele_frame_01(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[8] = rcs->indirect_instruction_recv_count & 0xff; // 指令接收计数
    send_data[9] = rcs->indirect_instruction_execute_count & 0xff; // 指令执行计数
    send_data[10] = rcs->time_broadcast_recv_count; // 时间广播接收计数，已删除
    send_data[11] = 0xAA;
    send_data[12] = (rcs->ais_frame_count >> 8) & 0xff;
    send_data[13] = (rcs->ais_frame_count) & 0xff;
    send_data[14] = (rcs->orbit_frame_count >> 8) & 0xff;
    send_data[15] = (rcs->orbit_frame_count) & 0xff;
    (*send_len) = 8;
    return RET_OK;
}

int32_t dllg_tele_frame_02(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[16] = rcs->send_to_mcu; // 向MCU发送帧计数 待补充
    send_data[17] = rcs->recv_from_mcu; // 从MCU接收帧计数 待补充
    send_data[18] = (rcs->dllg_temperature_and_flowth[0] >> 8) & 0xff; // 模组温度 
    send_data[19] = (rcs->dllg_temperature_and_flowth[0]) & 0xff; // 模组温度 
    send_data[20] = (char)dllg_get_power_mode(); // 模组功耗模式
    send_data[21] = dllg_get_cpu_usage() ; // 模组CPU占有率 待补充
    send_data[22] = dllg_get_mem_usage(); // 模组内存占用率
    send_data[23] = dllg_get_emmc_usage();  // 模组存储占用率
    (*send_len) = 8;
    return RET_OK;
}

int32_t dllg_tele_frame_03(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[24] = dllg_get_storage_usage(); // 外部存储占用率
    send_data[25] = rcs->data_trans_status; // 数传状态
    send_data[26] = (rcs->recv_from_txj >> 8) & 0xff;
    send_data[27] = (rcs->recv_from_txj) & 0xff;
    send_data[28] = 0xAA;
    send_data[29] = (rcs->send_to_txj >> 8) & 0xff;
    send_data[30] = (rcs->send_to_txj) & 0xff;
    send_data[31] = 0xAA;
    (*send_len) = 8;
    return RET_OK;
}

int32_t dllg_tele_frame_04(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[32] = rcs->telemetry_data.upload_channel; // 上注通道
    send_data[33] = rcs->telemetry_data.upload_status; // 上注状态
    send_data[34] = (rcs->telemetry_data.upload_count >> 16) & 0xff; // 接收上注帧计数
    send_data[35] = (rcs->telemetry_data.upload_count >> 8) & 0xff;  
    send_data[36] = rcs->telemetry_data.upload_count & 0xff; 
    send_data[37] = (rcs->dllg_upload_lost_count >> 8) & 0xff; // 上注帧错误计数
    send_data[38] = (rcs->dllg_upload_lost_count) & 0xff;
    send_data[39] = 0xAA;
    (*send_len) = 8;
    return RET_OK;
}

int32_t dllg_tele_frame_05(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[40] = 0xAA; // 预留
    send_data[41] = 0xAA; // 预留
    send_data[42] = 0xAA; // 预留
    send_data[43] = check_file_exist(DLLG_AIS_1_IMAGES) | check_file_exist(DLLG_AIS_2_IMAGES);  
    send_data[44] = check_file_exist("/emmc/dgp/dgp_data.dat"); 
    send_data[45] = check_file_exist("/emmc/dgp_2/dgp_data.dat"); 
    send_data[46] = check_file_exist("/emmc/red/red_data.dat"); 
    send_data[47] = 0x00;
    if (rcs->telemetry_data.app2_info[3]) send_data[47] = 0x02;
    else if (rcs->telemetry_data.app3_info[3]) send_data[47] = 0x01;
    else if (rcs->telemetry_data.app4_info[3]) send_data[47] = 0x04;
    else if (rcs->telemetry_data.app5_info[3]) send_data[47] = 0x05;
    (*send_len) = 8;
    return RET_OK;
}

int32_t dllg_tele_frame_06(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[48] = 1; // 软件版本号X
    send_data[49] = 2; // 软件版本号Y
    send_data[50] = (get_file_count(REMOTE_CTL_LOG_DIR) >> 8) & 0xff; // 日志文件数
    send_data[51] = (get_file_count(REMOTE_CTL_LOG_DIR)) & 0xff; // 日志文件数
    send_data[52] = 0xAA; // 预留
    send_data[53] = 0xAA; // 预留
    send_data[54] = 0xAA; // 预留
    send_data[55] = 1; // app组号
    (*send_len) = 8;
    return RET_OK;
}

int32_t dllg_tele_frame_07(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[56] = rcs->telemetry_data.app_id; // 第一个 APP ID
    send_data[57] = rcs->telemetry_data.app_version; 
    send_data[58] = rcs->telemetry_data.app_install_status; 
    send_data[59] = rcs->telemetry_data.app_run_status; 
    send_data[60] = rcs->telemetry_data.app_data_status;  
    send_data[61] = rcs->telemetry_data.app_pkg_status; 
    send_data[62] = rcs->telemetry_data.app_update_pkg_status;  
    send_data[63] = 0xAA;
    (*send_len) = 8;
    return RET_OK;
}

int32_t dllg_tele_frame_08(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[64] = 0xAA; // 预留
    send_data[65] = 0xAA; // 预留
    send_data[66] = 0xAA; // 预留
    send_data[67] = 0xAA; // 预留
    send_data[68] = 0xAA; // 预留
    send_data[69] = 0xAA; // 预留
    send_data[70] = 0xAA; // 预留
    send_data[71] = 0xAA; // 预留
    (*send_len) = 8;
    return RET_OK;
}

int32_t dllg_tele_frame_09(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[72] = 0xAA; // 预留
    send_data[73] = 0xAA; // 预留
    send_data[74] = 0xAA; // 预留
    send_data[75] = 0xAA; // 预留
    send_data[76] = 0xAA; // 预留
    send_data[77] = 0xAA; // 预留
    send_data[78] = 0xAA; // 预留
    send_data[79] = 0xAA; // 预留
    (*send_len) = 8;
    return RET_OK;
}

int32_t dllg_tele_frame_10(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[80] = rcs->telemetry_data.app2_info[0]; // 第二个 APP ID
    send_data[81] = rcs->telemetry_data.app2_info[1]; 
    send_data[82] = rcs->telemetry_data.app2_info[2]; 
    send_data[83] = rcs->telemetry_data.app2_info[3]; 
    send_data[84] = rcs->telemetry_data.app2_info[4]; 
    send_data[85] = rcs->telemetry_data.app2_info[5]; 
    send_data[86] = rcs->telemetry_data.app2_info[6]; 
    send_data[87] = 0xAA;
    (*send_len) = 8;
    return RET_OK;
}

int32_t dllg_tele_frame_11(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[88] = rcs->dllg_dds_app_02[0]; 
    send_data[89] = rcs->dllg_dds_app_02[1]; 
    send_data[90] = rcs->dllg_dds_app_02[2]; 
    send_data[91] = rcs->dllg_dds_app_02[3]; 
    send_data[92] = rcs->dllg_dds_app_02[4]; 
    send_data[93] = rcs->dllg_dds_app_02[5]; 
    send_data[94] = rcs->dllg_dds_app_02[6]; 
    send_data[95] = rcs->dllg_dds_app_02[7];
    (*send_len) = 8;
    return RET_OK;
}

int32_t dllg_tele_frame_12(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[96] = rcs->dllg_dds_app_02[8];  
    send_data[97] = rcs->dllg_dds_app_02[9];  
    send_data[98] = rcs->dllg_dds_app_02[10];  
    send_data[99] = rcs->dllg_dds_app_02[11];  
    send_data[100] = rcs->dllg_dds_app_02[12]; 
    send_data[101] = rcs->dllg_dds_app_02[13]; 
    send_data[102] = rcs->dllg_dds_app_02[14]; 
    send_data[103] = rcs->dllg_dds_app_02[15];  
    (*send_len) = 8;
    return RET_OK;
}

int32_t dllg_tele_frame_13(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[104] = rcs->dllg_dds_app_02[16]; 
    send_data[105] = rcs->dllg_dds_app_02[17];
    send_data[106] = rcs->dllg_dds_app_02[18];
    send_data[107] = rcs->dllg_dds_app_02[19];
    send_data[108] = rcs->dllg_dds_app_02[20];
    send_data[109] = rcs->dllg_dds_app_02[21];
    send_data[110] = rcs->dllg_dds_app_02[22];
    send_data[111] = rcs->dllg_dds_app_02[23];
    (*send_len) = 8;
    return RET_OK;
}

int32_t dllg_tele_frame_14(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[112] = rcs->dllg_dds_app_02[24]; 
    send_data[113] = rcs->dllg_dds_app_02[25];
    send_data[114] = rcs->dllg_dds_app_02[26];
    send_data[115] = rcs->dllg_dds_app_02[27];
    send_data[116] = rcs->dllg_dds_app_02[28]; 
    send_data[117] = rcs->dllg_dds_app_02[29];
    send_data[118] = rcs->dllg_dds_app_02[30];
    send_data[119] = rcs->dllg_dds_app_02[31];
    (*send_len) = 8;
    return RET_OK;
}

int32_t dllg_tele_frame_15(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[120] = rcs->telemetry_data.app3_info[0]; // 第三个 APP ID
    send_data[121] = rcs->telemetry_data.app3_info[1]; 
    send_data[122] = rcs->telemetry_data.app3_info[2]; 
    send_data[123] = rcs->telemetry_data.app3_info[3]; 
    send_data[124] = rcs->telemetry_data.app3_info[4]; 
    send_data[125] = rcs->telemetry_data.app3_info[5];
    send_data[126] = rcs->telemetry_data.app3_info[6];
    send_data[127] = 0xAA;
    (*send_len) = 8;
    return RET_OK;
}

int32_t dllg_tele_frame_16(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[128] = rcs->dllg_dds_app_03[0];
    send_data[129] = rcs->dllg_dds_app_03[1];
    send_data[130] = rcs->dllg_dds_app_03[2];
    send_data[131] = rcs->dllg_dds_app_03[3];
    send_data[132] = 0xAA;
    send_data[133] = 0xAA;
    send_data[134] = 0xAA;
    send_data[135] = 0xAA;
    (*send_len) = 8;
    return RET_OK;
}

int32_t dllg_tele_frame_17(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[136] = 0xAA;
    send_data[137] = 0xAA;
    send_data[138] = 0xAA;
    send_data[139] = 0xAA;
    send_data[140] = 0xAA;
    send_data[141] = 0xAA;
    send_data[142] = 0xAA;
    send_data[143] = 0xAA;
    (*send_len) = 8;
    return RET_OK;
}

int32_t dllg_tele_frame_18(char *send_data, int32_t *send_len)
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

int32_t dllg_tele_frame_19(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[152] = rcs->dllg_dds_app_04[0];
    send_data[153] = rcs->dllg_dds_app_04[1];
    send_data[154] = rcs->dllg_dds_app_04[2];
    send_data[155] = rcs->dllg_dds_app_04[3];
    send_data[156] = rcs->dllg_dds_app_04[4];
    send_data[157] = rcs->dllg_dds_app_04[5];
    send_data[158] = rcs->dllg_dds_app_04[6];
    send_data[159] = rcs->dllg_dds_app_04[7];
    (*send_len) = 8;
    return RET_OK;
}

int32_t dllg_tele_frame_20(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[160] = rcs->dllg_dds_app_04[8];
    send_data[161] = rcs->dllg_dds_app_04[9];
    send_data[162] = rcs->dllg_dds_app_04[10];
    send_data[163] = rcs->dllg_dds_app_04[11];
    send_data[164] = rcs->dllg_dds_app_04[12];
    send_data[165] = rcs->dllg_dds_app_04[13];
    send_data[166] = rcs->dllg_dds_app_04[14];
    send_data[167] = rcs->dllg_dds_app_04[15];
    (*send_len) = 8;
    return RET_OK;
}

int32_t dllg_tele_frame_21(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[168] = rcs->dllg_dds_app_04[16];
    send_data[169] = rcs->dllg_dds_app_04[17];
    send_data[170] = rcs->dllg_dds_app_04[18];
    send_data[171] = rcs->dllg_dds_app_04[19];
    send_data[172] = rcs->dllg_dds_app_04[20];
    send_data[173] = rcs->dllg_dds_app_04[21];
    send_data[174] = rcs->dllg_dds_app_04[22];
    send_data[175] = rcs->dllg_dds_app_04[23];
    (*send_len) = 8;
    return RET_OK;
}

int32_t dllg_tele_frame_22(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[176] = rcs->telemetry_data.app5_info[0]; // 第五个 APP ID
    send_data[177] = rcs->telemetry_data.app5_info[1];
    send_data[178] = rcs->telemetry_data.app5_info[2];
    send_data[179] = rcs->telemetry_data.app5_info[3];
    send_data[180] = rcs->telemetry_data.app5_info[4];
    send_data[181] = rcs->telemetry_data.app5_info[5];
    send_data[182] = rcs->telemetry_data.app5_info[6];
    send_data[183] = 0xAA;
    (*send_len) = 8;
    return RET_OK;
}

int32_t dllg_tele_frame_23(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[184] = rcs->dllg_dds_app_02[32];
    send_data[185] = rcs->dllg_dds_app_02[33];
    send_data[186] = rcs->dllg_dds_app_02[34];
    send_data[187] = rcs->dllg_dds_app_02[35];
    send_data[188] = rcs->dllg_dds_app_02[36];
    send_data[189] = rcs->dllg_dds_app_02[37];
    send_data[190] = rcs->dllg_dds_app_02[38];
    send_data[191] = rcs->dllg_dds_app_02[39];
    (*send_len) = 8;
    return RET_OK;
}

int32_t dllg_tele_frame_24(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[192] = rcs->dllg_dds_app_02[40];
    send_data[193] = rcs->dllg_dds_app_02[41];
    send_data[194] = rcs->dllg_dds_app_02[42];
    send_data[195] = rcs->dllg_dds_app_02[43];
    send_data[196] = rcs->dllg_dds_app_02[44];
    send_data[197] = rcs->dllg_dds_app_02[45];
    send_data[198] = rcs->dllg_dds_app_02[46];
    send_data[199] = rcs->dllg_dds_app_02[47];
    (*send_len) = 8;
    return RET_OK;
}

int32_t dllg_tele_frame_25(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[200] = rcs->dllg_dds_app_02[48];
    send_data[201] = rcs->dllg_dds_app_02[49];
    send_data[202] = rcs->dllg_dds_app_02[50];
    send_data[203] = rcs->dllg_dds_app_02[51];
    send_data[204] = rcs->dllg_dds_app_02[52];
    send_data[205] = rcs->dllg_dds_app_02[53];
    send_data[206] = rcs->dllg_dds_app_02[54];
    send_data[207] = rcs->dllg_dds_app_02[55];
    (*send_len) = 8;
    return RET_OK;
}

int32_t dllg_tele_frame_26(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[208] = rcs->dllg_dds_app_02[56];
    send_data[209] = rcs->dllg_dds_app_02[57];
    send_data[210] = rcs->dllg_dds_app_02[58]; 
    send_data[211] = rcs->dllg_dds_app_02[59];
    send_data[212] = rcs->dllg_dds_app_02[60];
    send_data[213] = rcs->dllg_dds_app_02[61];
    send_data[214] = rcs->dllg_dds_app_02[62];
    send_data[215] = rcs->dllg_dds_app_02[63];
    (*send_len) = 8;
    return RET_OK;
}

int32_t dllg_tele_frame_27(char *send_data, int32_t *send_len)
{
    remote_ctl_status_t *rcs = get_rcs();
    send_data[216] = rcs->dllg_dds_app_02[64];
    send_data[217] = rcs->dllg_dds_app_02[65];
    send_data[218] = rcs->dllg_dds_app_02[66];
    send_data[219] = rcs->dllg_dds_app_02[67];
    send_data[220] = rcs->dllg_dds_app_02[68];
    send_data[221] = rcs->dllg_dds_app_02[69];
    send_data[222] = rcs->dllg_dds_app_02[70];
    send_data[223] = rcs->dllg_dds_app_02[71];
    (*send_len) = 8;
    return RET_OK;
}
