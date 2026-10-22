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

// 内部统一通信
// 同步头
int32_t sync_head(char *send_data, int32_t *send_len)
{
  remote_ctl_status_t *rcs = get_rcs();
  send_data[0] = 0xEB;
  send_data[1] = 0x90;
  (*send_len) = 2;
  return RET_OK;
}

// 数据长度
int32_t data_length(char *send_data, int32_t *send_len)
{
  remote_ctl_status_t *rcs = get_rcs();
  send_data[0] = (rcs->dds_frame.data_len - 5) >> 8;
  // send_data[0] = 5;
  send_data[1] = rcs->dds_frame.data_len - 5;
  (*send_len) = 2;
  return RET_OK;
}

// 发送载荷主ID
int32_t send_payload_main_id(char *send_data, int32_t *send_len)
{
  remote_ctl_status_t *rcs = get_rcs();
  send_data[0] = 0x02; // 处理模块
  (*send_len) = 1;
  return RET_OK;
}

// 发送载荷子ID
int32_t send_payload_sub_id(char *send_data, int32_t *send_len)
{
  remote_ctl_status_t *rcs = get_rcs();
  send_data[0] = 0x10; // D2000
  (*send_len) = 1;
  return RET_OK;
}

// 接收载荷主ID
int32_t receive_payload_main_id(char *send_data, int32_t *send_len)
{
  remote_ctl_status_t *rcs = get_rcs();
  send_data[0] = 0x02;
  (*send_len) = 1;
  return RET_OK;
}

// 接收载荷子ID
int32_t receive_payload_sub_id(char *send_data, int32_t *send_len)
{
  remote_ctl_status_t *rcs = get_rcs();
  send_data[0] = 0x10;
  (*send_len) = 1;
  return RET_OK;
}

// 数据类型
int32_t data_type(char *send_data, int32_t *send_len)
{
  remote_ctl_status_t *rcs = get_rcs();
  send_data[0] = rcs->dds_frame.type;
  // send_data[0] = 1;
  (*send_len) = 1;
  return RET_OK;
}

// 发送帧计数
int32_t send_frame_count(char *send_data, int32_t *send_len)
{
  remote_ctl_status_t *rcs = get_rcs();
  send_data[0] = 0;
  (*send_len) = 1;
  return RET_OK;
}

// 发送时刻
int32_t time_of_send(char *send_data, int32_t *send_len)
{
  remote_ctl_status_t *rcs = get_rcs();
  get_system_time(send_data, send_len); // 计算当前时间距离2018年1月1日的秒数
  (*send_len) = 4;
  return RET_OK;
}

// 数据正文
int32_t data_message(char *send_data, int32_t *send_len)
{
  remote_ctl_status_t *rcs = get_rcs();
  // int data_message_length = sizeof(rcs->data);
  for (int i = 0; i < 24; ++i)
  {
    send_data[i] = rcs->dds_frame.data[i];
    // send_data[i] = 1;
  }
  (*send_len) = 24;
  return RET_OK;
}

// 校验和
int32_t sum_of_check(char *send_data, int32_t *send_len)
{
  remote_ctl_status_t *rcs = get_rcs();
  send_data[0] = 0;
  (*send_len) = 1;
  return RET_CHECK_SUM;
}
