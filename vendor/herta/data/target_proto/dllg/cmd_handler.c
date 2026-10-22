#include <stdio.h>
#include <stdint.h>
#include "frame_process.h"
#include "cmd_utils.h"
#include "handler.h"
#include "fpgaMiscFunc.h"

#include "C_DDSPub.h"

// 检查 appid 范围
static int32_t check_appid(char appid)
{
  if (appid < MINI_APP_ID || appid > MAX_APP_ID)
    return RET_ERR;
  return RET_OK;
}

// 检查上注通道设置
static int32_t check_channel(char channel)
{
  // 0x0F 表示所有平台
  if (channel == 0x00 || channel == 0x01 || channel == 0x02 || channel == 0x0F)
    return RET_OK;
  return RET_ERR;
}

// 检查 01 类型数据，比如平台等
static int32_t check_01(char platform)
{
  if (platform == 0x00 || platform == 0x01)
    return RET_OK;
  return RET_ERR;
}


extern char internal_send_data[INTERNAL_DATA_LEN];
extern DDSPub_t *pub;
/*
* @brief 通过dds发送内部数据
* @param topicType 主题类型
* @return 函数执行状态
*/
// topicType=0
static int dds_send_internal_data(int topicType)
{
	char jsonStr[500] = "";
	int INTERNAL_INSTRUCTION_LEN = sizeof(internal_send_data); // 已经被赋值，就是从数据库发给app的数据，internal_send_data=buf，buf在frame_process.c文件里
  	// log_debug("%d", INTERNAL_INSTRUCTION_LEN);
	
  // 把传来的internal_send_data变成json格式存到jsonStr里
	if (pub != NULL) 
	{
		// 组装json字符串
		sprintf(jsonStr + strlen(jsonStr), "{\"sync_head\":");
		sprintf(jsonStr + strlen(jsonStr), "%d", (unsigned char)internal_send_data[0] << 8 | 
												(unsigned char)internal_send_data[1]);
		sprintf(jsonStr + strlen(jsonStr), ", ");
		
		sprintf(jsonStr + strlen(jsonStr), "\"data_length\":");
		sprintf(jsonStr + strlen(jsonStr), "%d", (unsigned char)internal_send_data[2] << 8 |
												(unsigned char)internal_send_data[3]);
		sprintf(jsonStr + strlen(jsonStr), ", ");

		sprintf(jsonStr + strlen(jsonStr), "\"send_payload_main_id\":%d, ", (unsigned char)internal_send_data[4]);
		sprintf(jsonStr + strlen(jsonStr), "\"send_payload_sub_id\":%d, ", (unsigned char)internal_send_data[5]);
		sprintf(jsonStr + strlen(jsonStr), "\"receive_payload_main_id\":%d, ", (unsigned char)internal_send_data[6]);
		sprintf(jsonStr + strlen(jsonStr), "\"receive_payload_sub_id\":%d, ", (unsigned char)internal_send_data[7]);
		sprintf(jsonStr + strlen(jsonStr), "\"data_type\":%d, ", (unsigned char)internal_send_data[8]);
		sprintf(jsonStr + strlen(jsonStr), "\"send_frame_count\":%d, ", (unsigned char)internal_send_data[9]);

		sprintf(jsonStr + strlen(jsonStr), "\"time_of_send\":");
		sprintf(jsonStr + strlen(jsonStr), "%d", (unsigned char)internal_send_data[10] << 24 |
												(unsigned char)internal_send_data[11] << 16 |
												(unsigned char)internal_send_data[12] << 8 |
												(unsigned char)internal_send_data[13]);

		sprintf(jsonStr + strlen(jsonStr), ", ");

		// 格式化字节序列为十六进制字符串
		sprintf(jsonStr + strlen(jsonStr), "\"data_message\":[");
		for (size_t i = 14; i < INTERNAL_INSTRUCTION_LEN - 2; i++) {
			sprintf(jsonStr + strlen(jsonStr), "%d, ", (unsigned char)internal_send_data[i]);
		}
		sprintf(jsonStr + strlen(jsonStr), "%d", (unsigned char)internal_send_data[INTERNAL_INSTRUCTION_LEN - 2]);
		sprintf(jsonStr + strlen(jsonStr), "], ");

		sprintf(jsonStr + strlen(jsonStr), "\"sum_of_check\":%d}", (unsigned char)internal_send_data[INTERNAL_INSTRUCTION_LEN - 1]);
		jsonStr[strlen(jsonStr)] = '\0';
		
		log_debug("jsonStr: %s\n", jsonStr);
    if(topicType == APP_DATA_INDEX)
    {
      log_debug("dds topicType: APP_DATA_INDEX\n");
    }
    else if(topicType == INSTRUCTIONS_DATA_INDEX)
    {
      log_debug("dds topicType: INSTRUCTIONS_DATA_INDEX\n");
    }
    
		if (DDSPub_sendMsg(pub, topicType, jsonStr)) 
		{
			log_debug("dds send data: %s\n", jsonStr);
      log_debug("Message sent successfully.\n");
		}
		else 
		{
			log_err("Failed to send message.\n");
		}
		
	} 
	else
	{
		log_err("Failed to create DDSPub_t instance.\n");
	}
	return RET_OK;
}

int32_t app_start(char *cmd)
{
  char appid = cmd[2];
  // 检验间接指令正确性
  if (check_appid(appid))
  {
    log_err("cmd check error.\n");
    return RET_ERR;
  }
  dds_send_internal_data(INSTRUCTIONS_DATA_INDEX); // INSTRUCTIONS_DATA_INDEX=0
  return RET_OK;
}

int32_t app_stop(char *cmd)
{
  char appid = cmd[2];
  // 检验间接指令正确性
  if (check_appid(appid))
  {
    log_err("cmd check error.\n");
    return RET_ERR;
  }
  dds_send_internal_data(INSTRUCTIONS_DATA_INDEX);
  return RET_OK;
}

int32_t app_update(char *cmd)
{
  char appid = cmd[2];
  // 检验间接指令正确性
  if (check_appid(appid))
  {
    log_err("cmd check error.\n");
    return RET_ERR;
  }
  dds_send_internal_data(INSTRUCTIONS_DATA_INDEX);
  return RET_OK;
}

int32_t app_install(char *cmd)
{
  char appid = cmd[2];
  // 检验间接指令正确性
  if (check_appid(appid))
  {
    log_err("cmd check error.\n");
    return RET_ERR;
  }
  dds_send_internal_data(INSTRUCTIONS_DATA_INDEX);
  return RET_OK;
}

int32_t app_uninstall(char *cmd)
{
  char appid = cmd[2];
  // 检验间接指令正确性
  if (check_appid(appid))
  {
    log_err("cmd check error.\n");
    return RET_ERR;
  }
  dds_send_internal_data(INSTRUCTIONS_DATA_INDEX);
  return RET_OK;
}

int32_t app_roll_back(char *cmd)
{
  char appid = cmd[2];
  // 检验间接指令正确性
  if (check_appid(appid))
  {
    log_err("cmd check error.\n");
    return RET_ERR;
  }
  dds_send_internal_data(INSTRUCTIONS_DATA_INDEX);
  return RET_OK;
}

int32_t app_data_clear(char *cmd)
{
  char appid = cmd[2];
  // 检验间接指令正确性
  if (check_appid(appid))
  {
    log_err("cmd check error.\n");
    return RET_ERR;
  }
  dds_send_internal_data(INSTRUCTIONS_DATA_INDEX);
  return RET_OK;
}

int32_t app_internal_indirect_instructions(char *cmd)
{
  char appid = cmd[2];
  // 检验间接指令正确性
  if (check_appid(appid))
  {
    log_err("cmd check error.\n");
    return RET_ERR;
  }
  dds_send_internal_data(APP_DATA_INDEX);
  log_debug("app_internal_indirect_instructions\n");
  return RET_OK;
}

int32_t app_id_switch(char *cmd)
{
  char appid = cmd[2];
  // 检验间接指令正确性
  if (check_appid(appid))
  {
    log_err("cmd check error.\n");
    return RET_ERR;
  }
  dds_send_internal_data(INSTRUCTIONS_DATA_INDEX);
  return RET_OK;
}

int32_t app_upload_start(char *cmd)
{
  char channel = cmd[2];
  // 检验间接指令正确性
  if (check_channel(channel))
  {
    log_err("cmd check error.\n");
    return RET_ERR;
  }
  dds_send_internal_data(INSTRUCTIONS_DATA_INDEX);
  return RET_OK;
}

int32_t app_upload_pause(char *cmd)
{
  char channel = cmd[2];
  // 检验间接指令正确性
  if (check_channel(channel))
  {
    log_err("cmd check error.\n");
    return RET_ERR;
  }
  dds_send_internal_data(INSTRUCTIONS_DATA_INDEX);
  return RET_OK;
}

int32_t app_upload_stop(char *cmd)
{
  char channel = cmd[2];
  // 检验间接指令正确性
  if (check_channel(channel))
  {
    log_err("cmd check error.\n");
    return RET_ERR;
  }
  dds_send_internal_data(INSTRUCTIONS_DATA_INDEX);
  return RET_OK;
}

int32_t app_upload_exit(char *cmd)
{
  char channel = cmd[2];
  // 检验间接指令正确性
  if (check_channel(channel))
  {
    log_err("cmd check error.\n");
    return RET_ERR;
  }
  dds_send_internal_data(INSTRUCTIONS_DATA_INDEX);
  return RET_OK;
}

int32_t app_clear_upload_packet(char *cmd)
{
  dds_send_internal_data(INSTRUCTIONS_DATA_INDEX);
  return RET_OK;
}

int32_t log_download(char *cmd)
{
  char logid = cmd[2];
  // 检验间接指令正确性
  if (check_01(logid))
  {
    log_err("cmd check error.\n");
    return RET_ERR;
  }
  dds_send_internal_data(INSTRUCTIONS_DATA_INDEX);
  return RET_OK;
}

int32_t app_data_download(char *cmd)
{
  char appid = cmd[2];
  // 检验间接指令正确性
  if (check_appid(appid))
  {
    log_err("cmd check error.\n");
    return RET_ERR;
  }
  dds_send_internal_data(INSTRUCTIONS_DATA_INDEX);
  return RET_OK;
}

int32_t log_clear(char *cmd)
{
  dds_send_internal_data(INSTRUCTIONS_DATA_INDEX);
  return RET_OK;
}

int32_t select_board_status(char *cmd)
{
  char boardid = cmd[2];
  // 检验间接指令正确性
  if (check_01(boardid))
  {
    log_err("cmd check error.\n");
    return RET_ERR;
  }
  dds_send_internal_data(INSTRUCTIONS_DATA_INDEX);
  return RET_OK;
}

// dllg授时指令 
int32_t give_time(char *cmd)
{
  /*char boardid = cmd[2];
  // 检验间接指令正确性
  if (check_01(boardid))
  {
    log_err("cmd check error.\n");
    return RET_ERR;
  }*/
  dds_send_internal_data(INSTRUCTIONS_DATA_INDEX);
  return RET_OK;
}

// dllg 工程参数采集指令
int32_t engineer_parameter_collect(char *cmd)
{
  /*char boardid = cmd[2];
  // 检验间接指令正确性
  if (check_01(boardid))
  {
    log_err("cmd check error.\n");
    return RET_ERR;
  }*/
  dds_send_internal_data(INSTRUCTIONS_DATA_INDEX);
  return RET_OK;
}

// dllg app管理指令
int32_t app_manager(char *cmd)
{
  /*char boardid = cmd[2];
  // 检验间接指令正确性
  if (check_01(boardid))
  {
    log_err("cmd check error.\n");
    return RET_ERR;
  }*/
  dds_send_internal_data(INSTRUCTIONS_DATA_INDEX);
  return RET_OK;
}

// dllg 上注管理指令
int32_t upload_manager(char *cmd)
{
  /*char boardid = cmd[2];
  // 检验间接指令正确性
  if (check_01(boardid))
  {
    log_err("cmd check error.\n");
    return RET_ERR;
  }*/
  dds_send_internal_data(INSTRUCTIONS_DATA_INDEX);
  return RET_OK;
}

// dllg 数传指令
int32_t data_trans(char *cmd)
{
  /*char boardid = cmd[2];
  // 检验间接指令正确性
  if (check_01(boardid))
  {
    log_err("cmd check error.\n");
    return RET_ERR;
  }*/
  dds_send_internal_data(INSTRUCTIONS_DATA_INDEX);
  return RET_OK;
}

// dllg 数据清除指令
int32_t data_clear(char *cmd)
{
  /*char boardid = cmd[2];
  // 检验间接指令正确性
  if (check_01(boardid))
  {
    log_err("cmd check error.\n");
    return RET_ERR;
  }*/
  dds_send_internal_data(INSTRUCTIONS_DATA_INDEX);
  return RET_OK;
}

// dllg 设置功耗模式指令帧
int32_t set_power_mode(char *cmd)
{
  /*char boardid = cmd[2];
  // 检验间接指令正确性
  if (check_01(boardid))
  {
    log_err("cmd check error.\n");
    return RET_ERR;
  }*/
  dds_send_internal_data(INSTRUCTIONS_DATA_INDEX);
  return RET_OK;
}

// gdwx app管理指令
int32_t gdwx_app_manager(char *cmd)
{
  /*char boardid = cmd[2];
  // 检验间接指令正确性
  if (check_01(boardid))
  {
    log_err("cmd check error.\n");
    return RET_ERR;
  }*/
  dds_send_internal_data(INSTRUCTIONS_DATA_INDEX);
  return RET_OK;
}

// gdwx 内部间接指令
int32_t gdwx_internal_cmd(char *cmd)
{
  /*char boardid = cmd[2];
  // 检验间接指令正确性
  if (check_01(boardid))
  {
    log_err("cmd check error.\n");
    return RET_ERR;
  }*/
  dds_send_internal_data(INSTRUCTIONS_DATA_INDEX);
  return RET_OK;
}

// gdwx 上注管理指令
int32_t gdwx_upload_manager(char *cmd)
{
  /*char boardid = cmd[2];
  // 检验间接指令正确性
  if (check_01(boardid))
  {
    log_err("cmd check error.\n");
    return RET_ERR;
  }*/
  dds_send_internal_data(INSTRUCTIONS_DATA_INDEX);
  return RET_OK;
}

// gdwx 数传指令
int32_t gdwx_data_trans(char *cmd)
{
  /*char boardid = cmd[2];
  // 检验间接指令正确性
  if (check_01(boardid))
  {
    log_err("cmd check error.\n");
    return RET_ERR;
  }*/
  dds_send_internal_data(INSTRUCTIONS_DATA_INDEX);
  return RET_OK;
}

// gdwx 数据清除指令
int32_t gdwx_data_clear(char *cmd)
{
  /*char boardid = cmd[2];
  // 检验间接指令正确性
  if (check_01(boardid))
  {
    log_err("cmd check error.\n");
    return RET_ERR;
  }*/
  dds_send_internal_data(INSTRUCTIONS_DATA_INDEX);
  return RET_OK;
}

// gdwx 设置功耗模式指令帧
int32_t gdwx_set_power_mode(char *cmd)
{
  /*char boardid = cmd[2];
  // 检验间接指令正确性
  if (check_01(boardid))
  {
    log_err("cmd check error.\n");
    return RET_ERR;
  }*/
  dds_send_internal_data(INSTRUCTIONS_DATA_INDEX);
  return RET_OK;
}

// gdwx 设置任务模式指令帧
int32_t gdwx_set_task_mode(char *cmd)
{
  /*char boardid = cmd[2];
  // 检验间接指令正确性
  if (check_01(boardid))
  {
    log_err("cmd check error.\n");
    return RET_ERR;
  }*/
  dds_send_internal_data(INSTRUCTIONS_DATA_INDEX);
  return RET_OK;
}
