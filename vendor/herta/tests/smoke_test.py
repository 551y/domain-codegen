"""
冒烟测试脚本

验证：
1. 能读取环境变量
2. 能初始化 BigModelClient
3. 能调用 embed_texts
4. 能调用 chat

运行前请确保：
- 已激活 herta-rag 环境: conda activate herta-rag
- 已设置环境变量: export ZAI_API_KEY='your-api-key'

运行方式：
    cd herta
    python tests/smoke_test.py
"""

import os
import sys

# 将 src 目录添加到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from herta import configure_logging, get_settings, reload_settings
from herta.llm import BigModelClient


def test_environment_variables():
    """测试1: 验证环境变量读取"""
    print("\n" + "=" * 50)
    print("测试1: 环境变量读取")
    print("=" * 50)

    # 检查关键环境变量
    api_key = os.environ.get("ZAI_API_KEY")

    if not api_key:
        print("❌ ZAI_API_KEY 未设置")
        print("   请先设置: export ZAI_API_KEY='your-api-key'")
        return False

    print(f" ZAI_API_KEY 已设置 (长度: {len(api_key)})")

    # 尝试加载配置
    try:
        # 清除缓存，强制重新加载
        settings = reload_settings()
        print(f"   配置加载成功")
        print(f"   聊天模型: {settings.chat_model}")
        print(f"   Embedding模型: {settings.embedding_model}")
        print(f"   Embedding维度: {settings.embedding_dimensions}")
        return True
    except ValueError as e:
        print(f"   配置加载失败: {e}")
        return False


def test_client_initialization():
    """测试2: 验证 BigModelClient 初始化"""
    print("\n" + "=" * 50)
    print("测试2: BigModelClient 初始化")
    print("=" * 50)

    try:
        settings = get_settings()
        client = BigModelClient(settings)
        print("   BigModelClient 初始化成功")
        print(f"   API Key 前8位: {settings.zai_api_key[:8]}...")
        return True, client
    except Exception as e:
        print(f"   BigModelClient 初始化失败: {e}")
        return False, None


def test_embed_texts(client):
    """测试3: 验证 embed_texts 调用"""
    print("\n" + "=" * 50)
    print("测试3: embed_texts 调用")
    print("=" * 50)

    try:
        texts = ["hello world", "你好世界", "这是一个测试"]
        print(f"输入文本: {texts}")

        embeddings = client.embed_texts(texts)

        print(f"   调用成功，返回 {len(embeddings)} 个向量")

        for i, emb in enumerate(embeddings):
            print(f"   向量{i+1} 维度: {len(emb)}")

        # 验证维度
        expected_dim = get_settings().embedding_dimensions
        if len(embeddings[0]) == expected_dim:
            print(f"   向量维度正确 ({expected_dim})")
        else:
            print(f"   向量维度错误，期望 {expected_dim}，实际 {len(embeddings[0])}")

        return True
    except Exception as e:
        print(f"   embed_texts 调用失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chat(client):
    """测试4: 验证 chat 调用"""
    print("\n" + "=" * 50)
    print("测试4: chat 调用")
    print("=" * 50)

    try:
        messages = [
            {"role": "system", "content": "你是一个简洁的助手。"},
            {"role": "user", "content": "你好，请用一句话介绍自己。"},
        ]
        print(f"输入消息: {messages}")

        response = client.chat(messages, temperature=0.1)

        print(f"   调用成功")
        print(f"   返回内容: {response[:100]}...")
        return True
    except Exception as e:
        print(f"   chat 调用失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_input_validation(client):
    """测试5: 验证输入校验"""
    print("\n" + "=" * 50)
    print("测试5: 输入校验")
    print("=" * 50)

    all_passed = True

    # 测试 embed_texts 空输入
    try:
        client.embed_texts([])
        print("   embed_texts 空列表应抛出 ValueError")
        all_passed = False
    except ValueError as e:
        print(f"   embed_texts 空列表校验通过: {e}")

    # 测试 embed_texts 空字符串
    try:
        client.embed_texts([""])
        print("   embed_texts 空字符串应抛出 ValueError")
        all_passed = False
    except ValueError as e:
        print(f"   embed_texts 空字符串校验通过: {e}")

    # 测试 chat 空输入
    try:
        client.chat([])
        print("   chat 空列表应抛出 ValueError")
        all_passed = False
    except ValueError as e:
        print(f"   chat 空列表校验通过: {e}")

    # 测试 embed_query
    try:
        client.embed_query("")
        print("   embed_query 空字符串应抛出 ValueError")
        all_passed = False
    except ValueError as e:
        print(f"   embed_query 空字符串校验通过: {e}")

    return all_passed


def main():
    """主测试流程"""
    print("\n" + "=" * 50)
    print("Herta Stage 1  - SmokeTest")
    print("=" * 50)

    # 配置日志
    configure_logging("INFO")

    # 运行测试
    results = []

    # 测试1: 环境变量
    passed = test_environment_variables()
    results.append(("环境变量读取", passed))

    if not passed:
        print("\n 环境变量测试失败，跳过后续测试")
        print("请设置 ZAI_API_KEY 环境变量后再试")
        sys.exit(1)

    # 测试2: 客户端初始化
    passed, client = test_client_initialization()
    results.append(("BigModelClient 初始化", passed))

    if not passed or client is None:
        print("\n 客户端初始化失败，跳过后续测试")
        sys.exit(1)

    # 测试3: embed_texts
    passed = test_embed_texts(client)
    results.append(("embed_texts 调用", passed))

    # 测试4: chat
    passed = test_chat(client)
    results.append(("chat 调用", passed))

    # 测试5: 输入校验
    passed = test_input_validation(client)
    results.append(("输入校验", passed))

    # 汇总结果
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)

    for name, passed in results:
        status = "通过" if passed else "失败"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n所有测试通过！Stage 1 SmokeTest 成功。")
        return 0
    else:
        print("\n 部分测试失败，请检查错误信息。")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
