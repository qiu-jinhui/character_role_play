#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Tuple
import zhipuai
import requests
import logging
from tqdm import tqdm
import sys

# 确保使用UTF-8编码，解决中文显示问题
if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

# 配置日志系统，记录程序运行过程
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RolePlayGenerator:
    def __init__(self, api_key=None):
        """
        初始化角色扮演对话生成器
        
        参数:
            api_key: 智谱AI的API密钥，如未提供则尝试从环境变量获取
        """
        self.api_key = api_key or os.environ.get("ZHIPUAI_API_KEY")
        if not self.api_key:
            raise ValueError("需要提供 ZHIPUAI API KEY")
        # 初始化智谱AI客户端
        self.client = zhipuai.ZhipuAI(api_key=self.api_key)

    def generate_character_profile(self, text: str, character_name=None) -> str:
        """
        根据输入的文本生成角色人设
        
        参数:
            text: 用于生成角色人设的文本内容
            character_name: 指定角色名称，如不指定则自动生成
            
        返回:
            str: 生成的角色人设描述文本
        """
        logger.info("正在根据文本生成角色人设...")
        
        # 创建一个备用的角色人设模板，当API调用失败时使用
        fallback_name = character_name or '默认角色'
        fallback_profile = f"""
角色名称：{fallback_name}
年龄：30岁
性别：未知
外貌特征：普通外表
性格特点：平和、友善
说话风格：平实、客观
背景故事：普通人的生活经历
行为方式：正常社交行为
        """
        
        # 构建提示词，根据是否指定角色名称调整指令
        name_instruction = ""
        if character_name:
            name_instruction = f"请将角色名称设置为'{character_name}'。"
        else:
            name_instruction = "请自行创建一个合适的角色名称。"
        
        # 构建完整的提示词，要求模型生成结构化的角色人设
        prompt = f"""请根据以下文本创建一个角色人设。{name_instruction}人设应包括：
1. 角色名称
2. 年龄
3. 性别
4. 外貌特征
5. 性格特点
6. 说话风格和习惯用语
7. 背景故事
8. 行为方式和习惯

以下是文本：
{text}

请以结构化的格式输出完整的角色人设："""

        try:
            # 调用智谱AI的ChatGLM模型生成角色人设
            response = self.client.chat.completions.create(
                model="chatglm_turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,  # 控制生成内容的随机性
                top_p=0.9,        # 控制词汇分布的采样范围
            )
            
            # 从API响应中提取生成的内容
            character_profile = response.choices[0].message.content
            logger.info("角色人设生成成功")
            return character_profile
        except Exception as e:
            # 异常处理，返回备用人设
            logger.error(f"生成角色人设失败: {str(e)}")
            return fallback_profile

    def generate_dialogue(self, character1: str, character2: str, num_turns: int = 5, init_prompt: str = None) -> List[Dict]:
        """
        生成两个角色之间的对话
        
        参数:
            character1: 第一个角色的人设
            character2: 第二个角色的人设
            num_turns: 生成对话的回合数
            init_prompt: 初始话题或场景提示
            
        返回:
            List[Dict]: 生成的对话列表，每项包含角色和对话内容
        """
        logger.info(f"开始生成 {num_turns} 轮对话...")
        
        # 如果未提供初始场景，使用默认场景
        if not init_prompt:
            init_prompt = "你们偶然在咖啡店相遇，开始一段对话。"
        
        dialogues = []  # 存储生成的对话
        history = []    # 存储对话历史，用于API调用
        
        # 构建系统提示词，用于指导模型生成对话
        system_prompt = f"""你将模拟两个角色之间的对话。这些角色是：
        
角色1: {character1}

角色2: {character2}

初始场景: {init_prompt}

请生成角色之间的对话，确保每个角色的回应符合其人设。每次只需要生成一个角色的一句话。"""
        
        # 定义备用回复，在API调用失败时使用
        default_replies = [
            "你好，很高兴认识你。",
            "今天天气真不错，不是吗？",
            "最近过得怎么样？",
            "我最近在思考一些人生问题。",
            "有时候我觉得生活充满了惊喜。",
            "能和你聊天很开心。",
            "我们下次再聊吧。"
        ]
        
        # 交替生成两个角色的对话
        current_character = 1  # 从角色1开始
        
        try:
            # 使用tqdm显示进度条
            for turn in tqdm(range(num_turns * 2)):
                # 构建提示词，根据是第一轮还是后续轮次调整
                if turn == 0:
                    prompt = f"请生成角色1的第一句对话。"
                else:
                    prompt = f"请根据上下文，生成角色{current_character}的下一句对话。"
                
                # 将用户提示添加到历史中
                history.append({"role": "user", "content": prompt})
                
                try:
                    # 调用智谱AI的ChatGLM模型生成对话
                    response = self.client.chat.completions.create(
                        model="chatglm_turbo",
                        messages=[
                            # 在system提示中说明当前应该扮演哪个角色
                            {"role": "system", "content": f"{system_prompt}\n当前请扮演角色{current_character}，根据对话历史生成下一句话。"},
                            # 包含之前的对话历史
                            *history
                        ],
                        temperature=0.8,  # 增加随机性，使对话更多样化
                        top_p=0.9,
                    )
                    
                    # 从API响应中提取生成的内容
                    reply = response.choices[0].message.content
                    
                    # 记录对话
                    dialogues.append({
                        "role": f"角色{current_character}",
                        "content": reply
                    })
                    
                    # 更新历史，添加模型的回复
                    history.append({"role": "assistant", "content": reply})
                    
                    # 切换角色，实现交替对话
                    current_character = 2 if current_character == 1 else 1
                    
                    # 防止API限流，添加延迟
                    time.sleep(1)
                except Exception as e:
                    # 异常处理，使用备用回复
                    logger.error(f"对话生成失败: {str(e)}")
                    # 使用预定义的回复列表，根据当前轮次选择不同回复
                    default_reply = default_replies[turn % len(default_replies)]
                    dialogues.append({
                        "role": f"角色{current_character}",
                        "content": default_reply
                    })
                    history.append({"role": "assistant", "content": default_reply})
                    current_character = 2 if current_character == 1 else 1
                    time.sleep(1)
        except Exception as e:
            # 整体异常处理
            logger.error(f"生成对话时出错: {str(e)}")
        
        logger.info(f"成功生成 {len(dialogues)} 条对话")
        return dialogues

    def save_dialogue(self, dialogues: List[Dict], character1: str, character2: str, output_dir: str = "outputs"):
        """
        保存生成的对话到文件
        
        参数:
            dialogues: 对话列表
            character1: 第一个角色的人设
            character2: 第二个角色的人设
            output_dir: 输出目录
        """
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 使用时间戳作为文件名，确保唯一性
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"dialogue_{timestamp}.json")
        
        # 从人设中提取角色名称
        char1_name = self._extract_character_name(character1)
        char2_name = self._extract_character_name(character2)
        
        # 构建对话数据结构，包含时间戳、角色信息和对话内容
        data = {
            "timestamp": timestamp,
            "characters": {
                "character1": {
                    "name": char1_name,
                    "profile": character1
                },
                "character2": {
                    "name": char2_name,
                    "profile": character2
                }
            },
            "dialogues": dialogues
        }
        
        # 保存为JSON格式
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"对话已保存到 {output_file}")
        
        # 同时保存一个易于阅读的文本版本
        txt_output_file = os.path.join(output_dir, f"dialogue_{timestamp}.txt")
        with open(txt_output_file, 'w', encoding='utf-8') as f:
            # 写入角色信息
            f.write(f"角色1 ({char1_name}):\n{character1}\n\n")
            f.write(f"角色2 ({char2_name}):\n{character2}\n\n")
            f.write("="*50 + "\n对话内容\n" + "="*50 + "\n\n")
            
            # 写入对话内容，替换角色编号为角色名称
            for dialogue in dialogues:
                role = dialogue["role"]
                # 替换角色名
                if role == "角色1":
                    role = char1_name
                elif role == "角色2":
                    role = char2_name
                    
                f.write(f"{role}: {dialogue['content']}\n\n")
                
        logger.info(f"对话文本版本已保存到 {txt_output_file}")
    
    def _extract_character_name(self, profile: str) -> str:
        """
        从人设中提取角色名称
        
        参数:
            profile: 角色人设文本
            
        返回:
            str: 提取出的角色名称，如果提取失败则返回"未知角色"
        """
        try:
            # 尝试查找"角色名称"或"名称"或"姓名"后面的内容
            lines = profile.split('\n')
            for line in lines:
                if "角色名称" in line or "名称" in line or "姓名" in line:
                    return line.split("：")[-1].strip()
            # 如果找不到，返回默认名称
            return "未知角色"
        except:
            return "未知角色"

def main():
    """
    主函数，处理命令行参数并执行角色对话生成流程
    """
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="角色扮演对话生成工具")
    parser.add_argument("--api_key", type=str, help="智谱AI API密钥")
    parser.add_argument("--text_file", type=str, help="用于生成角色的文本文件")
    parser.add_argument("--char1", type=str, help="角色1的人设描述")
    parser.add_argument("--char2", type=str, help="角色2的人设描述")
    parser.add_argument("--name1", type=str, help="角色1的名称")
    parser.add_argument("--name2", type=str, help="角色2的名称")
    parser.add_argument("--turns", type=int, default=5, help="对话回合数")
    parser.add_argument("--init_prompt", type=str, help="对话初始提示")
    parser.add_argument("--output_dir", type=str, default="outputs", help="输出目录")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    
    args = parser.parse_args()
    
    # 启用调试模式，输出更详细的日志
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("调试模式已启用")
    
    # 获取API密钥，优先使用命令行参数，其次是环境变量
    api_key = args.api_key or os.environ.get("ZHIPUAI_API_KEY")
    
    # 如果未提供API密钥，提示用户输入
    if not api_key:
        api_key = input("请输入智谱API密钥: ")
    
    try:
        # 创建角色对话生成器实例
        generator = RolePlayGenerator(api_key=api_key)
        
        # 从命令行参数获取角色人设
        character1 = args.char1
        character2 = args.char2
        
        # 如果提供了文本文件且未提供角色人设，则根据文本生成角色人设
        if args.text_file and not (character1 and character2):
            try:
                # 读取文本文件
                with open(args.text_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                    
                # 生成角色1的人设
                if not character1:
                    print("正在生成角色1的人设...")
                    character1 = generator.generate_character_profile(text, character_name=args.name1)
                    print(f"角色1人设:\n{character1}\n")
                
                # 生成角色2的人设
                if not character2:
                    print("正在生成角色2的人设...")
                    # 为角色2生成不同的名称，避免与角色1重名
                    char2_name = args.name2
                    if not char2_name and args.name1:
                        char2_name = "另一个角色"  # 如果只指定了角色1的名称，为角色2设置一个默认值
                    character2 = generator.generate_character_profile(text, character_name=char2_name)
                    print(f"角色2人设:\n{character2}\n")
            except Exception as e:
                print(f"读取文件或生成人设时出错: {e}")
                return
        
        # 如果仍未提供角色人设，使用示例文本生成
        if not character1 or not character2:
            example_text = """
            # 《红楼梦》节选
            
            贾母笑道："你这个老货，真是又耳聋又眼花的。这是宝玉的小丫头襻纱，那是宝玉的。襻纱因那日开了稍子，拿去给他作那镯子的带子，宝玉见了，说送给她去了。你敢说是我们家的丫头就拿了去了的不成？"刘姥姥听了，忙陪笑道："我眼花，我眼花。姑娘家大了，他们那里肯亲近这些东西。依我说，小姑娘大了，身上带的东西，越发要好才是，姑娘的体面多显得大方。姑娘一出门，人家只看这些妆饰，就知道是大家小姐了。
            
            只见凤姐儿笑向宝钗道："你瞧瞧这老货，'吃着碗里瞧着锅里'，这会子又拿你来讨好儿。"刘姥姥听了，忙笑道："这那里说起，我见姑娘说话理道儿又好听，又有救人的心肠，怎么不正眼儿瞧瞧我．没的不讨一口好气儿呢。"
            """
            
            print("使用示例文本生成角色人设...")
            # 使用红楼梦中的角色名称
            character1 = generator.generate_character_profile(example_text, character_name="贾母")
            print(f"角色1人设:\n{character1}\n")
            
            character2 = generator.generate_character_profile(example_text, character_name="刘姥姥")
            print(f"角色2人设:\n{character2}\n")
        
        # 生成对话
        print(f"开始生成 {args.turns} 轮对话...")
        dialogues = generator.generate_dialogue(
            character1=character1, 
            character2=character2, 
            num_turns=args.turns,
            init_prompt=args.init_prompt
        )
        
        # 保存对话
        generator.save_dialogue(
            dialogues=dialogues,
            character1=character1,
            character2=character2,
            output_dir=args.output_dir
        )
        
        print("对话生成完成！")
    except Exception as e:
        # 捕获并记录所有异常
        logger.error(f"执行过程中出错: {str(e)}")
        print(f"执行过程中出错: {e}")

if __name__ == "__main__":
    main() 