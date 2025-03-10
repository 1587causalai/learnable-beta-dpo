import os
import json
import torch
import torch.nn as nn

from .transformer import QwenTransformer


class QwenLLM(nn.Module):
    """
    类似于Qwen-0.5B的完整语言模型
    包括模型加载、保存和推理功能
    """
    def __init__(
        self,
        config=None,
        checkpoint_path=None,
    ):
        super().__init__()
        
        # 加载配置
        if isinstance(config, dict):
            # 如果配置是字典，直接使用
            self.config = config
        elif isinstance(config, str) and os.path.isfile(config):
            # 如果配置是文件路径，加载配置文件
            with open(config, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            # 使用默认配置
            self.config = {
                "vocab_size": 151646,
                "hidden_size": 896,
                "num_hidden_layers": 24,
                "num_q_heads": 14,
                "num_kv_heads": 2,
                "head_dim": 64,
                "intermediate_size": 4864,
                "activation_type": "swiglu",
                "max_position_embeddings": 32768,
                "tie_word_embeddings": True,
                "layer_norm_eps": 1e-5,
                "dropout": 0.0,
                "attention_dropout": 0.0,
                "use_dual_chunk": False,
                "pad_token_id": 0,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "initializer_range": 0.02,
                "use_cache": True,
            }
        
        # 创建模型
        self.transformer = QwenTransformer(**self.config)
        
        # 如果提供了检查点路径，加载权重
        if checkpoint_path:
            self.load_weights(checkpoint_path)
    
    def forward(self, *args, **kwargs):
        """
        前向传播，委托给内部transformer
        """
        return self.transformer(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """
        文本生成，委托给内部transformer
        """
        return self.transformer.generate(*args, **kwargs)
    
    def save_pretrained(self, save_dir):
        """
        保存模型权重和配置到指定目录
        """
        # 确保目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存配置
        with open(os.path.join(save_dir, "config.json"), 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2)
        
        # 保存模型权重
        model_path = os.path.join(save_dir, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        print(f"模型配置和权重已保存到 {save_dir}")
    
    def load_weights(self, checkpoint_path):
        """
        从检查点加载权重
        """
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"检查点路径 {checkpoint_path} 不存在")
        
        # 加载权重
        if os.path.isdir(checkpoint_path):
            # 如果是目录，尝试加载 pytorch_model.bin
            model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location="cpu")
                self.load_state_dict(state_dict)
                print(f"从 {model_path} 加载权重成功")
                
                # 尝试加载配置文件
                config_path = os.path.join(checkpoint_path, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    print(f"从 {config_path} 加载配置成功")
            else:
                raise ValueError(f"在检查点目录 {checkpoint_path} 中找不到 pytorch_model.bin")
        else:
            # 直接加载检查点文件
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.load_state_dict(state_dict)
            print(f"从 {checkpoint_path} 加载权重成功")
    
    @staticmethod
    def from_pretrained(model_path):
        """
        从预训练模型创建实例
        """
        # 首先尝试加载配置
        config_path = os.path.join(model_path, "config.json") if os.path.isdir(model_path) else None
        if config_path and os.path.exists(config_path):
            config = config_path
        else:
            config = None
        
        # 创建模型实例
        model = QwenLLM(config=config)
        
        # 加载权重
        model.load_weights(model_path)
        
        return model
    
    def get_parameter_count(self):
        """
        计算模型的参数数量
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total": total_params,
            "trainable": trainable_params
        }
    
    def print_model_stats(self):
        """
        打印模型统计信息
        """
        param_counts = self.get_parameter_count()
        total_params = param_counts["total"]
        trainable_params = param_counts["trainable"]
        
        print(f"模型总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        
        # 显示各个部分的参数量
        embedding_params = sum(p.numel() for p in self.transformer.embedding.parameters())
        print(f"嵌入层参数量: {embedding_params:,}")
        
        # 计算每层的参数量
        layer_params = [sum(p.numel() for p in layer.parameters()) for layer in self.transformer.layers]
        avg_layer_params = sum(layer_params) / len(layer_params)
        print(f"平均每层参数量: {avg_layer_params:,.0f}")
        
        # 显示配置概要
        print("\n模型配置:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
            
    @torch.no_grad()
    def sample_and_print(self, prompt, tokenizer, max_length=100, temperature=0.7, top_p=0.9):
        """
        根据提示生成文本并打印
        """
        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.transformer.embedding.weight.device)
        
        # 生成文本
        output_ids = self.transformer.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
        
        # 解码输出
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # 打印结果
        print(f"输入:\n{prompt}")
        print(f"\n输出:\n{generated_text}")
        
        return generated_text 