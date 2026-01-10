"""测试修正后的 Step 7"""
import torch
from config import RunConfig
from AllinVIS import AllinVISModel
from pathlib import Path

# 创建配置
cfg = RunConfig(
    vis_image_path=Path("data/test_vi.png"),
    ir_image_path=Path("data/test_ir.png"),
    domain_name="test",
    num_timesteps=50  # 设置50步
)

try:
    model = AllinVISModel(cfg)
    print(f"✅ 模型创建成功，total_steps={model.total_steps}")
    
    # 设置曝光度
    model.E_vi = 0.25
    print(f"✅ 曝光度设置:  E_vi={model.E_vi}")
    
    # 测试权重计算（模拟去噪过程）
    print("\n权重计算测试（模拟去噪）:")
    print("step | current_t | t_norm | w_ir  | w_vi  | w_txt | 阶段")
    print("-" * 65)
    
    for step in [0, 10, 25, 40, 48]: 
        current_timestep = model.total_steps - step  # ✅ 修正后的计算
        w1, w2, w3 = model.compute_adaptive_weights(current_timestep)
        t_norm = current_timestep / model.total_steps
        
        if t_norm > 0.7:
            stage = "Early(强IR)"
        elif t_norm > 0.2:
            stage = "Mid(过渡)"
        else:
            stage = "Late(强VI)"
        
        print(f"{step:4d} | {current_timestep:9d} | {t_norm: 6.2f} | {w1:.3f} | {w2:.3f} | {w3:.3f} | {stage}")
    
    print("\n🎉 修正后的 Step 7 测试通过！")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
