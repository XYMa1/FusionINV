from typing import List, Optional, Callable

import torch
import torch.nn.functional as F

from config import RunConfig
from constants import OUT_INDEX, IR_INDEX, VIS_INDEX
from models.stable_diffusion import FusionINVAttentionStableDiffusionPipeline
from utils import attention_utils
from utils.fusion_utils import maskedfusionin, fusion_in, adain, fusiondetails_in, maskedadain
from utils.model_utils import get_stable_diffusion_model
from utils.segmentation import Segmentor


# ========== 新增：辅助函数 ==========
def adaptive_pool_1d(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    自适应池化：将序列长度调整到目标长度

    Args:
        x: [batch, seq_len, dim]
        target_len: 目标序列长度

    Returns:
        pooled: [batch, target_len, dim]
    """
    batch, seq_len, dim = x.shape

    if seq_len == target_len:
        return x

    # 使用插值
    x_permute = x.permute(0, 2, 1)  # [batch, dim, seq_len]
    pooled = torch.nn.functional.interpolate(
        x_permute,
        size=target_len,
        mode='linear',
        align_corners=False
    )
    pooled = pooled.permute(0, 2, 1)  # [batch, target_len, dim]

    return pooled


# ===================================
# ========== 新增：维度投影函数 ==========
def adaptive_dimension_projection(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    """
    自适应维度投影：将特征维度调整到目标维度

    Args:
        x: [seq_len, dim]
        target_dim: 目标特征维度

    Returns:
        projected: [seq_len, target_dim]
    """
    seq_len, current_dim = x.shape

    if current_dim == target_dim:
        return x

    # 降维：使用分组平均（比截断更平滑）
    if current_dim > target_dim:
        group_size = current_dim // target_dim
        remainder = current_dim % target_dim

        if remainder == 0:
            # 完美整除：分组平均
            projected = x.reshape(seq_len, target_dim, group_size).mean(dim=2)
        else:
            # 不能整除：使用插值
            projected = torch.nn.functional.interpolate(
                x.unsqueeze(0).permute(0, 2, 1),  # [1, current_dim, seq_len]
                size=target_dim,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)[0]  # [seq_len, target_dim]

    # 升维：使用线性插值
    else:
        projected = torch.nn.functional.interpolate(
            x.unsqueeze(0).permute(0, 2, 1),  # [1, current_dim, seq_len]
            size=target_dim,
            mode='linear',
            align_corners=False
        ).permute(0, 2, 1)[0]  # [seq_len, target_dim]

    return projected

# ===================================


class AllinVISModel:

    def __init__(self, config: RunConfig, pipe: Optional[FusionINVAttentionStableDiffusionPipeline] = None):
        self.config = config
        self.pipe = get_stable_diffusion_model() if pipe is None else pipe
        self.register_attention_control()
        self.segmentor = Segmentor(prompt=config.prompt, object_nouns=[config.object_noun])
        self.latents_vis, self.latents_ir = None, None
        self.zs_vis, self.zs_ir = None, None
        self.image_vis_mask_32, self.image_vis_mask_64 = None, None
        self.image_ir_mask_32, self.image_ir_mask_64 = None, None
        self.enable_edit = False
        self.step = 0

        # ========== 新增：LIT-Fusion 相关参数 ==========
        self.E_vi = 0.5  # 曝光度，默认值（会在 inversion 后更新）
        self.total_steps = config.num_timesteps  # ✅ 修正：从配置读取
        self.cached_text_K = None  # 缓存的文本 K
        self.cached_text_V = None  # 缓存的文本 V
        # =============================================

    def set_latents(self, latents_vis: torch.Tensor, latents_ir: torch.Tensor):
        self.latents_vis = latents_vis
        self.latents_ir = latents_ir

    def set_noise(self, zs_vis: torch.Tensor, zs_ir: torch.Tensor):
        self.zs_vis = zs_vis
        self.zs_ir = zs_ir

    def set_masks(self, masks: List[torch.Tensor]):
        self.image_vis_mask_32, self.image_ir_mask_32, self.image_vis_mask_64, self.image_ir_mask_64 = masks

    def compute_adaptive_weights(self, t: int) -> tuple:
        """
        计算自适应三流权重（线性版本 - MVP）

        Args:
            t: 当前时间步（从 total_steps 递减到 0）

        Returns:
            (w1, w2, w3): IR权重, VI权重, Text权重
        """
        # 归一化时间步 [1. 0 → 0.0]
        t_norm = t / self.total_steps

        # 根据时间阶段设置基础权重
        if t_norm > 0.7:  # Early:  T → 0.7T (强结构)
            w1_base = 0.7
            w2_base = 0.2
            w3_base = 0.1
        elif t_norm > 0.2:  # Mid: 0.7T → 0.2T (平衡过渡)
            # 线性插值
            alpha = (t_norm - 0.2) / 0.5  # 映射到 [1, 0]
            w1_base = 0.7 * alpha + 0.3 * (1 - alpha)
            w2_base = 0.2 * alpha + 0.6 * (1 - alpha)
            w3_base = 0.1
        else:  # Late: 0.2T → 0 (强语义)
            w1_base = 0.3
            w2_base = 0.6
            w3_base = 0.1

        # 基于曝光度调制
        E_vi = self.E_vi

        # VI 权重：低光时降低（因为信息少）
        w2 = w2_base * E_vi

        # Text 权重：低光时增强（补偿信息缺失）
        w3 = w3_base * (1 + 2 * (1 - E_vi))

        # IR 权重：保证归一化
        w1 = 1.0 - w2 - w3
        w1 = max(0.1, w1)  # 防止过小

        # 重新归一化（确保和为1）
        total = w1 + w2 + w3
        w1, w2, w3 = w1 / total, w2 / total, w3 / total

        return w1, w2, w3

    def register_attention_control(self):
        """注册三流混合注意力处理器"""

        model_self = self

        class AttentionProcessor:
            """三流混合注意力处理器���GAP版本）"""

            def __init__(self, place_in_unet: str):
                self.place_in_unet = place_in_unet
                self.cached_text_K = None
                self.cached_text_V = None

            def __call__(self,
                         attn,
                         hidden_states: torch.Tensor,
                         encoder_hidden_states: Optional[torch.Tensor] = None,
                         attention_mask=None,
                         temb=None,
                         perform_swap: bool = False):

                residual = hidden_states

                # 判断是否是 Cross-Attention
                is_cross = encoder_hidden_states is not None

                if attn.spatial_norm is not None:
                    hidden_states = attn.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )

                if attention_mask is not None:
                    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                    attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

                if attn.group_norm is not None:
                    hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = attn.to_q(hidden_states)

                # ========== Cross-Attention:  缓存文本特征 ==========
                if is_cross:
                    if attn.norm_cross:
                        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

                    key = attn.to_k(encoder_hidden_states)  # [B, 77, C]
                    value = attn.to_v(encoder_hidden_states)

                    # 只在 Decoder（up block）中缓存，节省显存
                    if "up" in self.place_in_unet and model_self.enable_edit:
                        self.cached_text_K = key.detach()
                        self.cached_text_V = value.detach()

                # ========== Self-Attention: 三流融合 ==========
                else:
                    encoder_hidden_states = hidden_states
                    key = attn.to_k(encoder_hidden_states)
                    value = attn.to_v(encoder_hidden_states)

                inner_dim = key.shape[-1]
                head_dim = inner_dim // attn.heads

                # ========== LIT-Fusion 三流融合机制 ==========
                should_mix = False

                # 判断是否应该执行融合（Self-Attention + Decoder + 启用编辑）
                if perform_swap and not is_cross and "up" in self.place_in_unet and model_self.enable_edit:
                    from utils import attention_utils
                    if attention_utils.should_mix_keys_and_values(model_self, hidden_states):
                        should_mix = True

                        # 【核心】计算自适应权重
                        current_timestep = model_self.total_steps - model_self.step
                        w1, w2, w3 = self.compute_adaptive_weights(
                            current_timestep,
                            model_self.total_steps,
                            model_self.E_vi
                        )

                        # 提取三个流的特征（在 reshape 之前）
                        from constants import IR_INDEX, VIS_INDEX, OUT_INDEX

                        K_ir = key[IR_INDEX]  # [seq_len, dim]
                        V_ir = value[IR_INDEX]
                        K_vi = key[VIS_INDEX]
                        V_vi = value[VIS_INDEX]

                        # 【GAP方案】提取文本流特征
                        if self.cached_text_K is not None:
                            try:
                                # 1. 全局平均池化 [B, 77, C] -> [B, 1, C]
                                text_k_pooled = torch.mean(self.cached_text_K, dim=1, keepdim=True)
                                text_v_pooled = torch.mean(self.cached_text_V, dim=1, keepdim=True)

                                # 2. 扩展到空间维度 [B, 1, C] -> [B, H*W, C]
                                spatial_len = K_ir.shape[0]
                                K_txt_batch = text_k_pooled.repeat(1, spatial_len, 1)
                                V_txt_batch = text_v_pooled.repeat(1, spatial_len, 1)

                                # 3. 提取单样本（用于OUT_INDEX）
                                K_txt = K_txt_batch[0]  # [H*W, C]
                                V_txt = V_txt_batch[0]

                            except Exception as e:
                                print(f"  [Warning] 文本特征处理失败: {e}")
                                K_txt = torch.zeros_like(K_ir)
                                V_txt = torch.zeros_like(V_ir)
                                w3 = 0.0
                                # 重新归一化
                                total = w1 + w2
                                w1, w2 = w1 / total, w2 / total
                        else:
                            K_txt = torch.zeros_like(K_ir)
                            V_txt = torch.zeros_like(V_ir)

                        # 【三流融合】
                        K_fused = w1 * K_ir + w2 * K_vi + w3 * K_txt
                        V_fused = w1 * V_ir + w2 * V_vi + w3 * V_txt

                        # 更新融合结果到输出索引
                        key[OUT_INDEX] = K_fused
                        value[OUT_INDEX] = V_fused

                        # 调试输出（每10步打印一次）
                        if model_self.step % 10 == 0:
                            print(f"  [Fusion] Step={model_self.step}/{model_self.total_steps}, "
                                  f"t={current_timestep}, E_vi={model_self.E_vi:. 3f}, "
                                  f"w_ir={w1:.2f}, w_vi={w2:.2f}, w_txt={w3:.2f}")
                # ============================================

                query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                # 计算 Attention
                from utils import attention_utils
                hidden_states, attn_weight = attention_utils.compute_scaled_dot_product_attention(
                    query, key, value,
                    edit_map=perform_swap and model_self.enable_edit and should_mix,
                    is_cross=is_cross,
                    contrast_strength=model_self.config.contrast_strength,
                    mask=attention_mask
                )

                # 更新 segmentation 的 attention map
                if model_self.config.use_masked_adain and model_self.step == model_self.config.adain_range.start - 1:
                    model_self.segmentor.update_attention(attn_weight, is_cross)

                hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                hidden_states = hidden_states.to(query.dtype)

                # linear proj
                hidden_states = attn.to_out[0](hidden_states)
                # dropout
                hidden_states = attn.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                if attn.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / attn.rescale_output_factor

                return hidden_states

            @staticmethod
            def compute_adaptive_weights(t, T, E_vi):
                """
                计算自适应三流权重（线性版本 - MVP）

                Args:
                    t:  当前时间步（从 T 递减到 0）
                    T: 总时间步数
                    E_vi: 曝光度 [0, 1]

                Returns:
                    (w1, w2, w3): IR权重, VI权重, Text权重
                """
                t_norm = t / T  # 归一化到 [1. 0 → 0.0]

                # 根据时间阶段设置基础权重
                if t_norm > 0.6:  # Early:  T → 0.6T (强结构)
                    w1_base = 0.8
                    w2_base = 0.1
                    w3_base = 0.1
                elif t_norm > 0.2:  # Mid: 0.6T → 0.2T (平衡过渡)
                    alpha = (t_norm - 0.2) / 0.4  # 映射到 [1, 0]
                    w1_base = 0.8 * alpha + 0.3 * (1 - alpha)
                    w2_base = 0.1 * alpha + 0.6 * (1 - alpha)
                    w3_base = 0.1
                else:  # Late: 0.2T → 0 (强语义)
                    w1_base = 0.3
                    w2_base = 0.6
                    w3_base = 0.1

                # 基于曝光度调制
                w2 = w2_base * E_vi  # VI权重：低光时降低
                w3 = w3_base * (1 + 2 * (1 - E_vi))  # Text权重：低光时增强
                w1 = 1.0 - w2 - w3  # IR权重：保证归一化

                # 防御性归一化
                total = w1 + w2 + w3
                w1, w2, w3 = w1 / total, w2 / total, w3 / total

                return w1, w2, w3

        # ========== 注册处理器到 U-Net ==========
        def register_recr(net_, count, place_in_unet):
            if net_.__class__.__name__ == 'Attention':
                net_.set_processor(AttentionProcessor(place_in_unet + f"_{count + 1}"))
                return count + 1
            elif hasattr(net_, 'children'):
                for net__ in net_.children():
                    count = register_recr(net__, count, place_in_unet)
            return count

        cross_att_count = 0
        sub_nets = self.pipe.unet.named_children()
        for net in sub_nets:
            if "down" in net[0]:
                cross_att_count += register_recr(net[1], 0, "down")
            elif "up" in net[0]:
                cross_att_count += register_recr(net[1], 0, "up")
            elif "mid" in net[0]:
                cross_att_count += register_recr(net[1], 0, "mid")
