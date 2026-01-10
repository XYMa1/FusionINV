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

    def get_adain_callback(self):

        def callback(st: int, timestep: int, latents: torch.FloatTensor) -> Callable:
            self.step = st
            # Compute the masks using prompt mixing self-segmentation and use the masks for AdaIN operation
            if self.config.use_masked_adain and self.step == self.config.adain_range.start:
                masks = self.segmentor.get_object_masks()
                self.set_masks(masks)
            # Apply AdaIN operation using the computed masks
            if self.config.adain_range.start <= self.step < self.config.adain_range.end:
                if self.config.use_masked_adain:
                    latents[0] = maskedadain(latents[0], latents[1], self.image_ir_mask_64, self.image_vis_mask_64)
                else:
                    latents[0] = adain(latents[0], latents[1])

        return callback

    def register_attention_control(self):

        model_self = self

        class AttentionProcessor:

            def __init__(self, place_in_unet: str):
                self.place_in_unet = place_in_unet
                if not hasattr(F, "scaled_dot_product_attention"):
                    raise ImportError("AttnProcessor2_0 requires torch 2.0, to use it, please upgrade torch to 2.0.")

                # ========== 新增：用于识别 Cross-Attention 层 ==========
                self.is_cross_attn_layer = False  # 标记是否是 Cross-Attention 层
                # ====================================================

            def __call__(self,
                         attn,
                         hidden_states: torch.Tensor,
                         encoder_hidden_states: Optional[torch.Tensor] = None,
                         attention_mask=None,
                         temb=None,
                         perform_swap: bool = False):

                residual = hidden_states

                # ========== 新增：判断是否是 Cross-Attention ==========
                is_cross = encoder_hidden_states is not None

                # 如果是 Cross-Attention，标记当前层
                if is_cross and model_self.enable_edit:
                    self.is_cross_attn_layer = True
                # ==================================================

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

                # is_cross 已经在前面定义了
                if not is_cross:
                    encoder_hidden_states = hidden_states
                elif attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)

                inner_dim = key.shape[-1]
                head_dim = inner_dim // attn.heads

                # ========== 新增：在 Cross-Attention 中缓存文本特征 ==========
                if is_cross and model_self.enable_edit and "up" in self.place_in_unet:
                    # 缓存文本的 K 和 V（只取第一个样本，即 fusion 的文本）
                    model_self.cached_text_K = key[OUT_INDEX: OUT_INDEX + 1].detach()  # [1, seq_len, dim]
                    model_self.cached_text_V = value[OUT_INDEX: OUT_INDEX + 1].detach()
                # =============================================================

                # ========== LIT-Fusion 三流融合机制 ==========
                should_mix = False
                config_contrast_strength = model_self.config.contrast_strength

                # 判断是否应该执行融合（Self-Attention + Decoder）
                if perform_swap and not is_cross and "up" in self.place_in_unet and model_self.enable_edit:
                    if attention_utils.should_mix_keys_and_values(model_self, hidden_states):
                        should_mix = True

                        # 【核心】计算自适应权重
                        # ✅ 修正：将递增的 step 转换为递减的 timestep
                        current_timestep = model_self.total_steps - model_self.step
                        w1, w2, w3 = model_self.compute_adaptive_weights(current_timestep)

                        # 提取三个流的特征（在 reshape 之前）
                        K_ir = key[IR_INDEX]  # [seq_len, dim]
                        V_ir = value[IR_INDEX]
                        K_vi = key[VIS_INDEX]
                        V_vi = value[VIS_INDEX]

                        # 获取文本流特征
                        if model_self.cached_text_K is not None:
                            # 调整文本特征的序列长度到当前空间分辨率
                            spatial_len = hidden_states.shape[1]  # 如 64x64 = 4096

                            K_txt = adaptive_pool_1d(
                                model_self.cached_text_K.to(key.dtype),
                                target_len=spatial_len
                            )[0]  # [spatial_len, dim]

                            V_txt = adaptive_pool_1d(
                                model_self.cached_text_V.to(value.dtype),
                                target_len=spatial_len
                            )[0]  # [spatial_len, dim]
                        else:
                            # 如果还没缓存文本特征，使用零向量
                            K_txt = torch.zeros_like(K_ir)
                            V_txt = torch.zeros_like(V_ir)

                        # 【三流融合】
                        K_fused = w1 * K_ir + w2 * K_vi + w3 * K_txt
                        V_fused = w1 * V_ir + w2 * V_vi + w3 * V_txt

                        # 更新融合结果到输出索引
                        key[OUT_INDEX] = K_fused
                        value[OUT_INDEX] = V_fused

                        # ✅ 优化：增强调试输出（每10步打印一次）
                        if model_self.step % 10 == 0:
                            t_current = model_self.total_steps - model_self.step
                            print(f"  [Fusion] Step={model_self.step}/{model_self.total_steps}, "
                                  f"t={t_current}, E_vi={model_self.E_vi:.3f}, "
                                  f"w_ir={w1:.2f}, w_vi={w2:.2f}, w_txt={w3:.2f}")
                # ============================================

                query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                # Compute the cross attention and apply our contrasting operation
                hidden_states, attn_weight = attention_utils.compute_scaled_dot_product_attention(
                    query, key, value,
                    edit_map=perform_swap and model_self.enable_edit and should_mix,
                    is_cross=is_cross,
                    contrast_strength=config_contrast_strength,
                    mask=attention_mask
                )

                # Update attention map for segmentation
                if model_self.config.use_masked_adain and model_self.step == model_self.config.adain_range.start - 1:
                    model_self.segmentor.update_attention(attn_weight, is_cross)

                hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                hidden_states = hidden_states.to(query[OUT_INDEX].dtype)

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

        def register_recr(net_, count, place_in_unet):
            if net_.__class__.__name__ == 'ResnetBlock2D':
                pass
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
