from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, Optional


class Range(NamedTuple):
    start: int
    end:  int


@dataclass
class RunConfig:
    # Visible image path
    vis_image_path: Path = Path('./data/in_vis.png')
    # Infrared image path
    ir_image_path: Path = Path('./data/in_ir.png')
    # Domain name (e.g., buildings, animals)
    domain_name: Optional[str] = None
    # Output path
    output_path: Path = Path('./output')
    # Random seed
    seed: int = 41
    # Input prompt for inversion (will use domain name as default)
    prompt: Optional[str] = None
    # Number of timesteps
    num_timesteps: int = 100
    # Whether to use a binary mask for performing AdaIN
    use_masked_adain: bool = False  # ✅ 改为 False（LIT-Fusion 不需要）
    # Timesteps to apply cross-attention on 64x64 layers
    cross_attn_64_range: Range = Range(start=1, end=90)
    # Timesteps to apply cross-attention on 32x32 layers
    cross_attn_32_range: Range = Range(start=1, end=90)
    # Timesteps to apply AdaIn
    adain_range: Range = Range(start=90, end=90)  # 保持原样（几乎不用）
    # Swap guidance scale
    swap_guidance_scale: float = 3.5
    # Attention contrasting strength
    contrast_strength:  float = 1.67
    # Object nouns to use for self-segmentation (will use the domain name as default)
    object_noun: Optional[str] = None
    # Whether to load previously saved inverted latent codes
    load_latents: bool = True
    # Number of steps to skip in the denoising process (used value from original edit-friendly DDPM paper)
    skip_steps: int = 32

    direction_step_size: float = -0.12
    
    # ========== 新增：LIT-Fusion 参数 ==========
    E_vi: float = 0.5  # 曝光度（会在 inversion 后自动更新）
    # =========================================

    def __post_init__(self):
        save_name = f'vis={self.vis_image_path. stem}---ir={self.ir_image_path.stem}'

        # 处理 domain_name 为 None 的情况
        if self.domain_name is None:
            self.domain_name = "general"

        self.output_path = self.output_path / self.domain_name / save_name
        self.output_path. mkdir(parents=True, exist_ok=True)

        # Handle the domain name, prompt, and object nouns used for masking, etc.
        # ✅ 修改：use_masked_adain 改为 False 时也允许运行
        if self.prompt is None:
            if self.domain_name:
                self.prompt = f"A photo of a {self.domain_name}"
            else:
                self.prompt = ""
        
        if self.object_noun is None and self.domain_name:
            self.object_noun = self. domain_name

        # Define the paths to store the inverted latents to
        self.latents_path = Path(self.output_path) / "latents"
        self. latents_path.mkdir(parents=True, exist_ok=True)
        self.vis_latent_save_path = self.latents_path / f"{self.vis_image_path.stem}_vis.pt"
        self.ir_latent_save_path = self.latents_path / f"{self.ir_image_path.stem}_ir.pt"
