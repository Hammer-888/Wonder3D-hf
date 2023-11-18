import os
import torch
from rembg import remove
import torch.nn as nn
from einops import rearrange
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel
from mvdiffusion.data.single_image_dataset import (
    SingleImageDataset as MVDiffusionDataset,
)
from mvdiffusion.pipelines.pipeline_mvdiffusion_image import MVDiffusionImagePipeline
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from common import (
    get_config,
    save_image,
    save_image_numpy,
    save_image_to_disk,
    preprocess_data,
    expand2square,
    tensor2pil,
)


class Wonder3d(nn.Module):
    def __init__(
        self,
        device,
    ):
        super(Wonder3d, self).__init__()
        self.device = device
        self.fp16 = torch.float16
        self.cfg = get_config()
        self.pipe = self.init_pipeline()
        torch.set_grad_enabled(False)
        self.pipe.to(self.device)
        self.emmbedding = None

    def init_pipeline(self):
        """
        Initializes and returns a pipeline object for multi-modal diffusion image processing.

        Args:
            cfg (Config): The configuration object containing all the necessary parameters.
            weight_dtype (torch.dtype): The data type to which the weights of the models should be cast.

        Returns:
            MVDiffusionImagePipeline: The pipeline object for multi-modal diffusion image processing.
        """
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="image_encoder",
            revision=self.cfg.revision,
        )
        feature_extractor = CLIPImageProcessor.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="feature_extractor",
            revision=self.cfg.revision,
        )
        vae = AutoencoderKL.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="vae",
            revision=self.cfg.revision,
        )
        unet = UNetMV2DConditionModel.from_pretrained_2d(
            self.cfg.pretrained_unet_path,
            subfolder="unet",
            revision=self.cfg.revision,
            **self.cfg.unet_from_pretrained_kwargs,
        )
        unet.enable_xformers_memory_efficient_attention()

        # Move text_encode and vae to gpu and cast to weight_dtype
        image_encoder.to(dtype=self.fp16)
        vae.to(dtype=self.fp16)
        unet.to(dtype=self.fp16)

        pipeline = MVDiffusionImagePipeline(
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            vae=vae,
            unet=unet,
            safety_checker=None,
            scheduler=DDIMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path, subfolder="scheduler"
            ),
            **self.cfg.pipe_kwargs,
        )
        return pipeline

    def get_img_embeds(self, image: torch.Tensor):
        # x: image tensor in [0, 1]
        image = tensor2pil(image)
        self.emmbedding = image

    def refine(
        self,
        single_image,
        guidance_scale,
        steps,
        seed,
        crop_size,
        write_image=False,
    ):
        """
        Runs the pipeline to process a single image.

        Args:
            single_image (Tensor): The input image to be processed.
            guidance_scale (float): The scale factor for the guidance.
            steps (int): The number of inference steps to perform.
            seed (int): The seed for random number generation.
            crop_size (int): The size of the image crop.
            chk_group (Optional[str]): The checkpoint group.

        Returns:
            List[Image]: A list of processed images.
        """
        # if chk_group is not None:
        #     write_image = "Write Results" in chk_group

        batch = preprocess_data(self.emmbedding, crop_size)

        self.pipe.set_progress_bar_config(disable=True)
        seed = int(seed)
        generator = torch.Generator(device=self.pipe.unet.device).manual_seed(seed)

        # repeat  (2B, Nv, 3, H, W)
        imgs_in = torch.cat([batch["imgs_in"]] * 2, dim=0).to(self.fp16)

        # (2B, Nv, Nce)
        camera_embeddings = torch.cat([batch["camera_embeddings"]] * 2, dim=0).to(
            self.fp16
        )

        task_embeddings = torch.cat(
            [batch["normal_task_embeddings"], batch["color_task_embeddings"]], dim=0
        ).to(self.fp16)

        camera_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1).to(
            self.fp16
        )

        # (B*Nv, 3, H, W)
        imgs_in = rearrange(imgs_in, "Nv C H W -> (Nv) C H W")
        # (B*Nv, Nce)
        # camera_embeddings = rearrange(camera_embeddings, "B Nv Nce -> (B Nv) Nce")

        out = self.pipe(
            imgs_in,
            camera_embeddings,
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            output_type="pt",
            num_images_per_prompt=1,
            **self.cfg.pipe_validation_kwargs,
        ).images

        bsz = out.shape[0] // 2
        normals_pred = out[:bsz]
        images_pred = out[bsz:]
        num_views = 6
        # if write_image:
        #     VIEWS = ["front", "front_right", "right", "back", "left", "front_left"]
        #     cur_dir = os.path.join(
        #         "./outputs", f"cropsize-{crop_size}-cfg{guidance_scale:.1f}"
        #     )

        #     scene = "scene"
        #     scene_dir = os.path.join(cur_dir, scene)
        #     normal_dir = os.path.join(scene_dir, "normals")
        #     masked_colors_dir = os.path.join(scene_dir, "masked_colors")
        #     os.makedirs(normal_dir, exist_ok=True)
        #     os.makedirs(masked_colors_dir, exist_ok=True)
        #     for j in range(num_views):
        #         view = VIEWS[j]
        #         normal = normals_pred[j]
        #         color = images_pred[j]

        #         normal_filename = f"normals_000_{view}.png"
        #         rgb_filename = f"rgb_000_{view}.png"
        #         normal = save_image_to_disk(
        #             normal, os.path.join(normal_dir, normal_filename)
        #         )
        #         color = save_image_to_disk(color, os.path.join(scene_dir, rgb_filename))

        #         rm_normal = remove(normal)
        #         rm_color = remove(color)

        #         save_image_numpy(rm_normal, os.path.join(scene_dir, normal_filename))
        #         save_image_numpy(
        #             rm_color, os.path.join(masked_colors_dir, rgb_filename)
        #         )

        # normals_pred = [save_image(normals_pred[i]) for i in range(bsz)]
        # images_pred = [save_image(images_pred[i]) for i in range(bsz)]

        out = images_pred[0] + normals_pred[0]
        output = out.unsqueeze(0)
        return out


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="input image path")
    parser.add_argument(
        "--guidance_scale",
        type=int,
        default=3,
        help="Classifier Free Guidance Scale, 1-5",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of Diffusion Inference Steps,15-100",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--crop_size", type=int, default=-1, help="Crop Size")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")

    opt = parser.parse_args()

    print(f"[INFO] init model ...")
    wonder3d = Wonder3d(opt.device)  # 初始化Wonder3d模型

    print(f"[INFO] loading image from {opt.input} ...")
    image = cv2.imread(opt.input, cv2.IMREAD_UNCHANGED)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    image = (
        torch.from_numpy(image)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
        .to(opt.device)
    )

    wonder3d.get_img_embeds(image)

    # input_image = Image.open(opt.input)
    # input_image = expand2square(input_image, (127, 127, 127, 0))
    # print(f"[INFO] loading image from {opt.input} ...")
    # output = wonder3d.run_pipeline(  # 推理输入的图像
    #     input_image,
    #     opt.guidance_scale,
    #     opt.steps,
    #     opt.seed,
    #     opt.crop_size,
    #     write_image=True,
    # )
    # torch.cuda.empty_cache()
    output = wonder3d.refine(
        image, opt.guidance_scale, opt.steps, opt.seed, opt.crop_size
    )
    torch.cuda.empty_cache()
    plt.imshow(output.float().cpu().numpy().transpose(0, 2, 3, 1))
    plt.show()
