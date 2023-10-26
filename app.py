import os
import sys
import numpy
import torch
import rembg
import threading
import urllib.request
from PIL import Image
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import streamlit as st
import huggingface_hub
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel
from mvdiffusion.data.single_image_dataset import SingleImageDataset as MVDiffusionDataset
from mvdiffusion.pipelines.pipeline_mvdiffusion_image import MVDiffusionImagePipeline
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from einops import rearrange

@dataclass
class TestConfig:
    pretrained_model_name_or_path: str
    pretrained_unet_path:str
    revision: Optional[str]
    validation_dataset: Dict
    save_dir: str
    seed: Optional[int]
    validation_batch_size: int
    dataloader_num_workers: int

    local_rank: int

    pipe_kwargs: Dict
    pipe_validation_kwargs: Dict
    unet_from_pretrained_kwargs: Dict
    validation_guidance_scales: List[float]
    validation_grid_nrow: int
    camera_embedding_lr_mult: float

    num_views: int
    camera_embedding_type: str

    pred_type: str  # joint, or ablation

    enable_xformers_memory_efficient_attention: bool

    cond_on_normals: bool
    cond_on_colors: bool

img_example_counter = 0
iret_base = 'example_images'
iret = [
    dict(rimageinput=os.path.join(iret_base, x), dispi=os.path.join(iret_base, x))
    for x in sorted(os.listdir(iret_base))
]

def save_image(tensor):
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    # pdb.set_trace()
    im = Image.fromarray(ndarr)
    return ndarr

weight_dtype = torch.float16

class SAMAPI:
    predictor = None

    @staticmethod
    @st.cache_resource
    def get_instance(sam_checkpoint=None):
        if SAMAPI.predictor is None:
            if sam_checkpoint is None:
                sam_checkpoint = "./sam_pt/sam_vit_h_4b8939.pth"
            if not os.path.exists(sam_checkpoint):
                os.makedirs('sam_pt', exist_ok=True)
                urllib.request.urlretrieve(
                    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                    sam_checkpoint
                )
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model_type = "default"

            from segment_anything import sam_model_registry, SamPredictor

            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)

            predictor = SamPredictor(sam)
            SAMAPI.predictor = predictor
        return SAMAPI.predictor

    @staticmethod
    def segment_api(rgb, mask=None, bbox=None, sam_checkpoint=None):
        """

        Parameters
        ----------
        rgb : np.ndarray h,w,3 uint8
        mask: np.ndarray h,w bool

        Returns
        -------

        """
        np = numpy
        predictor = SAMAPI.get_instance(sam_checkpoint)
        predictor.set_image(rgb)
        if mask is None and bbox is None:
            box_input = None
        else:
            # mask to bbox
            if bbox is None:
                y1, y2, x1, x2 = np.nonzero(mask)[0].min(), np.nonzero(mask)[0].max(), np.nonzero(mask)[1].min(), \
                                 np.nonzero(mask)[1].max()
            else:
                x1, y1, x2, y2 = bbox
            box_input = np.array([[x1, y1, x2, y2]])
        masks, scores, logits = predictor.predict(
            box=box_input,
            multimask_output=True,
            return_logits=False,
        )
        mask = masks[-1]
        return mask


def image_examples(samples, ncols, return_key=None, example_text="Examples"):
    global img_example_counter
    trigger = False
    with st.expander(example_text, True):
        for i in range(len(samples) // ncols):
            cols = st.columns(ncols)
            for j in range(ncols):
                idx = i * ncols + j
                if idx >= len(samples):
                    continue
                entry = samples[idx]
                with cols[j]:
                    st.image(entry['dispi'])
                    img_example_counter += 1
                    with st.columns(5)[2]:
                        this_trigger = st.button('\+', key='imgexuse%d' % img_example_counter)
                    trigger = trigger or this_trigger
                    if this_trigger:
                        trigger = entry[return_key]
    return trigger


def segment_img(img: Image):
    output = rembg.remove(img)
    mask = numpy.array(output)[:, :, 3] > 0
    sam_mask = SAMAPI.segment_api(numpy.array(img)[:, :, :3], mask)
    segmented_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
    segmented_img.paste(img, mask=Image.fromarray(sam_mask))
    return segmented_img


def segment_6imgs(imgs):
    segmented_imgs = []
    for i, img in enumerate(imgs):
        output = rembg.remove(img)
        mask = numpy.array(output)[:, :, 3]
        mask = SAMAPI.segment_api(numpy.array(img)[:, :, :3], mask)
        data = numpy.array(img)[:,:,:3]
        data[mask == 0] = [255, 255, 255]
        segmented_imgs.append(data)
    result = numpy.concatenate([
        numpy.concatenate([segmented_imgs[0], segmented_imgs[1]], axis=1),
        numpy.concatenate([segmented_imgs[2], segmented_imgs[3]], axis=1),
        numpy.concatenate([segmented_imgs[4], segmented_imgs[5]], axis=1)
    ])
    return Image.fromarray(result)

def pack_6imgs(imgs):
    import pdb
    # pdb.set_trace()
    result = numpy.concatenate([
        numpy.concatenate([imgs[0], imgs[1]], axis=1),
        numpy.concatenate([imgs[2], imgs[3]], axis=1),
        numpy.concatenate([imgs[4], imgs[5]], axis=1)
    ])
    return Image.fromarray(result)


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


@st.cache_data
def check_dependencies():
    reqs = []
    try:
        import diffusers
    except ImportError:
        import traceback
        traceback.print_exc()
        print("Error: `diffusers` not found.", file=sys.stderr)
        reqs.append("diffusers==0.20.2")
    else:
        if not diffusers.__version__.startswith("0.20"):
            print(
                f"Warning: You are using an unsupported version of diffusers ({diffusers.__version__}), which may lead to performance issues.",
                file=sys.stderr
            )
            print("Recommended version is `diffusers==0.20.2`.", file=sys.stderr)
    try:
        import transformers
    except ImportError:
        import traceback
        traceback.print_exc()
        print("Error: `transformers` not found.", file=sys.stderr)
        reqs.append("transformers==4.29.2")
    if torch.__version__ < '2.0':
        try:
            import xformers
        except ImportError:
            print("Warning: You are using PyTorch 1.x without a working `xformers` installation.", file=sys.stderr)
            print("You may see a significant memory overhead when running the model.", file=sys.stderr)
    if len(reqs):
        print(f"Info: Fix all dependency errors with `pip install {' '.join(reqs)}`.")


@st.cache_resource
def load_wonder3d_pipeline():
    # Load scheduler, tokenizer and models.
    # noise_scheduler = DDPMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="image_encoder", revision=cfg.revision)
    feature_extractor = CLIPImageProcessor.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="feature_extractor", revision=cfg.revision)
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="vae", revision=cfg.revision)
    unet = UNetMV2DConditionModel.from_pretrained_2d(cfg.pretrained_unet_path, subfolder="unet", revision=cfg.revision, **cfg.unet_from_pretrained_kwargs)
    unet.enable_xformers_memory_efficient_attention()

    # Move text_encode and vae to gpu and cast to weight_dtype
    image_encoder.to(dtype=weight_dtype)
    vae.to(dtype=weight_dtype)
    unet.to(dtype=weight_dtype)

    pipeline = MVDiffusionImagePipeline(
        image_encoder=image_encoder, feature_extractor=feature_extractor, vae=vae, unet=unet, safety_checker=None,
        scheduler=DDIMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler"),
        **cfg.pipe_kwargs
    )

    if torch.cuda.is_available():
        pipeline.to('cuda:0')
    sys.main_lock = threading.Lock()
    return pipeline

from mvdiffusion.data.single_image_dataset import SingleImageDataset
def prepare_data(single_image):
    dataset = SingleImageDataset(
        root_dir = None,
        num_views = 6,
        img_wh=[256, 256],
        bg_color='white',
        crop_size=crop_size,
        single_image=single_image
    )
    return dataset[0]


def run_pipeline(pipeline, batch, guidance_scale, seed):

    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=pipeline.unet.device).manual_seed(seed)

    # repeat  (2B, Nv, 3, H, W)
    imgs_in = torch.cat([batch['imgs_in']]*2, dim=0).to(weight_dtype)
    
    # (2B, Nv, Nce)
    camera_embeddings = torch.cat([batch['camera_embeddings']]*2, dim=0).to(weight_dtype)

    task_embeddings = torch.cat([batch['normal_task_embeddings'], batch['color_task_embeddings']], dim=0).to(weight_dtype)

    camera_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1).to(weight_dtype)

    # (B*Nv, 3, H, W)
    imgs_in = rearrange(imgs_in, "Nv C H W -> (Nv) C H W")
    # (B*Nv, Nce)
    # camera_embeddings = rearrange(camera_embeddings, "B Nv Nce -> (B Nv) Nce")

    out = pipeline(
        imgs_in, camera_embeddings, generator=generator, guidance_scale=guidance_scale, 
        output_type='pt', num_images_per_prompt=1, **cfg.pipe_validation_kwargs
    ).images

    bsz = out.shape[0] // 2
    normals_pred = out[:bsz]
    images_pred = out[bsz:]

    normals_pred = [save_image(normals_pred[i]) for i in range(bsz)]
    images_pred = [save_image(images_pred[i]) for i in range(bsz)]

    return normals_pred, images_pred

from utils.misc import load_config    
from omegaconf import OmegaConf
# parse YAML config to OmegaConf
cfg = load_config("./configs/mvdiffusion-joint-ortho-6views.yaml")
# print(cfg)
schema = OmegaConf.structured(TestConfig)
# cfg = OmegaConf.load(args.config)
cfg = OmegaConf.merge(schema, cfg)

check_dependencies()
pipeline = load_wonder3d_pipeline()
SAMAPI.get_instance()
torch.set_grad_enabled(False)

st.title("Wonder3D: Single Image to 3D using Cross-Domain Diffusion")
# st.caption("For faster inference without waiting in queue, you may clone the space and run it yourself.")

pic = st.file_uploader("Upload an Image", key='imageinput', type=['png', 'jpg', 'webp'])
left, right = st.columns(2)
# with left:
#     rem_input_bg = st.checkbox("Remove Input Background")
# with right:
#     rem_output_bg = st.checkbox("Remove Output Background")
with left:
    num_inference_steps = st.slider("Number of Inference Steps", 15, 100, 50)
    # st.caption("Diffusion Steps. For general real or synthetic objects, around 28 is enough. For objects with delicate details such as faces (either realistic or illustration), you may need 75 or more steps.")
with right:
    cfg_scale = st.slider("Classifier Free Guidance Scale", 1.0, 10.0, 3.0)
with left:
    seed = int(st.text_input("Seed", "42"))
with right:
    crop_size = int(st.text_input("crop_size", "192"))
# submit = False
# if st.button("Submit"):
#     submit = True
submit = True
prog = st.progress(0.0, "Idle")
results_container = st.container()
sample_got = image_examples(iret, 4, 'rimageinput')
if sample_got:
    pic = sample_got
with results_container:
    if sample_got or pic is not None:
        prog.progress(0.03, "Waiting in Queue...")
        
        seed = int(seed)
        torch.manual_seed(seed)
        img = Image.open(pic)

        if max(img.size) > 1280:
            w, h = img.size
            w = round(1280 / max(img.size) * w)
            h = round(1280 / max(img.size) * h)
            img = img.resize((w, h))
        left, right = st.columns(2)
        with left:
            st.caption("Input Image")
            st.image(img)
        prog.progress(0.1, "Preparing Inputs")
        
        with right:
            img = segment_img(img)
            st.caption("Input (Background Removed)")
            st.image(img)
            
        img = expand2square(img, (127, 127, 127, 0))
        # pipeline.set_progress_bar_config(disable=True)
        prog.progress(0.3, "Run cross-domain diffusion model")
        data = prepare_data(img)
        normals_pred, images_pred = run_pipeline(pipeline, data, cfg_scale, seed)
        prog.progress(0.9, "finishing")
        left, right = st.columns(2)
        with left:
            st.caption("Generated Normals")
            st.image(pack_6imgs(normals_pred))
            
        with right:
            st.caption("Generated Color Images")
            st.image(pack_6imgs(images_pred))
        # if rem_output_bg:
        #     normals_pred = segment_6imgs(normals_pred)
        #     images_pred = segment_6imgs(images_pred)
        #     with right:
        #         st.image(normals_pred)
        #         st.image(images_pred)
        #         st.caption("Result (Background Removed)")
        prog.progress(1.0, "Idle")