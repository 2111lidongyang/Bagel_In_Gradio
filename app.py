import gradio as gr
from PIL import Image

import os
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from modeling.bagel import BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from data.data_utils import pil_img2rgb, add_special_tokens
from data.transforms import ImageTransform
# 假设inferencer和相关模型已经正确初始化
from inferencer import InterleaveInferencer


# from inference.ipynb中的初始化代码

def initialize_model(
        model_path: str = "/openbayes/input/input0/BAGEL-7B-MoT",
        max_mem_per_gpu: str = "40GiB",
        offload_folder: str = "./offload",
        torch_dtype=torch.bfloat16
):
    # 确保offload文件夹存在
    os.makedirs(offload_folder, exist_ok=True)

    # 初始化LLM配置
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    # 初始化ViT配置
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    # 加载VAE
    vae_model, vae_config = load_ae(os.path.join(model_path, "ae.safetensors"))

    # 构建Bagel配置
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    # 空权重初始化
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    # 设备映射
    device_map = infer_auto_device_map(
        model,
        max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    # 关键模块设备对齐
    same_device_modules = [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed',
        'vae2llm',
        'llm2vae',
        'connector',
        'vit_pos_embed'
    ]
    first_device = device_map.get(same_device_modules[0], "cuda:0" if torch.cuda.is_available() else "cpu")
    for k in same_device_modules:
        device_map[k] = first_device

    # 加载检查点
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_path, "ema.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        dtype=torch_dtype,
    )
    model = model.eval()
    print('Model loaded')
    # 初始化tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # 图像变换
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids
    )
    return inferencer
    # return {
    #     "model": model.eval(),
    #     "vae_model": vae_model,
    #     "tokenizer": tokenizer,
    #     "vae_transform": vae_transform,
    #     "vit_transform": vit_transform,
    #     "new_token_ids": new_token_ids
    # }


def create_interface(inferencer):
    # 图像生成功能函数
    def generate_image(text, cfg_text, num_steps, t_shift):
        inference_hyper = dict(
            cfg_text_scale=cfg_text,
            cfg_img_scale=1.0,
            cfg_interval=[0.4, 1.0],
            timestep_shift=t_shift,
            num_timesteps=int(num_steps),
            cfg_renorm_min=1.0,
            cfg_renorm_type="global",
        )
        output = inferencer(
            text=text,
            **inference_hyper
        )
        return output['image']

    # 带思考的图像生成
    def generate_with_think(text, max_think, cfg_text, num_steps, t_shift):
        inference_hyper = dict(
            max_think_token_n=int(max_think),
            do_sample=False,
            # text_temperature=0.3,
            cfg_text_scale=cfg_text,
            cfg_img_scale=1.0,
            cfg_interval=[0.4, 1.0],
            timestep_shift=t_shift,
            num_timesteps=int(num_steps),
            cfg_renorm_min=1.0,
            cfg_renorm_type="global",
        )

        output = inferencer(
            text=text,
            think=True,
            **inference_hyper
        )
        return output['image'], output.get('text', "")

    # 图像编辑功能
    def edit_image(image, text, cfg_text, cfg_img, num_steps, t_shift):
        inference_hyper = dict(
            cfg_text_scale=cfg_text,
            cfg_img_scale=cfg_img,
            cfg_interval=[0.0, 1.0],
            timestep_shift=t_shift,
            num_timesteps=int(num_steps),
            cfg_renorm_min=1.0,
            cfg_renorm_type="text_channel",
        )
        output = inferencer(
            image=image,
            text=text,
            **inference_hyper
        )
        return output['image']

    # 带思考的图像编辑
    def edit_with_think(image, text, max_think, cfg_text, cfg_img, num_steps, t_shift):
        inference_hyper = dict(
            max_think_token_n=int(max_think),
            do_sample=False,
            # text_temperature=0.3,
            cfg_text_scale=cfg_text,
            cfg_img_scale=cfg_img,
            cfg_interval=[0.4, 1.0],
            timestep_shift=t_shift,
            num_timesteps=int(num_steps),
            cfg_renorm_min=0.0,
            cfg_renorm_type="text_channel",
        )
        output = inferencer(
            image=image,
            text=text,
            think=True,
            **inference_hyper
        )
        return output['image'], output.get('text', "")

    # 图像理解功能
    def understand_image(image, text):
        inference_hyper = dict(
            max_think_token_n=1000,
            do_sample=False,
            # text_temperature=0.3,
        )
        output = inferencer(
            image=image,
            text=text,
            understanding_output=True,
            **inference_hyper
        )
        return output.get('text', "")

    with gr.Blocks(title="BAGEL-7B-MoT Demo") as demo:
        gr.Markdown("# 🥯 BAGEL-7B-MoT Multimodal Demo")

        # ========== 图像生成标签页 ==========
        with gr.Tab("Image Generation"):
            with gr.Row():
                # 输入列
                with gr.Column():
                    text_input = gr.Textbox(label="Prompt",
                                            lines=3,
                                            placeholder="Enter your image description here...")
                    generate_btn = gr.Button("Generate", variant="primary")

                # 输出列
                with gr.Column():
                    image_output = gr.Image(label="Generated Image",
                                            height=512)

            # 参数控制行
            with gr.Row():
                cfg_text_gen = gr.Slider(0.1, 10.0, value=4.0,
                                         label="Text Guidance Scale",
                                         info="Controls text influence (higher = more strict)")
                num_steps_gen = gr.Slider(10, 100, value=50, step=1,
                                          label="Generation Steps",
                                          info="More steps = better quality but slower")
                t_shift_gen = gr.Slider(0.0, 10.0, value=3.0,
                                        label="Timestep Shift",
                                        info="Controls generation progression")

            generate_btn.click(
                generate_image,
                inputs=[text_input, cfg_text_gen, num_steps_gen, t_shift_gen],
                outputs=image_output
            )

        # ========== 带思考的图像生成 ==========
        with gr.Tab("Image Generation with Think"):
            with gr.Row():
                with gr.Column():
                    text_input_think = gr.Textbox(label="Creative Prompt",
                                                  lines=3)
                    think_slider = gr.Slider(100, 2000, 1000,
                                             label="Max Thinking Tokens",
                                             info="Controls reasoning depth")
                    generate_think_btn = gr.Button("Generate with Reasoning", variant="primary")
                with gr.Column():
                    image_output_think = gr.Image(label="Generated Image", height=512)
                    text_output_think = gr.Textbox(label="Reasoning Process",
                                                   interactive=False)

            with gr.Row():
                cfg_text_think = gr.Slider(0.1, 10.0, value=4.0,
                                           label="Text Guidance Scale")
                num_steps_think = gr.Slider(10, 100, value=50, step=1,
                                            label="Generation Steps")
                t_shift_think = gr.Slider(0.0, 10.0, value=3.0,
                                          label="Timestep Shift")

            generate_think_btn.click(
                generate_with_think,
                inputs=[text_input_think, think_slider, cfg_text_think, num_steps_think, t_shift_think],
                outputs=[image_output_think, text_output_think]
            )

        # ========== 图像编辑标签页 ==========
        with gr.Tab("Image Editing"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil",
                                           label="Upload Source Image",
                                           height=300)
                    edit_text = gr.Textbox(label="Edit Instruction",
                                           lines=2,
                                           placeholder="Describe your edit...")
                    edit_btn = gr.Button("Apply Edit", variant="primary")
                with gr.Column():
                    edited_image = gr.Image(label="Edited Result",
                                            height=512)

            with gr.Row():
                cfg_text_edit = gr.Slider(0.1, 10.0, value=4.0,
                                          label="Text Guidance Strength")
                cfg_img_edit = gr.Slider(0.1, 10.0, value=1.0,
                                         label="Image Fidelity",
                                         info="Higher = preserve original more")
                num_steps_edit = gr.Slider(10, 100, value=50, step=1,
                                           label="Editing Steps")
                t_shift_edit = gr.Slider(0.0, 10.0, value=3.0,
                                         label="Edit Progression")

            edit_btn.click(
                edit_image,
                inputs=[image_input, edit_text, cfg_text_edit, cfg_img_edit, num_steps_edit, t_shift_edit],
                outputs=edited_image
            )

        # ========== 带思考的图像编辑 ==========
        with gr.Tab("Smart Editing"):
            with gr.Row():
                with gr.Column():
                    image_input_smart = gr.Image(type="pil",
                                                 label="Upload Image",
                                                 height=300)
                    edit_text_smart = gr.Textbox(label="Edit Request",
                                                 lines=2,
                                                 placeholder="What change would you like?")
                    think_slider_edit = gr.Slider(100, 2000, 1000,
                                                  label="Reasoning Depth")
                    smart_edit_btn = gr.Button("Smart Edit", variant="primary")
                with gr.Column():
                    edited_image_smart = gr.Image(label="Edited Result",
                                                  height=512)
                    reasoning_output = gr.Textbox(label="Editing Plan",
                                                  interactive=False)

            with gr.Row():
                cfg_text_smart = gr.Slider(0.1, 10.0, value=4.0,
                                           label="Text Guidance")
                cfg_img_smart = gr.Slider(0.1, 10.0, value=1.0,
                                          label="Original Preservation")
                num_steps_smart = gr.Slider(10, 100, value=50, step=1,
                                            label="Processing Steps")
                t_shift_smart = gr.Slider(0.0, 10.0, value=3.0,
                                          label="Edit Progression")

            smart_edit_btn.click(
                edit_with_think,
                inputs=[
                    image_input_smart,
                    edit_text_smart,
                    think_slider_edit,
                    cfg_text_smart,
                    cfg_img_smart,
                    num_steps_smart,
                    t_shift_smart
                ],
                outputs=[edited_image_smart, reasoning_output]
            )

        # ========== 图像理解标签页 ==========
        with gr.Tab("Image Understanding"):
            with gr.Row():
                with gr.Column():
                    uploaded_image = gr.Image(type="pil",
                                              label="Upload Image",
                                              height=300)
                    question_input = gr.Textbox(label="Your Question",
                                                lines=2,
                                                placeholder="Ask about the image...")
                    understand_btn = gr.Button("Analyze", variant="primary")
                with gr.Column():
                    answer_output = gr.Textbox(label="Analysis Result",
                                               lines=5,
                                               interactive=False)

            # with gr.Row():
            #     cfg_text_understand = gr.Slider(0.1, 10.0, value=4.0,
            #                                  label="Reasoning Focus")
            #     num_steps_understand = gr.Slider(10, 100, value=50, step=1,
            #                                   label="Processing Steps")
            #     t_shift_understand = gr.Slider(0.0, 10.0, value=3.0,
            #                                 label="Analysis Depth")

            understand_btn.click(
                understand_image,
                inputs=[
                    uploaded_image,
                    question_input
                ],
                outputs=answer_output
            )

    return demo


if __name__ == "__main__":
    # 初始化inferencer（需确保模型已正确加载）
    model_components = initialize_model()
    print("Model initialized successfully!")
    demo = create_interface(model_components)
    demo.launch(share=False, server_name='0.0.0.0', server_port=8080)
