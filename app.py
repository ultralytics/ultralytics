import gradio as gr
import numpy as np
import supervision as sv
import torch
from gradio_image_prompter import ImagePrompter
from huggingface_hub import hf_hub_download
from scipy.ndimage import binary_fill_holes

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
from ultralytics.utils.torch_utils import smart_inference_mode


def init_model(model_id, is_pf=False):
    filename = f"{model_id}-seg.pt" if not is_pf else f"{model_id}-seg-pf.pt"
    path = hf_hub_download(repo_id="jameslahm/yoloe", filename=filename)
    model = YOLOE(path)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model


@smart_inference_mode()
def yoloe_inference(image, prompts, target_image, model_id, image_size, conf_thresh, iou_thresh, prompt_type):
    model = init_model(model_id)
    kwargs = {}
    if prompt_type == "Text":
        texts = prompts["texts"]
        model.set_classes(texts, model.get_text_pe(texts))
    elif prompt_type == "Visual":
        kwargs = dict(prompts=prompts, predictor=YOLOEVPSegPredictor)
        if target_image:
            model.predict(source=image, imgsz=image_size, conf=conf_thresh, iou=iou_thresh, return_vpe=True, **kwargs)
            model.set_classes(["object0"], model.predictor.vpe)
            model.predictor = None  # unset VPPredictor
            image = target_image
            kwargs = {}
    elif prompt_type == "Prompt-free":
        vocab = model.get_vocab(prompts["texts"])
        model = init_model(model_id, is_pf=True)
        model.set_vocab(vocab, names=prompts["texts"])
        model.model.model[-1].is_fused = True
        model.model.model[-1].conf = 0.001
        model.model.model[-1].max_det = 1000

    results = model.predict(source=image, imgsz=image_size, conf=conf_thresh, iou=iou_thresh, **kwargs)
    detections = sv.Detections.from_ultralytics(results[0])

    resolution_wh = image.size
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence in zip(detections["class_name"], detections.confidence)
    ]

    annotated_image = image.copy()
    annotated_image = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX, opacity=0.4).annotate(
        scene=annotated_image, detections=detections
    )
    annotated_image = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX, thickness=thickness).annotate(
        scene=annotated_image, detections=detections
    )
    annotated_image = sv.LabelAnnotator(
        color_lookup=sv.ColorLookup.INDEX, text_scale=text_scale, smart_position=True
    ).annotate(scene=annotated_image, detections=detections, labels=labels)

    return annotated_image


def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    raw_image = gr.Image(type="pil", label="Image", visible=True, interactive=True)
                    box_image = ImagePrompter(type="pil", label="DrawBox", visible=False, interactive=True)
                    mask_image = gr.ImageEditor(
                        type="pil",
                        label="DrawMask",
                        visible=False,
                        interactive=True,
                        layers=False,
                        canvas_size=(640, 640),
                    )
                    target_image = gr.Image(type="pil", label="Target Image", visible=False, interactive=True)

                yoloe_infer = gr.Button(value="Detect & Segment Objects")
                prompt_type = gr.Textbox(value="Text", visible=False)

                with gr.Tab("Text") as text_tab:
                    texts = gr.Textbox(
                        label="Input Texts",
                        value="person,bus",
                        placeholder="person,bus",
                        visible=True,
                        interactive=True,
                    )

                with gr.Tab("Visual") as visual_tab:
                    with gr.Row():
                        visual_prompt_type = gr.Dropdown(
                            choices=["bboxes", "masks"], value="bboxes", label="Visual Type", interactive=True
                        )
                        visual_usage_type = gr.Radio(
                            choices=["Intra-Image", "Cross-Image"],
                            value="Intra-Image",
                            label="Intra/Cross Image",
                            interactive=True,
                        )

                with gr.Tab("Prompt-Free") as prompt_free_tab:
                    gr.HTML(
                        """
                        <p style='text-align: center'>
                        <b>Prompt-Free Mode is On</b>
                        </p>
                    """,
                        show_label=False,
                    )

                model_id = gr.Dropdown(
                    label="Model",
                    choices=[
                        "yoloe-v8s",
                        "yoloe-v8m",
                        "yoloe-v8l",
                        "yoloe-11s",
                        "yoloe-11m",
                        "yoloe-11l",
                    ],
                    value="yoloe-v8l",
                )
                image_size = gr.Slider(
                    label="Image Size",
                    minimum=320,
                    maximum=1280,
                    step=32,
                    value=640,
                )
                conf_thresh = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.25,
                )
                iou_thresh = gr.Slider(
                    label="IoU Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.70,
                )

            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated Image", visible=True)

        def update_text_image_visibility():
            return (
                gr.update(value="Text"),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )

        def update_visual_image_visiblity(visual_prompt_type, visual_usage_type):
            if visual_prompt_type == "bboxes":
                return (
                    gr.update(value="Visual"),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=(visual_usage_type == "Cross-Image")),
                )
            elif visual_prompt_type == "masks":
                return (
                    gr.update(value="Visual"),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=(visual_usage_type == "Cross-Image")),
                )

        def update_pf_image_visibility():
            return (
                gr.update(value="Prompt-free"),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )

        text_tab.select(
            fn=update_text_image_visibility,
            inputs=None,
            outputs=[prompt_type, raw_image, box_image, mask_image, target_image],
        )

        visual_tab.select(
            fn=update_visual_image_visiblity,
            inputs=[visual_prompt_type, visual_usage_type],
            outputs=[prompt_type, raw_image, box_image, mask_image, target_image],
        )

        prompt_free_tab.select(
            fn=update_pf_image_visibility,
            inputs=None,
            outputs=[prompt_type, raw_image, box_image, mask_image, target_image],
        )

        def update_visual_prompt_type(visual_prompt_type):
            if visual_prompt_type == "bboxes":
                return gr.update(visible=True), gr.update(visible=False)
            if visual_prompt_type == "masks":
                return gr.update(visible=False), gr.update(visible=True)
            return gr.update(visible=False), gr.update(visible=False)

        def update_visual_usage_type(visual_usage_type):
            if visual_usage_type == "Intra-Image":
                return gr.update(visible=False)
            if visual_usage_type == "Cross-Image":
                return gr.update(visible=True)
            return gr.update(visible=False)

        visual_prompt_type.change(
            fn=update_visual_prompt_type, inputs=[visual_prompt_type], outputs=[box_image, mask_image]
        )

        visual_usage_type.change(fn=update_visual_usage_type, inputs=[visual_usage_type], outputs=[target_image])

        def run_inference(
            raw_image,
            box_image,
            mask_image,
            target_image,
            texts,
            model_id,
            image_size,
            conf_thresh,
            iou_thresh,
            prompt_type,
            visual_prompt_type,
            visual_usage_type,
        ):
            # add text/built-in prompts
            if prompt_type == "Text" or prompt_type == "Prompt-free":
                target_image = None
                image = raw_image
                if prompt_type == "Prompt-free":
                    with open("tools/ram_tag_list.txt") as f:
                        texts = [x.strip() for x in f.readlines()]
                else:
                    texts = [text.strip() for text in texts.split(",")]
                prompts = {"texts": texts}
            # add visual prompt
            elif prompt_type == "Visual":
                if visual_usage_type != "Cross-Image":
                    target_image = None
                if visual_prompt_type == "bboxes":
                    image, points = box_image["image"], box_image["points"]
                    points = np.array(points)
                    if len(points) == 0:
                        gr.Warning("No boxes are provided. No image output.", visible=True)
                        return gr.update(value=None)
                    bboxes = np.array([p[[0, 1, 3, 4]] for p in points if p[2] == 2])
                    prompts = {"bboxes": bboxes, "cls": np.array([0] * len(bboxes))}
                elif visual_prompt_type == "masks":
                    image, masks = mask_image["background"], mask_image["layers"][0]
                    # image = image.convert("RGB")
                    masks = np.array(masks.convert("L"))
                    masks = binary_fill_holes(masks).astype(np.uint8)
                    masks[masks > 0] = 1
                    if masks.sum() == 0:
                        gr.Warning("No masks are provided. No image output.", visible=True)
                        return gr.update(value=None)
                    prompts = {"masks": masks[None], "cls": np.array([0])}
            return yoloe_inference(
                image, prompts, target_image, model_id, image_size, conf_thresh, iou_thresh, prompt_type
            )

        yoloe_infer.click(
            fn=run_inference,
            inputs=[
                raw_image,
                box_image,
                mask_image,
                target_image,
                texts,
                model_id,
                image_size,
                conf_thresh,
                iou_thresh,
                prompt_type,
                visual_prompt_type,
                visual_usage_type,
            ],
            outputs=[output_image],
        )

        ###################### Examples ##########################
        text_examples = gr.Examples(
            examples=[["ultralytics/assets/bus.jpg", "person,bus", "yoloe-v8l", 640, 0.25, 0.7]],
            inputs=[raw_image, texts, model_id, image_size, conf_thresh, iou_thresh],
            visible=True,
            cache_examples=False,
            label="Text Prompt Examples",
        )

        box_examples = gr.Examples(
            examples=[
                [
                    {"image": "ultralytics/assets/bus_box.jpg", "points": [[235, 408, 2, 342, 863, 3]]},
                    "ultralytics/assets/zidane.jpg",
                    "yoloe-v8l",
                    640,
                    0.2,
                    0.7,
                ]
            ],
            inputs=[box_image, target_image, model_id, image_size, conf_thresh, iou_thresh],
            visible=False,
            cache_examples=False,
            label="Box Visual Prompt Examples",
        )

        mask_examples = gr.Examples(
            examples=[
                [
                    {
                        "background": "ultralytics/assets/bus.jpg",
                        "layers": ["ultralytics/assets/bus_mask.png"],
                        "composite": "ultralytics/assets/bus_composite.jpg",
                    },
                    "ultralytics/assets/zidane.jpg",
                    "yoloe-v8l",
                    640,
                    0.15,
                    0.7,
                ]
            ],
            inputs=[mask_image, target_image, model_id, image_size, conf_thresh, iou_thresh],
            visible=False,
            cache_examples=False,
            label="Mask Visual Prompt Examples",
        )

        pf_examples = gr.Examples(
            examples=[
                [
                    "ultralytics/assets/bus.jpg",
                    "yoloe-v8l",
                    640,
                    0.25,
                    0.7,
                ]
            ],
            inputs=[raw_image, model_id, image_size, conf_thresh, iou_thresh],
            visible=False,
            cache_examples=False,
            label="Prompt-free Examples",
        )

        # Components update
        def load_box_example(visual_usage_type):
            return (
                gr.update(
                    visible=True,
                    value={"image": "ultralytics/assets/bus_box.jpg", "points": [[235, 408, 2, 342, 863, 3]]},
                ),
                gr.update(visible=(visual_usage_type == "Cross-Image")),
            )

        def load_mask_example(visual_usage_type):
            return gr.update(visible=True), gr.update(visible=(visual_usage_type == "Cross-Image"))

        box_examples.load_input_event.then(
            fn=load_box_example, inputs=visual_usage_type, outputs=[box_image, target_image]
        )

        mask_examples.load_input_event.then(
            fn=load_mask_example, inputs=visual_usage_type, outputs=[mask_image, target_image]
        )

        # Examples update
        def update_text_examples():
            return (
                gr.Dataset(visible=True),
                gr.Dataset(visible=False),
                gr.Dataset(visible=False),
                gr.Dataset(visible=False),
            )

        def update_pf_examples():
            return (
                gr.Dataset(visible=False),
                gr.Dataset(visible=False),
                gr.Dataset(visible=False),
                gr.Dataset(visible=True),
            )

        def update_visual_examples(visual_prompt_type):
            if visual_prompt_type == "bboxes":
                return (
                    gr.Dataset(visible=False),
                    gr.Dataset(visible=True),
                    gr.Dataset(visible=False),
                    gr.Dataset(visible=False),
                )
            elif visual_prompt_type == "masks":
                return (
                    gr.Dataset(visible=False),
                    gr.Dataset(visible=False),
                    gr.Dataset(visible=True),
                    gr.Dataset(visible=False),
                )

        text_tab.select(
            fn=update_text_examples,
            inputs=None,
            outputs=[text_examples.dataset, box_examples.dataset, mask_examples.dataset, pf_examples.dataset],
        )
        visual_tab.select(
            fn=update_visual_examples,
            inputs=[visual_prompt_type],
            outputs=[text_examples.dataset, box_examples.dataset, mask_examples.dataset, pf_examples.dataset],
        )
        prompt_free_tab.select(
            fn=update_pf_examples,
            inputs=None,
            outputs=[text_examples.dataset, box_examples.dataset, mask_examples.dataset, pf_examples.dataset],
        )
        visual_prompt_type.change(
            fn=update_visual_examples,
            inputs=[visual_prompt_type],
            outputs=[text_examples.dataset, box_examples.dataset, mask_examples.dataset, pf_examples.dataset],
        )
        visual_usage_type.change(
            fn=update_visual_examples,
            inputs=[visual_prompt_type],
            outputs=[text_examples.dataset, box_examples.dataset, mask_examples.dataset, pf_examples.dataset],
        )


gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    <img src="/file=figures/logo.png" width="2.5%" style="display:inline;padding-bottom:4px">
    YOLOE: Real-Time Seeing Anything
    </h1>
    """
    )
    gr.HTML(
        """
        <h3 style='text-align: center'>
        <a href='https://arxiv.org/abs/2503.07465' target='_blank'>arXiv</a> | <a href='https://github.com/THU-MIG/yoloe' target='_blank'>github</a>
        </h3>
        """
    )
    gr.Markdown(
        """
        We introduce **YOLOE(ye)**, a highly **efficient**, **unified**, and **open** object detection and segmentation model, like human eye, under different prompt mechanisms, like *texts*, *visual inputs*, and *prompt-free paradigm*.
        """
    )
    gr.Markdown(
        """
        If desired objects are not identified, pleaset set a **smaller** confidence threshold, e.g., for visual prompts with handcrafted shape or cross-image prompts.
        """
    )
    gr.Markdown(
        """
        Drawing **multiple** boxes or handcrafted shapes as visual prompt in an image is also supported, which leads to more accurate prompt.
        """
    )
    with gr.Row():
        with gr.Column():
            app()

if __name__ == "__main__":
    gradio_app.launch(allowed_paths=["figures"])
