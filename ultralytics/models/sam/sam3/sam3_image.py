# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

from __future__ import annotations
from copy import deepcopy
import torch
from .vl_combiner import SAM3VLBackbone
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.nn.modules.utils import inverse_sigmoid
from .geometry_encoders import Prompt


def _update_out(out, out_name, out_value, auxiliary=True, update_aux=True):
    out[out_name] = out_value[-1] if auxiliary else out_value
    if auxiliary and update_aux:
        if "aux_outputs" not in out:
            out["aux_outputs"] = [{} for _ in range(len(out_value) - 1)]
        assert len(out["aux_outputs"]) == len(out_value) - 1
        for aux_output, aux_value in zip(out["aux_outputs"], out_value[:-1]):
            aux_output[out_name] = aux_value


class Sam3Image(torch.nn.Module):
    def __init__(
        self,
        backbone: SAM3VLBackbone,
        transformer,
        input_geometry_encoder,
        segmentation_head=None,
        num_feature_levels=1,
        o2m_mask_predict=True,
        dot_prod_scoring=None,
        use_instance_query: bool = True,
        multimask_output: bool = True,
        use_act_checkpoint_seg_head: bool = True,
        matcher=None,
        use_dot_prod_scoring=True,
        supervise_joint_box_scores: bool = False,  # only relevant if using presence token/score
        detach_presence_in_joint_score: bool = False,  # only relevant if using presence token/score
        separate_scorer_for_instance: bool = False,
        num_interactive_steps_val: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.backbone = backbone
        self.geometry_encoder = input_geometry_encoder
        self.transformer = transformer
        self.hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.segmentation_head = segmentation_head

        self.o2m_mask_predict = o2m_mask_predict

        self.dot_prod_scoring = dot_prod_scoring
        self.use_act_checkpoint_seg_head = use_act_checkpoint_seg_head
        self.matcher = matcher

        self.num_interactive_steps_val = num_interactive_steps_val
        self.use_dot_prod_scoring = use_dot_prod_scoring

        if self.use_dot_prod_scoring:
            assert dot_prod_scoring is not None
            self.dot_prod_scoring = dot_prod_scoring
            self.instance_dot_prod_scoring = None
            if separate_scorer_for_instance:
                self.instance_dot_prod_scoring = deepcopy(dot_prod_scoring)
        else:
            self.class_embed = torch.nn.Linear(self.hidden_dim, 1)
            self.instance_class_embed = None
            if separate_scorer_for_instance:
                self.instance_class_embed = deepcopy(self.class_embed)

        self.supervise_joint_box_scores = supervise_joint_box_scores
        self.detach_presence_in_joint_score = detach_presence_in_joint_score

        # verify the number of queries for O2O and O2M
        num_o2o_static = self.transformer.decoder.num_queries
        num_o2m_static = self.transformer.decoder.num_o2m_queries
        assert num_o2m_static == (num_o2o_static if self.transformer.decoder.dac else 0)
        self.dac = self.transformer.decoder.dac

        self.use_instance_query = use_instance_query
        self.multimask_output = multimask_output

        self.text_embeddings = {}
        self.names = []

    # TODO: remove this
    def _get_img_feats(self, backbone_out, img_ids):
        """Retrieve correct image features from backbone output."""
        if "backbone_fpn" in backbone_out:
            if "id_mapping" in backbone_out and backbone_out["id_mapping"] is not None:
                img_ids = backbone_out["id_mapping"][img_ids]
                # If this assert fails, it likely means we're requesting different img_ids (perhaps a different frame?)
                # We currently don't expect this to happen. We could technically trigger a recompute here,
                # but likely at the cost of a cpu<->gpu sync point, which would deteriorate perf
                torch._assert_async((img_ids >= 0).all())

            vis_feats = backbone_out["backbone_fpn"][-self.num_feature_levels :]
            vis_pos_enc = backbone_out["vision_pos_enc"][-self.num_feature_levels :]
            vis_feat_sizes = [x.shape[-2:] for x in vis_pos_enc]  # (H, W) shapes
            # index and flatten visual features NxCxHxW => HWxNxC (batch-first => seq-first)
            img_feats = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_feats]
            img_pos_embeds = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_pos_enc]
            return backbone_out, img_feats, img_pos_embeds, vis_feat_sizes

        # Image features not available in backbone output, so we compute them on the fly
        # This case likely occurs for video. In that case, we want to forward only the current frame
        img_batch = backbone_out["img_batch_all_stages"]
        if img_ids.numel() > 1:
            # Only forward backbone on unique image ids to avoid repetitive computation
            unique_ids, _ = torch.unique(img_ids, return_inverse=True)
        else:
            unique_ids, _ = img_ids, slice(None)
        # Compute the image features on those unique image ids
        # note: we allow using a list (or other indexable types) of tensors as img_batch
        # (e.g. for async frame loading in demo). In this case we index img_batch.tensors directly
        if isinstance(img_batch, torch.Tensor):
            image = img_batch[unique_ids]
        elif unique_ids.numel() == 1:
            image = img_batch[unique_ids.item()].unsqueeze(0)
        else:
            image = torch.stack([img_batch[i] for i in unique_ids.tolist()])
        # `img_batch` might be fp16 and offloaded to CPU
        # Next time we call this function, we want to remember which indices we computed
        id_mapping = torch.full((len(img_batch),), -1, dtype=torch.long, device=image.device)
        id_mapping[unique_ids] = torch.arange(len(unique_ids), device=image.device)
        backbone_out = {
            **backbone_out,
            **self.backbone.forward_image(image),
            "id_mapping": id_mapping,
        }
        assert "backbone_fpn" in backbone_out
        return self._get_img_feats(backbone_out, img_ids=img_ids)

    def _encode_prompt(
        self,
        img_feats,
        img_pos_embeds,
        vis_feat_sizes,
        geometric_prompt,
        visual_prompt_embed=None,
        visual_prompt_mask=None,
        prev_mask_pred=None,
    ):
        """Encode the geometric and visual prompts."""
        if prev_mask_pred is not None:
            img_feats = [img_feats[-1] + prev_mask_pred]
        # Encode geometry
        geo_feats, geo_masks = self.geometry_encoder(
            geo_prompt=geometric_prompt,
            img_feats=img_feats,
            img_sizes=vis_feat_sizes,
            img_pos_embeds=img_pos_embeds,
        )
        if visual_prompt_embed is None:
            visual_prompt_embed = torch.zeros((0, *geo_feats.shape[1:]), device=geo_feats.device)
            visual_prompt_mask = torch.zeros(
                (*geo_masks.shape[:-1], 0),
                device=geo_masks.device,
                dtype=geo_masks.dtype,
            )
        prompt = torch.cat([geo_feats, visual_prompt_embed], dim=0)
        prompt_mask = torch.cat([geo_masks, visual_prompt_mask], dim=1)
        return prompt, prompt_mask

    def _run_encoder(
        self,
        img_feats,
        img_pos_embeds,
        vis_feat_sizes,
        prompt,
        prompt_mask,
        encoder_extra_kwargs: dict = None,
    ):
        """Run the transformer encoder."""
        # Run the encoder
        # make a copy of the image feature lists since the encoder may modify these lists in-place
        memory = self.transformer.encoder(
            src=img_feats.copy(),
            src_key_padding_mask=None,
            src_pos=img_pos_embeds.copy(),
            prompt=prompt,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=vis_feat_sizes,
            encoder_extra_kwargs=encoder_extra_kwargs,
        )
        encoder_out = {
            # encoded image features
            "encoder_hidden_states": memory["memory"],
            "pos_embed": memory["pos_embed"],
            "padding_mask": memory["padding_mask"],
            "spatial_shapes": memory["spatial_shapes"],
            "valid_ratios": memory["valid_ratios"],
            "vis_feat_sizes": vis_feat_sizes,
            # encoded text features (or other prompts)
            "prompt_before_enc": prompt,
            "prompt_after_enc": memory.get("memory_text", prompt),
            "prompt_mask": prompt_mask,
        }
        return encoder_out

    def _run_decoder(
        self,
        pos_embed,
        memory,
        src_mask,
        out,
        prompt,
        prompt_mask,
        encoder_out,
    ):
        """Run the transformer decoder."""
        bs = memory.shape[1]
        query_embed = self.transformer.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)

        hs, reference_boxes, dec_presence_out, _ = self.transformer.decoder(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=src_mask,
            pos=pos_embed,
            reference_boxes=None,
            spatial_shapes=encoder_out["spatial_shapes"],
            valid_ratios=encoder_out["valid_ratios"],
            tgt_mask=None,
            memory_text=prompt,
            text_attention_mask=prompt_mask,
            apply_dac=False,
        )
        hs = hs.transpose(1, 2)  # seq-first to batch-first
        reference_boxes = reference_boxes.transpose(1, 2)  # seq-first to batch-first
        if dec_presence_out is not None:
            # seq-first to batch-first
            dec_presence_out = dec_presence_out.transpose(1, 2)
        self._update_scores_and_boxes(
            out,
            hs,
            reference_boxes,
            prompt,
            prompt_mask,
            dec_presence_out=dec_presence_out,
        )
        return out, hs

    def _update_scores_and_boxes(
        self,
        out,
        hs,
        reference_boxes,
        prompt,
        prompt_mask,
        dec_presence_out=None,
        is_instance_prompt=False,
    ):
        """Update output dict with class scores and box predictions."""
        num_o2o = hs.size(2)
        # score prediction
        if self.use_dot_prod_scoring:
            dot_prod_scoring_head = self.dot_prod_scoring
            if is_instance_prompt and self.instance_dot_prod_scoring is not None:
                dot_prod_scoring_head = self.instance_dot_prod_scoring
            outputs_class = dot_prod_scoring_head(hs, prompt, prompt_mask)
        else:
            class_embed_head = self.class_embed
            if is_instance_prompt and self.instance_class_embed is not None:
                class_embed_head = self.instance_class_embed
            outputs_class = class_embed_head(hs)

        # box prediction
        box_head = self.transformer.decoder.bbox_embed
        if is_instance_prompt and self.transformer.decoder.instance_bbox_embed is not None:
            box_head = self.transformer.decoder.instance_bbox_embed
        anchor_box_offsets = box_head(hs)
        reference_boxes_inv_sig = inverse_sigmoid(reference_boxes)
        outputs_coord = (reference_boxes_inv_sig + anchor_box_offsets).sigmoid()
        outputs_boxes_xyxy = xywh2xyxy(outputs_coord)

        if dec_presence_out is not None:
            _update_out(out, "presence_logit_dec", dec_presence_out, update_aux=False)

        if self.supervise_joint_box_scores:
            assert dec_presence_out is not None
            prob_dec_presence_out = dec_presence_out.clone().sigmoid()
            if self.detach_presence_in_joint_score:
                prob_dec_presence_out = prob_dec_presence_out.detach()

            outputs_class = inverse_sigmoid(outputs_class.sigmoid() * prob_dec_presence_out.unsqueeze(2)).clamp(
                min=-10.0, max=10.0
            )

        _update_out(out, "pred_logits", outputs_class[:, :, :num_o2o], update_aux=False)
        _update_out(out, "pred_boxes", outputs_coord[:, :, :num_o2o], update_aux=False)
        _update_out(out, "pred_boxes_xyxy", outputs_boxes_xyxy[:, :, :num_o2o], update_aux=False)

    def _run_segmentation_heads(
        self,
        out,
        backbone_out,
        img_ids,
        encoder_hidden_states,
        prompt,
        prompt_mask,
        hs,
    ):
        """Run segmentation heads and get masks."""
        if self.segmentation_head is not None:
            num_o2o = hs.size(2)
            obj_queries = hs if self.o2m_mask_predict else hs[:, :, :num_o2o]
            seg_head_outputs = self.segmentation_head(
                backbone_feats=backbone_out["backbone_fpn"],
                obj_queries=obj_queries,
                image_ids=img_ids,
                encoder_hidden_states=encoder_hidden_states,
                prompt=prompt,
                prompt_mask=prompt_mask,
            )
            for k, v in seg_head_outputs.items():
                if k in self.segmentation_head.instance_keys:
                    _update_out(out, k, v[:, :num_o2o], auxiliary=False)
                else:
                    out[k] = v
        else:
            backbone_out.pop("backbone_fpn", None)

    def forward_grounding(self, backbone_out, find_input, geometric_prompt: Prompt = None):
        """Forward pass for grounding (detection + segmentation) given input images and text."""
        backbone_out.update({k: v for k, v in self.text_embeddings.items()})
        backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = self._get_img_feats(backbone_out, find_input.img_ids)
        with torch.profiler.record_function("SAM3Image._encode_prompt"):
            prompt, prompt_mask = self._encode_prompt(img_feats, img_pos_embeds, vis_feat_sizes, geometric_prompt)
        # index text features (note that regardless of early or late fusion, the batch size of
        # `txt_feats` is always the number of *prompts* in the encoder)
        txt_ids = find_input.text_ids
        txt_feats = backbone_out["language_features"][:, txt_ids]
        txt_masks = backbone_out["language_mask"][txt_ids]
        # encode text
        prompt = torch.cat([txt_feats, prompt], dim=0)
        prompt_mask = torch.cat([txt_masks, prompt_mask], dim=1)

        # Run the encoder
        with torch.profiler.record_function("SAM3Image._run_encoder"):
            encoder_out = self._run_encoder(img_feats, img_pos_embeds, vis_feat_sizes, prompt, prompt_mask)
        out = {"backbone_out": backbone_out}

        # Run the decoder
        with torch.profiler.record_function("SAM3Image._run_decoder"):
            out, hs = self._run_decoder(
                memory=encoder_out["encoder_hidden_states"],
                pos_embed=encoder_out["pos_embed"],
                src_mask=encoder_out["padding_mask"],
                out=out,
                prompt=prompt,
                prompt_mask=prompt_mask,
                encoder_out=encoder_out,
            )

        # Run segmentation heads
        with torch.profiler.record_function("SAM3Image._run_segmentation_heads"):
            self._run_segmentation_heads(
                out=out,
                backbone_out=backbone_out,
                img_ids=find_input.img_ids,
                encoder_hidden_states=encoder_out["encoder_hidden_states"],
                prompt=prompt,
                prompt_mask=prompt_mask,
                hs=hs,
            )
        return out

    def set_classes(self, text: list[str]):
        """Set the text embeddings for the given class names."""
        self.text_embeddings = self.backbone.forward_text(text)
        self.names = text

    def set_imgsz(self, imgsz: tuple[int, int]):
        """Set the image size for the model."""
        self.backbone.set_imgsz(imgsz)
