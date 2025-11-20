#![allow(clippy::type_complexity)]
// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

use ab_glyph::FontArc;
use anyhow::Result;
use image::{DynamicImage, GenericImageView, ImageBuffer};
use ndarray::{s, Array, Axis, IxDyn};
use rand::{thread_rng, Rng};
use std::path::PathBuf;

use crate::{
    gen_time_string, load_font, non_max_suppression, Args, Batch, Bbox, Embedding, OrtBackend,
    OrtConfig, OrtEP, Point2, YOLOResult, YOLOTask, SKELETON,
};

pub struct YOLOv8 {
    // YOLOv8 model for all yolo-tasks
    engine: OrtBackend,
    nc: u32,
    nk: u32,
    nm: u32,
    height: u32,
    width: u32,
    batch: u32,
    task: YOLOTask,
    conf: f32,
    kconf: f32,
    iou: f32,
    names: Vec<String>,
    color_palette: Vec<(u8, u8, u8)>,
    profile: bool,
    plot: bool,
}

impl YOLOv8 {
    pub fn new(config: Args) -> Result<Self> {
        // execution provider
        let ep = if config.trt {
            OrtEP::Trt(config.device_id)
        } else if config.cuda {
            OrtEP::CUDA(config.device_id)
        } else {
            OrtEP::CPU
        };

        // batch
        let batch = Batch {
            opt: config.batch,
            min: config.batch_min,
            max: config.batch_max,
        };

        // build ort engine
        let ort_args = OrtConfig {
            ep,
            batch,
            f: config.model,
            task: config.task,
            trt_fp16: config.fp16,
            image_size: (config.height, config.width),
        };
        let engine = OrtBackend::build(ort_args)?;

        //  get batch, height, width, tasks, nc, nk, nm
        let (batch, height, width, task) = (
            engine.batch(),
            engine.height(),
            engine.width(),
            engine.task(),
        );
        let nc = engine.nc().or(config.nc).unwrap_or_else(|| {
            panic!("Failed to get num_classes, make it explicit with `--nc`");
        });
        let (nk, nm) = match task {
            YOLOTask::Pose => {
                let nk = engine.nk().or(config.nk).unwrap_or_else(|| {
                    panic!("Failed to get num_keypoints, make it explicit with `--nk`");
                });
                (nk, 0)
            }
            YOLOTask::Segment => {
                let nm = engine.nm().or(config.nm).unwrap_or_else(|| {
                    panic!("Failed to get num_masks, make it explicit with `--nm`");
                });
                (0, nm)
            }
            _ => (0, 0),
        };

        // class names
        let names = engine.names().unwrap_or(vec!["Unknown".to_string()]);

        // color palette
        let mut rng = thread_rng();
        let color_palette: Vec<_> = names
            .iter()
            .map(|_| {
                (
                    rng.gen_range(0..=255),
                    rng.gen_range(0..=255),
                    rng.gen_range(0..=255),
                )
            })
            .collect();

        Ok(Self {
            engine,
            names,
            conf: config.conf,
            kconf: config.kconf,
            iou: config.iou,
            color_palette,
            profile: config.profile,
            plot: config.plot,
            nc,
            nk,
            nm,
            height,
            width,
            batch,
            task,
        })
    }

    pub fn scale_wh(&self, w0: f32, h0: f32, w1: f32, h1: f32) -> (f32, f32, f32) {
        let r = (w1 / w0).min(h1 / h0);
        (r, (w0 * r).round(), (h0 * r).round())
    }

    pub fn preprocess(&mut self, xs: &Vec<DynamicImage>) -> Result<Array<f32, IxDyn>> {
        let mut ys =
            Array::ones((xs.len(), 3, self.height() as usize, self.width() as usize)).into_dyn();
        ys.fill(144.0 / 255.0);
        for (idx, x) in xs.iter().enumerate() {
            let img = match self.task() {
                YOLOTask::Classify => x.resize_exact(
                    self.width(),
                    self.height(),
                    image::imageops::FilterType::Triangle,
                ),
                _ => {
                    let (w0, h0) = x.dimensions();
                    let w0 = w0 as f32;
                    let h0 = h0 as f32;
                    let (_, w_new, h_new) =
                        self.scale_wh(w0, h0, self.width() as f32, self.height() as f32); // f32 round
                    x.resize_exact(
                        w_new as u32,
                        h_new as u32,
                        if let YOLOTask::Segment = self.task() {
                            image::imageops::FilterType::CatmullRom
                        } else {
                            image::imageops::FilterType::Triangle
                        },
                    )
                }
            };

            for (x, y, rgb) in img.pixels() {
                let x = x as usize;
                let y = y as usize;
                let [r, g, b, _] = rgb.0;
                ys[[idx, 0, y, x]] = (r as f32) / 255.0;
                ys[[idx, 1, y, x]] = (g as f32) / 255.0;
                ys[[idx, 2, y, x]] = (b as f32) / 255.0;
            }
        }

        Ok(ys)
    }

    pub fn run(&mut self, xs: &Vec<DynamicImage>) -> Result<Vec<YOLOResult>> {
        // pre-process
        let t_pre = std::time::Instant::now();
        let xs_ = self.preprocess(xs)?;
        if self.profile {
            println!("[Model Preprocess]: {:?}", t_pre.elapsed());
        }

        // run
        let t_run = std::time::Instant::now();
        let ys = self.engine.run(xs_, self.profile)?;
        if self.profile {
            println!("[Model Inference]: {:?}", t_run.elapsed());
        }

        // post-process
        let t_post = std::time::Instant::now();
        let ys = self.postprocess(ys, xs)?;
        if self.profile {
            println!("[Model Postprocess]: {:?}", t_post.elapsed());
        }

        // plot and save
        if self.plot {
            self.plot_and_save(&ys, xs, Some(&SKELETON));
        }
        Ok(ys)
    }

    pub fn postprocess(
        &self,
        xs: Vec<Array<f32, IxDyn>>,
        xs0: &[DynamicImage],
    ) -> Result<Vec<YOLOResult>> {
        if let YOLOTask::Classify = self.task() {
            let mut ys = Vec::new();
            let preds = &xs[0];
            for batch in preds.axis_iter(Axis(0)) {
                ys.push(YOLOResult::new(
                    Some(Embedding::new(batch.into_owned())),
                    None,
                    None,
                    None,
                ));
            }
            Ok(ys)
        } else {
            const CXYWH_OFFSET: usize = 4; // cxcywh
            const KPT_STEP: usize = 3; // xyconf
            let preds = &xs[0];
            let protos = {
                if xs.len() > 1 {
                    Some(&xs[1])
                } else {
                    None
                }
            };
            let mut ys = Vec::new();
            for (idx, anchor) in preds.axis_iter(Axis(0)).enumerate() {
                // [bs, 4 + nc + nm, anchors]
                // input image
                let width_original = xs0[idx].width() as f32;
                let height_original = xs0[idx].height() as f32;
                let ratio = (self.width() as f32 / width_original)
                    .min(self.height() as f32 / height_original);

                // save each result
                let mut data: Vec<(Bbox, Option<Vec<Point2>>, Option<Vec<f32>>)> = Vec::new();
                for pred in anchor.axis_iter(Axis(1)) {
                    // split preds for different tasks
                    let bbox = pred.slice(s![0..CXYWH_OFFSET]);
                    let clss = pred.slice(s![CXYWH_OFFSET..CXYWH_OFFSET + self.nc() as usize]);
                    let kpts = {
                        if let YOLOTask::Pose = self.task() {
                            Some(pred.slice(s![pred.len() - KPT_STEP * self.nk() as usize..]))
                        } else {
                            None
                        }
                    };
                    let coefs = {
                        if let YOLOTask::Segment = self.task() {
                            Some(pred.slice(s![pred.len() - self.nm() as usize..]).to_vec())
                        } else {
                            None
                        }
                    };

                    // confidence and id
                    let (id, &confidence) = clss
                        .into_iter()
                        .enumerate()
                        .reduce(|max, x| if x.1 > max.1 { x } else { max })
                        .unwrap(); // definitely will not panic!

                    // confidence filter
                    if confidence < self.conf {
                        continue;
                    }

                    // bbox re-scale
                    let cx = bbox[0] / ratio;
                    let cy = bbox[1] / ratio;
                    let w = bbox[2] / ratio;
                    let h = bbox[3] / ratio;
                    let x = cx - w / 2.;
                    let y = cy - h / 2.;
                    let y_bbox = Bbox::new(
                        x.max(0.0f32).min(width_original),
                        y.max(0.0f32).min(height_original),
                        w,
                        h,
                        id,
                        confidence,
                    );

                    // kpts
                    let y_kpts = {
                        if let Some(kpts) = kpts {
                            let mut kpts_ = Vec::new();
                            // rescale
                            for i in 0..self.nk() as usize {
                                let kx = kpts[KPT_STEP * i] / ratio;
                                let ky = kpts[KPT_STEP * i + 1] / ratio;
                                let kconf = kpts[KPT_STEP * i + 2];
                                if kconf < self.kconf {
                                    kpts_.push(Point2::default());
                                } else {
                                    kpts_.push(Point2::new_with_conf(
                                        kx.max(0.0f32).min(width_original),
                                        ky.max(0.0f32).min(height_original),
                                        kconf,
                                    ));
                                }
                            }
                            Some(kpts_)
                        } else {
                            None
                        }
                    };

                    // data merged
                    data.push((y_bbox, y_kpts, coefs));
                }

                // nms
                non_max_suppression(&mut data, self.iou);

                // decode
                let mut y_bboxes: Vec<Bbox> = Vec::new();
                let mut y_kpts: Vec<Vec<Point2>> = Vec::new();
                let mut y_masks: Vec<Vec<u8>> = Vec::new();
                for elem in data.into_iter() {
                    if let Some(kpts) = elem.1 {
                        y_kpts.push(kpts)
                    }

                    // decode masks
                    if let Some(coefs) = elem.2 {
                        let proto = protos.unwrap().slice(s![idx, .., .., ..]);
                        let (nm, nh, nw) = proto.dim();

                        // coefs * proto -> mask
                        let coefs = Array::from_shape_vec((1, nm), coefs)?; // (n, nm)

                        let proto = proto.to_owned();
                        let proto = proto.to_shape((nm, nh * nw))?; // (nm, nh*nw)
                        let mask = coefs.dot(&proto); // (nh, nw, n)
                        let mask = mask.to_shape((nh, nw, 1))?;

                        // build image from ndarray
                        let mask_im: ImageBuffer<image::Luma<_>, Vec<f32>> =
                            match ImageBuffer::from_raw(
                                nw as u32,
                                nh as u32,
                                mask.to_owned().into_raw_vec_and_offset().0,
                            ) {
                                Some(image) => image,
                                None => panic!("can not create image from ndarray"),
                            };
                        let mut mask_im = image::DynamicImage::from(mask_im); // -> dyn

                        // rescale masks
                        let (_, w_mask, h_mask) =
                            self.scale_wh(width_original, height_original, nw as f32, nh as f32);
                        let mask_cropped = mask_im.crop(0, 0, w_mask as u32, h_mask as u32);
                        let mask_original = mask_cropped.resize_exact(
                            // resize_to_fill
                            width_original as u32,
                            height_original as u32,
                            match self.task() {
                                YOLOTask::Segment => image::imageops::FilterType::CatmullRom,
                                _ => image::imageops::FilterType::Triangle,
                            },
                        );

                        // crop-mask with bbox
                        let mut mask_original_cropped = mask_original.into_luma8();
                        for y in 0..height_original as usize {
                            for x in 0..width_original as usize {
                                if x < elem.0.xmin() as usize
                                    || x > elem.0.xmax() as usize
                                    || y < elem.0.ymin() as usize
                                    || y > elem.0.ymax() as usize
                                {
                                    mask_original_cropped.put_pixel(
                                        x as u32,
                                        y as u32,
                                        image::Luma([0u8]),
                                    );
                                }
                            }
                        }
                        y_masks.push(mask_original_cropped.into_raw());
                    }
                    y_bboxes.push(elem.0);
                }

                // save each result
                let y = YOLOResult {
                    probs: None,
                    bboxes: if !y_bboxes.is_empty() {
                        Some(y_bboxes)
                    } else {
                        None
                    },
                    keypoints: if !y_kpts.is_empty() {
                        Some(y_kpts)
                    } else {
                        None
                    },
                    masks: if !y_masks.is_empty() {
                        Some(y_masks)
                    } else {
                        None
                    },
                };
                ys.push(y);
            }

            Ok(ys)
        }
    }

    pub fn plot_and_save(
        &self,
        ys: &[YOLOResult],
        xs0: &[DynamicImage],
        skeletons: Option<&[(usize, usize)]>,
    ) {
        // check font then load
        let font: FontArc = load_font();
        for (_idb, (img0, y)) in xs0.iter().zip(ys.iter()).enumerate() {
            let mut img = img0.to_rgb8();

            // draw for classifier
            if let Some(probs) = y.probs() {
                for (i, k) in probs.topk(5).iter().enumerate() {
                    let legend = format!("{} {:.2}%", self.names[k.0], k.1);
                    let scale = 32;
                    let legend_size = img.width().max(img.height()) / scale;
                    let x = img.width() / 20;
                    let y = img.height() / 20 + i as u32 * legend_size;

                    imageproc::drawing::draw_text_mut(
                        &mut img,
                        image::Rgb([0, 255, 0]),
                        x as i32,
                        y as i32,
                        legend_size as f32,
                        &font,
                        &legend,
                    );
                }
            }

            // draw bboxes & keypoints
            if let Some(bboxes) = y.bboxes() {
                for (_idx, bbox) in bboxes.iter().enumerate() {
                    // rect
                    imageproc::drawing::draw_hollow_rect_mut(
                        &mut img,
                        imageproc::rect::Rect::at(bbox.xmin() as i32, bbox.ymin() as i32)
                            .of_size(bbox.width() as u32, bbox.height() as u32),
                        image::Rgb(self.color_palette[bbox.id()].into()),
                    );

                    // text
                    let legend = format!("{} {:.2}%", self.names[bbox.id()], bbox.confidence());
                    let scale = 40;
                    let legend_size = img.width().max(img.height()) / scale;
                    imageproc::drawing::draw_text_mut(
                        &mut img,
                        image::Rgb(self.color_palette[bbox.id()].into()),
                        bbox.xmin() as i32,
                        (bbox.ymin() - legend_size as f32) as i32,
                        legend_size as f32,
                        &font,
                        &legend,
                    );
                }
            }

            // draw kpts
            if let Some(keypoints) = y.keypoints() {
                for kpts in keypoints.iter() {
                    for kpt in kpts.iter() {
                        // filter
                        if kpt.confidence() < self.kconf {
                            continue;
                        }

                        // draw point
                        imageproc::drawing::draw_filled_circle_mut(
                            &mut img,
                            (kpt.x() as i32, kpt.y() as i32),
                            2,
                            image::Rgb([0, 255, 0]),
                        );
                    }

                    // draw skeleton if has
                    if let Some(skeletons) = skeletons {
                        for &(idx1, idx2) in skeletons.iter() {
                            let kpt1 = &kpts[idx1];
                            let kpt2 = &kpts[idx2];
                            if kpt1.confidence() < self.kconf || kpt2.confidence() < self.kconf {
                                continue;
                            }
                            imageproc::drawing::draw_line_segment_mut(
                                &mut img,
                                (kpt1.x(), kpt1.y()),
                                (kpt2.x(), kpt2.y()),
                                image::Rgb([233, 14, 57]),
                            );
                        }
                    }
                }
            }

            // draw mask
            if let Some(masks) = y.masks() {
                for (mask, _bbox) in masks.iter().zip(y.bboxes().unwrap().iter()) {
                    let mask_nd: ImageBuffer<image::Luma<_>, Vec<u8>> =
                        match ImageBuffer::from_vec(img.width(), img.height(), mask.to_vec()) {
                            Some(image) => image,
                            None => panic!("can not crate image from ndarray"),
                        };

                    for _x in 0..img.width() {
                        for _y in 0..img.height() {
                            let mask_p = imageproc::drawing::Canvas::get_pixel(&mask_nd, _x, _y);
                            if mask_p.0[0] > 0 {
                                let mut img_p = imageproc::drawing::Canvas::get_pixel(&img, _x, _y);
                                // img_p.0[2] = self.color_palette[bbox.id()].2 / 2;
                                // img_p.0[1] = self.color_palette[bbox.id()].1 / 2;
                                // img_p.0[0] = self.color_palette[bbox.id()].0 / 2;
                                img_p.0[2] /= 2;
                                img_p.0[1] = 255 - (255 - img_p.0[2]) / 2;
                                img_p.0[0] /= 2;
                                imageproc::drawing::Canvas::draw_pixel(&mut img, _x, _y, img_p)
                            }
                        }
                    }
                }
            }

            // mkdir and save
            let mut runs = PathBuf::from("runs");
            if !runs.exists() {
                std::fs::create_dir_all(&runs).unwrap();
            }
            runs.push(gen_time_string("-"));
            let saveout = format!("{}.jpg", runs.to_str().unwrap());
            let _ = img.save(saveout);
        }
    }

    pub fn summary(&self) {
        println!(
            "\nSummary:\n\
            > Task: {:?}{}\n\
            > EP: {:?} {}\n\
            > Dtype: {:?}\n\
            > Batch: {} ({}), Height: {} ({}), Width: {} ({})\n\
            > nc: {} nk: {}, nm: {}, conf: {}, kconf: {}, iou: {}\n\
            ",
            self.task(),
            match self.engine.author().zip(self.engine.version()) {
                Some((author, ver)) => format!(" ({} {})", author, ver),
                None => String::from(""),
            },
            self.engine.ep(),
            if let OrtEP::CPU = self.engine.ep() {
                ""
            } else {
                "(May still fall back to CPU)"
            },
            self.engine.dtype(),
            self.batch(),
            if self.engine.is_batch_dynamic() {
                "Dynamic"
            } else {
                "Const"
            },
            self.height(),
            if self.engine.is_height_dynamic() {
                "Dynamic"
            } else {
                "Const"
            },
            self.width(),
            if self.engine.is_width_dynamic() {
                "Dynamic"
            } else {
                "Const"
            },
            self.nc(),
            self.nk(),
            self.nm(),
            self.conf,
            self.kconf,
            self.iou,
        );
    }

    pub fn engine(&self) -> &OrtBackend {
        &self.engine
    }

    pub fn conf(&self) -> f32 {
        self.conf
    }

    pub fn set_conf(&mut self, val: f32) {
        self.conf = val;
    }

    pub fn conf_mut(&mut self) -> &mut f32 {
        &mut self.conf
    }

    pub fn kconf(&self) -> f32 {
        self.kconf
    }

    pub fn iou(&self) -> f32 {
        self.iou
    }

    pub fn task(&self) -> &YOLOTask {
        &self.task
    }

    pub fn batch(&self) -> u32 {
        self.batch
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn nc(&self) -> u32 {
        self.nc
    }

    pub fn nk(&self) -> u32 {
        self.nk
    }

    pub fn nm(&self) -> u32 {
        self.nm
    }

    pub fn names(&self) -> &Vec<String> {
        &self.names
    }
}
