// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

use anyhow::Result;
use clap::Parser;

use usls::{
    models::YOLO, Annotator, DataLoader, Device, Options, Viewer, Vision, YOLOScale, YOLOTask,
    YOLOVersion, COCO_SKELETONS_16,
};

#[derive(Parser, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Path to the ONNX model
    #[arg(long)]
    pub model: Option<String>,

    /// Input source path
    #[arg(long, default_value_t = String::from("../../ultralytics/assets/bus.jpg"))]
    pub source: String,

    /// YOLO Task
    #[arg(long, value_enum, default_value_t = YOLOTask::Detect)]
    pub task: YOLOTask,

    /// YOLO Version
    #[arg(long, value_enum, default_value_t = YOLOVersion::V8)]
    pub ver: YOLOVersion,

    /// YOLO Scale
    #[arg(long, value_enum, default_value_t = YOLOScale::N)]
    pub scale: YOLOScale,

    /// Batch size
    #[arg(long, default_value_t = 1)]
    pub batch_size: usize,

    /// Minimum input width
    #[arg(long, default_value_t = 224)]
    pub width_min: isize,

    /// Input width
    #[arg(long, default_value_t = 640)]
    pub width: isize,

    /// Maximum input width
    #[arg(long, default_value_t = 1024)]
    pub width_max: isize,

    /// Minimum input height
    #[arg(long, default_value_t = 224)]
    pub height_min: isize,

    /// Input height
    #[arg(long, default_value_t = 640)]
    pub height: isize,

    /// Maximum input height
    #[arg(long, default_value_t = 1024)]
    pub height_max: isize,

    /// Number of classes
    #[arg(long, default_value_t = 80)]
    pub nc: usize,

    /// Class confidence
    #[arg(long)]
    pub confs: Vec<f32>,

    /// Enable TensorRT support
    #[arg(long)]
    pub trt: bool,

    /// Enable CUDA support
    #[arg(long)]
    pub cuda: bool,

    /// Enable CoreML support
    #[arg(long)]
    pub coreml: bool,

    /// Use TensorRT half precision
    #[arg(long)]
    pub half: bool,

    /// Device ID to use
    #[arg(long, default_value_t = 0)]
    pub device_id: usize,

    /// Enable performance profiling
    #[arg(long)]
    pub profile: bool,

    /// Disable contour drawing, for saving time
    #[arg(long)]
    pub no_contours: bool,

    /// Show result
    #[arg(long)]
    pub view: bool,

    /// Do not save output
    #[arg(long)]
    pub nosave: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // logger
    if args.profile {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .init();
    }

    // model path
    let path = match &args.model {
        None => format!(
            "yolo/{}-{}-{}.onnx",
            args.ver.name(),
            args.scale.name(),
            args.task.name()
        ),
        Some(x) => x.to_string(),
    };

    // saveout
    let saveout = match &args.model {
        None => format!(
            "{}-{}-{}",
            args.ver.name(),
            args.scale.name(),
            args.task.name()
        ),
        Some(x) => {
            let p = std::path::PathBuf::from(&x);
            p.file_stem().unwrap().to_str().unwrap().to_string()
        }
    };

    // device
    let device = if args.cuda {
        Device::Cuda(args.device_id)
    } else if args.trt {
        Device::Trt(args.device_id)
    } else if args.coreml {
        Device::CoreML(args.device_id)
    } else {
        Device::Cpu(args.device_id)
    };

    // build options
    let options = Options::new()
        .with_model(&path)?
        .with_yolo_version(args.ver)
        .with_yolo_task(args.task)
        .with_device(device)
        .with_trt_fp16(args.half)
        .with_ixx(0, 0, (1, args.batch_size as _, 4).into())
        .with_ixx(0, 2, (args.height_min, args.height, args.height_max).into())
        .with_ixx(0, 3, (args.width_min, args.width, args.width_max).into())
        .with_confs(if args.confs.is_empty() {
            &[0.2, 0.15]
        } else {
            &args.confs
        })
        .with_nc(args.nc)
        .with_find_contours(!args.no_contours) // find contours or not
        // .with_names(&COCO_CLASS_NAMES_80)  // detection class names
        // .with_names2(&COCO_KEYPOINTS_17) // keypoints class names
        // .exclude_classes(&[0])
        // .retain_classes(&[0, 5])
        .with_profile(args.profile);

    // build model
    let mut model = YOLO::new(options)?;

    // build dataloader
    let dl = DataLoader::new(&args.source)?
        .with_batch(model.batch() as _)
        .build()?;

    // build annotator
    let annotator = Annotator::default()
        .with_skeletons(&COCO_SKELETONS_16)
        .without_masks(true) // no masks plotting when doing segment task
        .with_bboxes_thickness(3)
        .with_keypoints_name(false) // enable keypoints names
        .with_saveout_subs(&["YOLO"])
        .with_saveout(&saveout);

    // build viewer
    let mut viewer = if args.view {
        Some(Viewer::new().with_delay(5).with_scale(1.).resizable(true))
    } else {
        None
    };

    // run & annotate
    for (xs, _paths) in dl {
        let ys = model.forward(&xs, args.profile)?;
        let images_plotted = annotator.plot(&xs, &ys, !args.nosave)?;

        // show image
        match &mut viewer {
            Some(viewer) => viewer.imshow(&images_plotted)?,
            None => continue,
        }

        // check out window and key event
        match &mut viewer {
            Some(viewer) => {
                if !viewer.is_open() || viewer.is_key_pressed(usls::Key::Escape) {
                    break;
                }
            }
            None => continue,
        }

        // write video
        if !args.nosave {
            match &mut viewer {
                Some(viewer) => viewer.write_batch(&images_plotted)?,
                None => continue,
            }
        }
    }

    // finish video write
    if !args.nosave {
        if let Some(viewer) = &mut viewer {
            viewer.finish_write()?;
        }
    }

    Ok(())
}
