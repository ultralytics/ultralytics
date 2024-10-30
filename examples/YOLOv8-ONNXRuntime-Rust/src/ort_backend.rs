use anyhow::Result;
use clap::ValueEnum;
use half::f16;
use ndarray::{Array, CowArray, IxDyn};
use ort::{
    CPUExecutionProvider, CUDAExecutionProvider, ExecutionProvider, ExecutionProviderDispatch,
    TensorRTExecutionProvider,
};
use ort::{Session, SessionBuilder};
use ort::{TensorElementType, ValueType};
use regex::Regex;
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum YOLOTask {
    // YOLO tasks
    Classify,
    Detect,
    Pose,
    Segment,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum OrtEP {
    // ONNXRuntime execution provider
    CPU,
    CUDA(i32),
    Trt(i32),
}

#[derive(Debug)]
pub struct Batch {
    pub opt: u32,
    pub min: u32,
    pub max: u32,
}

impl Default for Batch {
    fn default() -> Self {
        Self {
            opt: 1,
            min: 1,
            max: 1,
        }
    }
}

#[derive(Debug, Default)]
pub struct OrtInputs {
    // ONNX model inputs attrs
    pub shapes: Vec<Vec<i64>>,
    //pub dtypes: Vec<TensorElementDataType>,
    pub dtypes: Vec<TensorElementType>,
    pub names: Vec<String>,
    pub sizes: Vec<Vec<u32>>,
}

impl OrtInputs {
    pub fn new(session: &Session) -> Self {
        let mut shapes = Vec::new();
        let mut dtypes = Vec::new();
        let mut names = Vec::new();
        for i in session.inputs.iter() {
            /*             let shape: Vec<i32> = i
                .dimensions()
                .map(|x| if let Some(x) = x { x as i32 } else { -1i32 })
                .collect();
            shapes.push(shape); */
            if let ort::ValueType::Tensor { ty, dimensions } = &i.input_type {
                dtypes.push(ty.clone());
                let shape = dimensions.clone();
                shapes.push(shape);
            } else {
                panic!("不支持的数据格式, {} - {}", file!(), line!());
            }
            //dtypes.push(i.input_type);
            names.push(i.name.clone());
        }
        Self {
            shapes,
            dtypes,
            names,
            ..Default::default()
        }
    }
}

#[derive(Debug)]
pub struct OrtConfig {
    // ORT config
    pub f: String,
    pub task: Option<YOLOTask>,
    pub ep: OrtEP,
    pub trt_fp16: bool,
    pub batch: Batch,
    pub image_size: (Option<u32>, Option<u32>),
}

#[derive(Debug)]
pub struct OrtBackend {
    // ORT engine
    session: Session,
    task: YOLOTask,
    ep: OrtEP,
    batch: Batch,
    inputs: OrtInputs,
}

impl OrtBackend {
    pub fn build(args: OrtConfig) -> Result<Self> {
        // build env & session
        // in version 2.x environment is removed
        /*         let env = ort::EnvironmentBuilder
        ::with_name("YOLOv8")
        .build()?
        .into_arc(); */
        let sessionbuilder = SessionBuilder::new()?;
        let session = sessionbuilder.commit_from_file(&args.f)?;
        //let session = SessionBuilder::new(&env)?.with_model_from_file(&args.f)?;

        // get inputs
        let mut inputs = OrtInputs::new(&session);

        // batch size
        let mut batch = args.batch;
        let batch = if inputs.shapes[0][0] == -1 {
            batch
        } else {
            assert_eq!(
                inputs.shapes[0][0] as u32, batch.opt,
                "Expected batch size: {}, got {}. Try using `--batch {}`.",
                inputs.shapes[0][0] as u32, batch.opt, inputs.shapes[0][0] as u32
            );
            batch.opt = inputs.shapes[0][0] as u32;
            batch
        };

        // input size: height and width
        let height = if inputs.shapes[0][2] == -1 {
            match args.image_size.0 {
                Some(height) => height,
                None => panic!("Failed to get model height. Make it explicit with `--height`"),
            }
        } else {
            inputs.shapes[0][2] as u32
        };
        let width = if inputs.shapes[0][3] == -1 {
            match args.image_size.1 {
                Some(width) => width,
                None => panic!("Failed to get model width. Make it explicit with `--width`"),
            }
        } else {
            inputs.shapes[0][3] as u32
        };
        inputs.sizes.push(vec![height, width]);

        // build provider
        let (ep, provider) = match args.ep {
            OrtEP::CUDA(device_id) => Self::set_ep_cuda(device_id),
            OrtEP::Trt(device_id) => Self::set_ep_trt(device_id, args.trt_fp16, &batch, &inputs),
            _ => (
                OrtEP::CPU,
                ExecutionProviderDispatch::from(CPUExecutionProvider::default()),
            ),
        };

        // build session again with the new provider
        let session = SessionBuilder::new()?
            // .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_execution_providers([provider])?
            .commit_from_file(args.f)?;

        // task: using given one or guessing
        let task = match args.task {
            Some(task) => task,
            None => match session.metadata() {
                Err(_) => panic!("No metadata found. Try making it explicit by `--task`"),
                Ok(metadata) => match metadata.custom("task") {
                    Err(_) => panic!("Can not get custom value. Try making it explicit by `--task`"),
                    Ok(value) => match value {
                        None => panic!("No corresponding value of `task` found in metadata. Make it explicit by `--task`"),
                        Some(task) => match task.as_str() {
                            "classify" => YOLOTask::Classify,
                            "detect" => YOLOTask::Detect,
                            "pose" => YOLOTask::Pose,
                            "segment" => YOLOTask::Segment,
                            x => todo!("{:?} is not supported for now!", x),
                        },
                    },
                },
            },
        };

        Ok(Self {
            session,
            task,
            ep,
            batch,
            inputs,
        })
    }

    pub fn fetch_inputs_from_session(
        session: &Session,
    ) -> (Vec<Vec<i64>>, Vec<TensorElementType>, Vec<String>) {
        // get inputs attrs from ONNX model
        let mut shapes = Vec::new();
        let mut dtypes = Vec::new();
        let mut names = Vec::new();
        for i in session.inputs.iter() {
            if let ort::ValueType::Tensor { ty, dimensions } = &i.input_type {
                dtypes.push(ty.clone());
                let shape = dimensions.clone();
                shapes.push(shape);
            } else {
                panic!("不支持的数据格式, {} - {}", file!(), line!());
            }
            names.push(i.name.clone());
        }
        (shapes, dtypes, names)
    }

    pub fn set_ep_cuda(device_id: i32) -> (OrtEP, ExecutionProviderDispatch) {
        let cuda_provider = CUDAExecutionProvider::default().with_device_id(device_id);
        if let Ok(true) = cuda_provider.is_available() {
            (
                OrtEP::CUDA(device_id),
                ExecutionProviderDispatch::from(cuda_provider), //PlantForm::CUDA(cuda_provider)
            )
        } else {
            println!("> CUDA is not available! Using CPU.");
            (
                OrtEP::CPU,
                ExecutionProviderDispatch::from(CPUExecutionProvider::default()), //PlantForm::CPU(CPUExecutionProvider::default())
            )
        }
    }

    pub fn set_ep_trt(
        device_id: i32,
        fp16: bool,
        batch: &Batch,
        inputs: &OrtInputs,
    ) -> (OrtEP, ExecutionProviderDispatch) {
        // set TensorRT
        let trt_provider = TensorRTExecutionProvider::default().with_device_id(device_id);

        //trt_provider.
        if let Ok(true) = trt_provider.is_available() {
            let (height, width) = (inputs.sizes[0][0], inputs.sizes[0][1]);
            if inputs.dtypes[0] == TensorElementType::Float16 && !fp16 {
                panic!(
                    "Dtype mismatch! Expected: Float32, got: {:?}. You should use `--fp16`",
                    inputs.dtypes[0]
                );
            }
            // dynamic shape: input_tensor_1:dim_1xdim_2x...,input_tensor_2:dim_3xdim_4x...,...
            let mut opt_string = String::new();
            let mut min_string = String::new();
            let mut max_string = String::new();
            for name in inputs.names.iter() {
                let s_opt = format!("{}:{}x3x{}x{},", name, batch.opt, height, width);
                let s_min = format!("{}:{}x3x{}x{},", name, batch.min, height, width);
                let s_max = format!("{}:{}x3x{}x{},", name, batch.max, height, width);
                opt_string.push_str(s_opt.as_str());
                min_string.push_str(s_min.as_str());
                max_string.push_str(s_max.as_str());
            }
            let _ = opt_string.pop();
            let _ = min_string.pop();
            let _ = max_string.pop();

            let trt_provider = trt_provider
                .with_profile_opt_shapes(opt_string)
                .with_profile_min_shapes(min_string)
                .with_profile_max_shapes(max_string)
                .with_fp16(fp16)
                .with_timing_cache(true);
            (
                OrtEP::Trt(device_id),
                ExecutionProviderDispatch::from(trt_provider),
            )
        } else {
            println!("> TensorRT is not available! Try using CUDA...");
            Self::set_ep_cuda(device_id)
        }
    }

    pub fn fetch_from_metadata(&self, key: &str) -> Option<String> {
        // fetch value from onnx model file by key
        match self.session.metadata() {
            Err(_) => None,
            Ok(metadata) => match metadata.custom(key) {
                Err(_) => None,
                Ok(value) => value,
            },
        }
    }

    pub fn run(&self, xs: Array<f32, IxDyn>, profile: bool) -> Result<Vec<Array<f32, IxDyn>>> {
        // ORT inference
        match self.dtype() {
            TensorElementType::Float16 => self.run_fp16(xs, profile),
            TensorElementType::Float32 => self.run_fp32(xs, profile),
            _ => todo!(),
        }
    }

    pub fn run_fp16(&self, xs: Array<f32, IxDyn>, profile: bool) -> Result<Vec<Array<f32, IxDyn>>> {
        // f32->f16
        let t = std::time::Instant::now();
        let xs = xs.mapv(f16::from_f32);
        if profile {
            println!("[ORT f32->f16]: {:?}", t.elapsed());
        }

        // h2d
        let t = std::time::Instant::now();
        let xs = CowArray::from(xs);
        if profile {
            println!("[ORT H2D]: {:?}", t.elapsed());
        }

        // run
        let t = std::time::Instant::now();
        let ys = self.session.run(ort::inputs![xs.view()]?)?;
        if profile {
            println!("[ORT Inference]: {:?}", t.elapsed());
        }

        // d2h
        Ok(ys
            .iter()
            .map(|(_k, v)| {
                // d2h
                let t = std::time::Instant::now();
                let v = v.try_extract_tensor().unwrap();
                //let v = v.try_extract::<_>().unwrap().view().clone().into_owned();
                if profile {
                    println!("[ORT D2H]: {:?}", t.elapsed());
                }

                // f16->f32
                let t_ = std::time::Instant::now();
                let v = v.mapv(f16::to_f32);
                if profile {
                    println!("[ORT f16->f32]: {:?}", t_.elapsed());
                }
                v
            })
            .collect::<Vec<Array<_, _>>>())
    }

    pub fn run_fp32(&self, xs: Array<f32, IxDyn>, profile: bool) -> Result<Vec<Array<f32, IxDyn>>> {
        // h2d
        let t = std::time::Instant::now();
        let xs = CowArray::from(xs);
        if profile {
            println!("[ORT H2D]: {:?}", t.elapsed());
        }

        // run
        let t = std::time::Instant::now();
        let ys = self.session.run(ort::inputs![xs.view()]?)?;
        if profile {
            println!("[ORT Inference]: {:?}", t.elapsed());
        }

        // d2h
        Ok(ys
            .iter()
            .map(|(_k, v)| {
                let t = std::time::Instant::now();
                let v = v.try_extract_tensor::<f32>().unwrap().into_owned();
                //let x = x.try_extract::<_>().unwrap().view().clone().into_owned();
                if profile {
                    println!("[ORT D2H]: {:?}", t.elapsed());
                }
                v
            })
            .collect::<Vec<Array<_, _>>>())
    }

    pub fn output_shapes(&self) -> Vec<Vec<i64>> {
        let mut shapes = Vec::new();
        for output in &self.session.outputs {
            if let ValueType::Tensor { ty: _, dimensions } = &output.output_type {
                let shape = dimensions.clone();
                shapes.push(shape);
            } else {
                panic!("not support data format, {} - {}", file!(), line!());
            }
        }
        shapes
    }

    pub fn output_dtypes(&self) -> Vec<TensorElementType> {
        let mut dtypes = Vec::new();
        for output in &self.session.outputs {
            if let ValueType::Tensor { ty, dimensions: _ } = &output.output_type {
                dtypes.push(ty.clone());
            } else {
                panic!("not support data format, {} - {}", file!(), line!());
            }
        }
        dtypes
    }

    pub fn input_shapes(&self) -> &Vec<Vec<i64>> {
        &self.inputs.shapes
    }

    pub fn input_names(&self) -> &Vec<String> {
        &self.inputs.names
    }

    pub fn input_dtypes(&self) -> &Vec<TensorElementType> {
        &self.inputs.dtypes
    }

    pub fn dtype(&self) -> TensorElementType {
        self.input_dtypes()[0]
    }

    pub fn height(&self) -> u32 {
        self.inputs.sizes[0][0]
    }

    pub fn width(&self) -> u32 {
        self.inputs.sizes[0][1]
    }

    pub fn is_height_dynamic(&self) -> bool {
        self.input_shapes()[0][2] == -1
    }

    pub fn is_width_dynamic(&self) -> bool {
        self.input_shapes()[0][3] == -1
    }

    pub fn batch(&self) -> u32 {
        self.batch.opt
    }

    pub fn is_batch_dynamic(&self) -> bool {
        self.input_shapes()[0][0] == -1
    }

    pub fn ep(&self) -> &OrtEP {
        &self.ep
    }

    pub fn task(&self) -> YOLOTask {
        self.task.clone()
    }

    pub fn names(&self) -> Option<Vec<String>> {
        // class names, metadata parsing
        // String format: `{0: 'person', 1: 'bicycle', 2: 'sports ball', ..., 27: "yellow_lady's_slipper"}`
        match self.fetch_from_metadata("names") {
            Some(names) => {
                let re = Regex::new(r#"(['"])([-()\w '"]+)(['"])"#).unwrap();
                let mut names_ = vec![];
                for (_, [_, name, _]) in re.captures_iter(&names).map(|x| x.extract()) {
                    names_.push(name.to_string());
                }
                Some(names_)
            }
            None => None,
        }
    }

    pub fn nk(&self) -> Option<u32> {
        // num_keypoints, metadata parsing: String `nk` in onnx model: `[17, 3]`
        match self.fetch_from_metadata("kpt_shape") {
            None => None,
            Some(kpt_string) => {
                let re = Regex::new(r"([0-9]+), ([0-9]+)").unwrap();
                let caps = re.captures(&kpt_string).unwrap();
                Some(caps.get(1).unwrap().as_str().parse::<u32>().unwrap())
            }
        }
    }

    pub fn nc(&self) -> Option<u32> {
        // num_classes
        match self.names() {
            // by names
            Some(names) => Some(names.len() as u32),
            None => match self.task() {
                // by task calculation
                YOLOTask::Classify => Some(self.output_shapes()[0][1] as u32),
                YOLOTask::Detect => {
                    if self.output_shapes()[0][1] == -1 {
                        None
                    } else {
                        // cxywhclss
                        Some(self.output_shapes()[0][1] as u32 - 4)
                    }
                }
                YOLOTask::Pose => {
                    match self.nk() {
                        None => None,
                        Some(nk) => {
                            if self.output_shapes()[0][1] == -1 {
                                None
                            } else {
                                // cxywhclss3*kpt
                                Some(self.output_shapes()[0][1] as u32 - 4 - 3 * nk)
                            }
                        }
                    }
                }
                YOLOTask::Segment => {
                    if self.output_shapes()[0][1] == -1 {
                        None
                    } else {
                        // cxywhclssnm
                        Some((self.output_shapes()[0][1] - self.output_shapes()[1][1]) as u32 - 4)
                    }
                }
            },
        }
    }

    pub fn nm(&self) -> Option<u32> {
        // num_masks
        match self.task() {
            YOLOTask::Segment => Some(self.output_shapes()[1][1] as u32),
            _ => None,
        }
    }

    pub fn na(&self) -> Option<u32> {
        // num_anchors
        match self.task() {
            YOLOTask::Segment | YOLOTask::Detect | YOLOTask::Pose => {
                if self.output_shapes()[0][2] == -1 {
                    None
                } else {
                    Some(self.output_shapes()[0][2] as u32)
                }
            }
            _ => None,
        }
    }

    pub fn author(&self) -> Option<String> {
        self.fetch_from_metadata("author")
    }

    pub fn version(&self) -> Option<String> {
        self.fetch_from_metadata("version")
    }
}
