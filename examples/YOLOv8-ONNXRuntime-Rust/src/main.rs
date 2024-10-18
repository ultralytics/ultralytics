use clap::Parser;

use yolov8_rs::{Args, YOLOv8};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // 1. load image
    let x = image::ImageReader::open(&args.source)?
        .with_guessed_format()?
        .decode()?;

    // 2. model support dynamic batch inference, so input should be a Vec
    let xs = vec![x];

    // You can test `--batch 2` with this
    // let xs = vec![x.clone(), x];

    // 3. build yolov8 model
    let mut model = YOLOv8::new(args)?;
    model.summary(); // model info

    // 4. run
    let ys = model.run(&xs)?;
    println!("{:?}", ys);

    Ok(())
}
