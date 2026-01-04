#![allow(clippy::type_complexity)]
// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

use std::io::{Read, Write};

pub mod cli;
pub mod model;
pub mod ort_backend;
pub mod yolo_result;
pub use crate::cli::Args;
pub use crate::model::YOLOv8;
pub use crate::ort_backend::{Batch, OrtBackend, OrtConfig, OrtEP, YOLOTask};
pub use crate::yolo_result::{Bbox, Embedding, Point2, YOLOResult};

pub fn non_max_suppression(
    xs: &mut Vec<(Bbox, Option<Vec<Point2>>, Option<Vec<f32>>)>,
    iou_threshold: f32,
) {
    xs.sort_by(|b1, b2| b2.0.confidence().partial_cmp(&b1.0.confidence()).unwrap());

    let mut current_index = 0;
    for index in 0..xs.len() {
        let mut drop = false;
        for prev_index in 0..current_index {
            let iou = xs[prev_index].0.iou(&xs[index].0);
            if iou > iou_threshold {
                drop = true;
                break;
            }
        }
        if !drop {
            xs.swap(current_index, index);
            current_index += 1;
        }
    }
    xs.truncate(current_index);
}

pub fn gen_time_string(delimiter: &str) -> String {
    let offset = chrono::FixedOffset::east_opt(8 * 60 * 60).unwrap(); // Beijing
    let t_now = chrono::Utc::now().with_timezone(&offset);
    let fmt = format!(
        "%Y{}%m{}%d{}%H{}%M{}%S{}%f",
        delimiter, delimiter, delimiter, delimiter, delimiter, delimiter
    );
    t_now.format(&fmt).to_string()
}

pub const SKELETON: [(usize, usize); 16] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
];

pub fn check_font(font: &str) -> rusttype::Font<'static> {
    // check then load font

    // ultralytics font path
    let font_path_config = match dirs::config_dir() {
        Some(mut d) => {
            d.push("Ultralytics");
            d.push(font);
            d
        }
        None => panic!("Unsupported operating system. Now support Linux, MacOS, Windows."),
    };

    // current font path
    let font_path_current = std::path::PathBuf::from(font);

    // check font
    let font_path = if font_path_config.exists() {
        font_path_config
    } else if font_path_current.exists() {
        font_path_current
    } else {
        println!("Downloading font...");
        let source_url = "https://ultralytics.com/assets/Arial.ttf";
        let resp = ureq::get(source_url)
            .timeout(std::time::Duration::from_secs(500))
            .call()
            .unwrap_or_else(|err| panic!("> Failed to download font: {source_url}: {err:?}"));

        // read to buffer
        let mut buffer = vec![];
        let total_size = resp
            .header("Content-Length")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap();
        let _reader = resp
            .into_reader()
            .take(total_size)
            .read_to_end(&mut buffer)
            .unwrap();

        // save
        let _path = std::fs::File::create(font).unwrap();
        let mut writer = std::io::BufWriter::new(_path);
        writer.write_all(&buffer).unwrap();
        println!("Font saved at: {:?}", font_path_current.display());
        font_path_current
    };

    // load font
    let buffer = std::fs::read(font_path).unwrap();
    rusttype::Font::try_from_vec(buffer).unwrap()
}

use ab_glyph::FontArc;
pub fn load_font() -> FontArc {
    use std::path::Path;
    let font_path = Path::new("./font/Arial.ttf");
    match font_path.try_exists() {
        Ok(true) => {
            let buffer = std::fs::read(font_path).unwrap();
            FontArc::try_from_vec(buffer).unwrap()
        }
        Ok(false) => {
            std::fs::create_dir_all("./font").unwrap();
            println!("Downloading font...");
            let source_url = "https://ultralytics.com/assets/Arial.ttf";
            let resp = ureq::get(source_url)
                .timeout(std::time::Duration::from_secs(500))
                .call()
                .unwrap_or_else(|err| panic!("> Failed to download font: {source_url}: {err:?}"));

            // read to buffer
            let mut buffer = vec![];
            let total_size = resp
                .header("Content-Length")
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap();
            let _reader = resp
                .into_reader()
                .take(total_size)
                .read_to_end(&mut buffer)
                .unwrap();
            // save
            let mut fd = std::fs::File::create(font_path).unwrap();
            fd.write_all(&buffer).unwrap();
            println!("Font saved at: {:?}", font_path.display());
            FontArc::try_from_vec(buffer).unwrap()
        }
        Err(e) => {
            panic!("Failed to load font {}", e);
        }
    }
}
