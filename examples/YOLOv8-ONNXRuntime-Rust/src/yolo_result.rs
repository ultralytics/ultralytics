// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

use ndarray::{Array, Axis, IxDyn};

#[derive(Clone, PartialEq, Default)]
pub struct YOLOResult {
    // YOLO tasks results of an image
    pub probs: Option<Embedding>,
    pub bboxes: Option<Vec<Bbox>>,
    pub keypoints: Option<Vec<Vec<Point2>>>,
    pub masks: Option<Vec<Vec<u8>>>,
}

impl std::fmt::Debug for YOLOResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("YOLOResult")
            .field(
                "Probs(top5)",
                &format_args!("{:?}", self.probs().map(|probs| probs.topk(5))),
            )
            .field("Bboxes", &self.bboxes)
            .field("Keypoints", &self.keypoints)
            .field(
                "Masks",
                &format_args!("{:?}", self.masks().map(|masks| masks.len())),
            )
            .finish()
    }
}

impl YOLOResult {
    pub fn new(
        probs: Option<Embedding>,
        bboxes: Option<Vec<Bbox>>,
        keypoints: Option<Vec<Vec<Point2>>>,
        masks: Option<Vec<Vec<u8>>>,
    ) -> Self {
        Self {
            probs,
            bboxes,
            keypoints,
            masks,
        }
    }

    pub fn probs(&self) -> Option<&Embedding> {
        self.probs.as_ref()
    }

    pub fn keypoints(&self) -> Option<&Vec<Vec<Point2>>> {
        self.keypoints.as_ref()
    }

    pub fn masks(&self) -> Option<&Vec<Vec<u8>>> {
        self.masks.as_ref()
    }

    pub fn bboxes(&self) -> Option<&Vec<Bbox>> {
        self.bboxes.as_ref()
    }

    pub fn bboxes_mut(&mut self) -> Option<&mut Vec<Bbox>> {
        self.bboxes.as_mut()
    }
}

#[derive(Debug, PartialEq, Clone, Default)]
pub struct Point2 {
    // A point2d with x, y, conf
    x: f32,
    y: f32,
    confidence: f32,
}

impl Point2 {
    pub fn new_with_conf(x: f32, y: f32, confidence: f32) -> Self {
        Self { x, y, confidence }
    }

    pub fn new(x: f32, y: f32) -> Self {
        Self {
            x,
            y,
            ..Default::default()
        }
    }

    pub fn x(&self) -> f32 {
        self.x
    }

    pub fn y(&self) -> f32 {
        self.y
    }

    pub fn confidence(&self) -> f32 {
        self.confidence
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Embedding {
    // An float32 n-dims tensor
    data: Array<f32, IxDyn>,
}

impl Embedding {
    pub fn new(data: Array<f32, IxDyn>) -> Self {
        Self { data }
    }

    pub fn data(&self) -> &Array<f32, IxDyn> {
        &self.data
    }

    pub fn topk(&self, k: usize) -> Vec<(usize, f32)> {
        let mut probs = self
            .data
            .iter()
            .enumerate()
            .map(|(a, b)| (a, *b))
            .collect::<Vec<_>>();
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut topk = Vec::new();
        for &(id, confidence) in probs.iter().take(k) {
            topk.push((id, confidence));
        }
        topk
    }

    pub fn norm(&self) -> Array<f32, IxDyn> {
        let std_ = self.data.mapv(|x| x * x).sum_axis(Axis(0)).mapv(f32::sqrt);
        self.data.clone() / std_
    }

    pub fn top1(&self) -> (usize, f32) {
        self.topk(1)[0]
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Bbox {
    // a bounding box around an object
    xmin: f32,
    ymin: f32,
    width: f32,
    height: f32,
    id: usize,
    confidence: f32,
}

impl Bbox {
    pub fn new_from_xywh(xmin: f32, ymin: f32, width: f32, height: f32) -> Self {
        Self {
            xmin,
            ymin,
            width,
            height,
            ..Default::default()
        }
    }

    pub fn new(xmin: f32, ymin: f32, width: f32, height: f32, id: usize, confidence: f32) -> Self {
        Self {
            xmin,
            ymin,
            width,
            height,
            id,
            confidence,
        }
    }

    pub fn width(&self) -> f32 {
        self.width
    }

    pub fn height(&self) -> f32 {
        self.height
    }

    pub fn xmin(&self) -> f32 {
        self.xmin
    }

    pub fn ymin(&self) -> f32 {
        self.ymin
    }

    pub fn xmax(&self) -> f32 {
        self.xmin + self.width
    }

    pub fn ymax(&self) -> f32 {
        self.ymin + self.height
    }

    pub fn tl(&self) -> Point2 {
        Point2::new(self.xmin, self.ymin)
    }

    pub fn br(&self) -> Point2 {
        Point2::new(self.xmax(), self.ymax())
    }

    pub fn cxcy(&self) -> Point2 {
        Point2::new(self.xmin + self.width / 2., self.ymin + self.height / 2.)
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    pub fn area(&self) -> f32 {
        self.width * self.height
    }

    pub fn intersection_area(&self, another: &Bbox) -> f32 {
        let l = self.xmin.max(another.xmin);
        let r = (self.xmin + self.width).min(another.xmin + another.width);
        let t = self.ymin.max(another.ymin);
        let b = (self.ymin + self.height).min(another.ymin + another.height);
        (r - l + 1.).max(0.) * (b - t + 1.).max(0.)
    }

    pub fn union(&self, another: &Bbox) -> f32 {
        self.area() + another.area() - self.intersection_area(another)
    }

    pub fn iou(&self, another: &Bbox) -> f32 {
        self.intersection_area(another) / self.union(another)
    }
}
