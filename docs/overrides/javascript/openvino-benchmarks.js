// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// OpenVINO models chart -----------------------------------------------------------------------------------------------
const data = {
  YOLO11n: [
    {
      precision: "FP32",
      format: "pytorch",
      device: "cpu",
      inference_speed: 42.59867649670923,
      mAP: 0.6316187870372707,
    },
    {
      precision: "FP32",
      format: "openvino",
      device: "intel:cpu",
      inference_speed: 30.699816255946644,
      mAP: 0.6082262589758726,
    },
    {
      precision: "FP32",
      format: "openvino",
      device: "intel:gpu",
      inference_speed: 20.07842450257158,
      mAP: 0.6112624032558861,
    },
    {
      precision: "FP32",
      format: "openvino",
      device: "intel:npu",
      inference_speed: 10.810200252308277,
      mAP: 0.6096821695067082,
    },
    {
      precision: "INT8",
      format: "openvino",
      device: "intel:cpu",
      inference_speed: 22.203564003575593,
      mAP: 0.6256298656891056,
    },
    {
      precision: "INT8",
      format: "openvino",
      device: "intel:gpu",
      inference_speed: 14.101927750743926,
      mAP: 0.6219334302000313,
    },
    {
      precision: "INT8",
      format: "openvino",
      device: "intel:npu",
      inference_speed: 12.841380997997476,
      mAP: 0.6380305732484748,
    },
    {
      precision: "FP16",
      format: "openvino",
      device: "intel:cpu",
      inference_speed: 18.694250247790478,
      mAP: 0.6256298656891056,
    },
    {
      precision: "FP16",
      format: "openvino",
      device: "intel:gpu",
      inference_speed: 15.932834001432639,
      mAP: 0.6219334302000313,
    },
    {
      precision: "FP16",
      format: "openvino",
      device: "intel:npu",
      inference_speed: 13.8224114998593,
      mAP: 0.6380305732484748,
    },
  ],
  YOLO11s: [
    {
      precision: "FP32",
      format: "pytorch",
      device: "cpu",
      inference_speed: 104.23563825315796,
      mAP: 0.7469595906029659,
    },
    {
      precision: "FP32",
      format: "openvino",
      device: "intel:cpu",
      inference_speed: 80.54260024800897,
      mAP: 0.7400250985774158,
    },
    {
      precision: "FP32",
      format: "openvino",
      device: "intel:gpu",
      inference_speed: 18.115033250069246,
      mAP: 0.7414403631689955,
    },
    {
      precision: "FP32",
      format: "openvino",
      device: "intel:npu",
      inference_speed: 14.644150502135744,
      mAP: 0.7430834884512186,
    },
    {
      precision: "INT8",
      format: "openvino",
      device: "intel:cpu",
      inference_speed: 41.47057525187847,
      mAP: 0.7433587822933835,
    },
    {
      precision: "INT8",
      format: "openvino",
      device: "intel:gpu",
      inference_speed: 14.769925004657125,
      mAP: 0.7362318587316888,
    },
    {
      precision: "INT8",
      format: "openvino",
      device: "intel:npu",
      inference_speed: 18.4113597497344,
      mAP: 0.7381027181630629,
    },
    {
      precision: "FP16",
      format: "openvino",
      device: "intel:cpu",
      inference_speed: 41.457037503278116,
      mAP: 0.7433587822933835,
    },
    {
      precision: "FP16",
      format: "openvino",
      device: "intel:gpu",
      inference_speed: 15.678620748076355,
      mAP: 0.7362318587316888,
    },
    {
      precision: "FP16",
      format: "openvino",
      device: "intel:npu",
      inference_speed: 17.821318499045447,
      mAP: 0.7381027181630629,
    },
  ],
  YOLO11m: [
    {
      precision: "FP32",
      format: "pytorch",
      device: "cpu",
      inference_speed: 283.9673072521691,
      mAP: 0.7717354952861968,
    },
    {
      precision: "FP32",
      format: "openvino",
      device: "intel:cpu",
      inference_speed: 239.59891100093955,
      mAP: 0.7642835928791765,
    },
    {
      precision: "FP32",
      format: "openvino",
      device: "intel:gpu",
      inference_speed: 33.73071875103051,
      mAP: 0.7642059083564425,
    },
    {
      precision: "FP32",
      format: "openvino",
      device: "intel:npu",
      inference_speed: 36.68519950224436,
      mAP: 0.7642797669353057,
    },
    {
      precision: "INT8",
      format: "openvino",
      device: "intel:cpu",
      inference_speed: 107.80364774473128,
      mAP: 0.7548775771392114,
    },
    {
      precision: "INT8",
      format: "openvino",
      device: "intel:gpu",
      inference_speed: 19.429137493716553,
      mAP: 0.744513029269246,
    },
    {
      precision: "INT8",
      format: "openvino",
      device: "intel:npu",
      inference_speed: 30.291000504803378,
      mAP: 0.7596300146219028,
    },
    {
      precision: "FP16",
      format: "openvino",
      device: "intel:cpu",
      inference_speed: 109.0995609993115,
      mAP: 0.7548775771392114,
    },
    {
      precision: "FP16",
      format: "openvino",
      device: "intel:gpu",
      inference_speed: 23.04280774842482,
      mAP: 0.744513029269246,
    },
    {
      precision: "FP16",
      format: "openvino",
      device: "intel:npu",
      inference_speed: 29.231624746898888,
      mAP: 0.7596300146219028,
    },
  ],
  YOLO11l: [
    {
      precision: "FP32",
      format: "pytorch",
      device: "cpu",
      inference_speed: 360.0904814993555,
      mAP: 0.7401581886548321,
    },
    {
      precision: "FP32",
      format: "openvino",
      device: "intel:cpu",
      inference_speed: 325.7347055005084,
      mAP: 0.7249969657566422,
    },
    {
      precision: "FP32",
      format: "openvino",
      device: "intel:gpu",
      inference_speed: 33.53390549818869,
      mAP: 0.7264455998912519,
    },
    {
      precision: "FP32",
      format: "openvino",
      device: "intel:npu",
      inference_speed: 39.95038975335774,
      mAP: 0.7264455998912519,
    },
    {
      precision: "INT8",
      format: "openvino",
      device: "intel:cpu",
      inference_speed: 137.27999624461518,
      mAP: 0.7316885039704188,
    },
    {
      precision: "INT8",
      format: "openvino",
      device: "intel:gpu",
      inference_speed: 25.628996998420916,
      mAP: 0.7290786639204325,
    },
    {
      precision: "INT8",
      format: "openvino",
      device: "intel:npu",
      inference_speed: 34.767028755595675,
      mAP: 0.7322057516951134,
    },
    {
      precision: "FP16",
      format: "openvino",
      device: "intel:cpu",
      inference_speed: 137.74089499929687,
      mAP: 0.7316885039704188,
    },
    {
      precision: "FP16",
      format: "openvino",
      device: "intel:gpu",
      inference_speed: 24.575013499998022,
      mAP: 0.7290786639204325,
    },
    {
      precision: "FP16",
      format: "openvino",
      device: "intel:npu",
      inference_speed: 38.46882299694698,
      mAP: 0.7322057516951134,
    },
  ],
  YOLO11x: [
    {
      precision: "FP32",
      format: "pytorch",
      device: "cpu",
      inference_speed: 658.8316067500273,
      mAP: 0.8467927188552188,
    },
    {
      precision: "FP32",
      format: "openvino",
      device: "intel:cpu",
      inference_speed: 743.8311755031464,
      mAP: 0.8308300958300957,
    },
    {
      precision: "FP32",
      format: "openvino",
      device: "intel:gpu",
      inference_speed: 43.291456498991465,
      mAP: 0.8308300958300957,
    },
    {
      precision: "FP32",
      format: "openvino",
      device: "intel:npu",
      inference_speed: 83.18733275154955,
      mAP: 0.8308415968951681,
    },
    {
      precision: "INT8",
      format: "openvino",
      device: "intel:cpu",
      inference_speed: 285.44155424970086,
      mAP: 0.8197542080026454,
    },
    {
      precision: "INT8",
      format: "openvino",
      device: "intel:gpu",
      inference_speed: 29.318969500309322,
      mAP: 0.8156527508142971,
    },
    {
      precision: "INT8",
      format: "openvino",
      device: "intel:npu",
      inference_speed: 59.72102974919835,
      mAP: 0.8149252252606212,
    },
    {
      precision: "FP16",
      format: "openvino",
      device: "intel:cpu",
      inference_speed: 299.16491400581435,
      mAP: 0.8197542080026454,
    },
    {
      precision: "FP16",
      format: "openvino",
      device: "intel:gpu",
      inference_speed: 32.99894874726306,
      mAP: 0.8156527508142971,
    },
    {
      precision: "FP16",
      format: "openvino",
      device: "intel:npu",
      inference_speed: 62.51684175003902,
      mAP: 0.8149252252606212,
    },
  ],
};

let openvinoComparisonChart = null; // chart variable will hold the reference to the current chart instance.

function renderOpenvinoChart(selectedDevice) {
  if (openvinoComparisonChart) openvinoComparisonChart.destroy(); // If a chart instance already exists, destroy it.

  // Define a specific color map for precision.
  const colormap = {
    PyTorch: "#10b981",
    FP32: "#3b82f6",
    INT8: "#ec4899",
    FP16: "#d97706",
  };

  const models = Object.keys(data);
  const isCPU = selectedDevice === "intel:cpu";
  const precisions = isCPU
    ? ["PyTorch", "FP32", "INT8", "FP16"]
    : ["FP32", "INT8", "FP16"];
  const groupedData = {};
  precisions.forEach((p) => (groupedData[p] = []));

  models.forEach((model) => {
    const entries = data[model];

    if (isCPU) {
      // Add PyTorch only for CPU
      const pytorch = entries.find(
        (e) =>
          e.format === "pytorch" &&
          e.precision === "FP32" &&
          e.device === "cpu",
      );
      groupedData.PyTorch.push(pytorch ? pytorch.inference_speed : null);
    }

    ["FP32", "INT8", "FP16"].forEach((p) => {
      const item = entries.find(
        (e) =>
          e.format === "openvino" &&
          e.device === selectedDevice &&
          e.precision === p,
      );
      groupedData[p].push(item ? item.inference_speed : null);
    });
  });

  const datasets = precisions.map((p) => ({
    label: p,
    data: groupedData[p],
    backgroundColor: colormap[p],
  }));
  const labels = models.map((m) => m);

  openvinoComparisonChart = new Chart(
    document.getElementById("openvinoChart").getContext("2d"),
    {
      type: "bar",
      data: {
        labels,
        datasets,
      },
      options: {
        responsive: true,
        aspectRatio: 2.5, // higher is wider
        plugins: {
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const model = models[ctx.dataIndex];
                const label = ctx.dataset.label;
                let match;

                if (label === "PyTorch") {
                  match = data[model].find((e) => e.format === "pytorch");
                } else {
                  match = data[model].find(
                    (e) =>
                      e.format === "openvino" &&
                      e.device === selectedDevice &&
                      e.precision === label,
                  );
                }

                return `${label}: ${ctx.raw?.toFixed(2)} ms, mAP: ${(match?.mAP * 100).toFixed(2)}%`;
              },
            },
          },
          legend: { labels: { color: "#fff" } },
          title: {
            display: true,
            text: "OpenVINO vs PyTorch Inference Time by Model and Precision",
            color: "#fff",
          },
        },
        scales: {
          x: {
            ticks: { color: "#fff" },
            grid: { color: "rgba(255,255,255,0.1)" },
          },
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: "Inference Time (ms)",
              color: "#fff",
            },
            ticks: { color: "#fff" },
            grid: { color: "rgba(255,255,255,0.1)" },
          },
        },
      },
    },
  );
}

document$.subscribe(() => {
  const canvas = document.getElementById("openvinoChart");
  const selector = document.getElementById("openvino-device");
  if (canvas && selector) {
    renderOpenvinoChart(selector.value); // Initial chart
    selector.addEventListener("change", () => {
      renderOpenvinoChart(selector.value); // Update on change
    });
  }
});
