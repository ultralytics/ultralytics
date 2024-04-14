// Ultralytics YOLO 🚀, AGPL-3.0 license
// Documentation: https://docs.ultralytics.com/analytics/index.md
// Example usage: yolo dashboard "path/to/custom_data.yaml"

document.addEventListener("DOMContentLoaded", function()
{
    let chart = null;


    const fetchDataAndUpdateChart = () =>   // Function to fetch data and update the chart
    {
        fetch('/total_detections').then(response => response.json()).then(data =>
        {
            const classWiseDict = data.class_wise_dict;
            const labels = Object.keys(classWiseDict);
            const dataValues = Object.values(classWiseDict);
            const backgroundColors = labels.map(className => generateColor(className));

            if (chart)
            {
              chart.destroy();
            }

            const ctx = document.getElementById('barChart').getContext('2d');
            chart = new Chart(ctx,
            {
              type: 'bar',
              data:
              {
                labels: labels,
                datasets:
                [
                    {
                      label: 'Class-wise Data',
                      data: dataValues,
                      backgroundColor: backgroundColors,
                      borderColor: 'rgba(54, 162, 235, 1)',
                      borderWidth: 1
                    }
                ]
              },
              options:
              {
                responsive: true,
                scales:
                {
                  y:
                  {
                    beginAtZero: true
                  }
                }
              }
            });
        }).catch(error => console.error('Error fetching data:', error));};

    fetchDataAndUpdateChart();

    setInterval(fetchDataAndUpdateChart, 5000); // Refresh barchart every 5 second
})


var ctxLine = document.getElementById('lineChart').getContext('2d');
var lineChart = new Chart(ctxLine,
{
    type: 'line',
    data:
    {
        labels: [],
        datasets:
        [
            {
                label: 'Objects Count',
                data: [],
                borderColor: 'rgba(255, 0, 255, 255)',
                backgroundColor: 'rgba(17, 31, 104, 255)',
                tension: 0.1
            }
        ]
    },

    options:
    {
        scales:
        {
            y:
            {
                beginAtZero: true,
                ticks:
                {
                    color: 'rgba(17, 31, 104, 1)' // Set label color to white
                }
            },

            x:
            {
                ticks:
                {
                    color: 'rgba(17, 31, 104, 1)' // Set label color to white
                }
            }
        },

        plugins:
        {
            legend:
            {
                labels:
                {
                    color: 'rgba(17, 31, 104, 1)' // Set legend label color to white
                }
            }
        }
    }
});


function updateDetection()
{
    fetch('/total_detections').then(response => response.json()).then(data =>
    {
        document.getElementById('detectionCount').innerText = data.total_detections;
        document.getElementById('fps').innerText = data.fps;
        document.getElementById('incounts').innerText = data.in_counts;
        document.getElementById('outcounts').innerText = data.out_counts;
        document.getElementById('mode').innerText = data.mode;

        var time = new Date().toLocaleTimeString();
        lineChart.data.labels.push(time);
        lineChart.data.datasets[0].data.push(data.total_detections);
        lineChart.update();
    }
    ).catch(error => console.error('Error fetching detection count:', error));


    if (lineChart.data.labels.length > 45)  // Remove the first point every 45 seconds
    {
        lineChart.data.labels.splice(0, 1);
        lineChart.data.datasets[0].data.splice(0, 1);
    }
}

// Function to generate a deterministic color based on class name
const generateColor = className =>
{
    let hash = 0;
    for (let i = 0; i < className.length; i++)
    {
        hash = className.charCodeAt(i) + ((hash << 5) - hash);
    }
    const c = (hash & 0x00FFFFFF).toString(16).toUpperCase();
    return '#' + '00000'.substring(0, 6 - c.length) + c;
};

setInterval(updateDetection, 1000);
