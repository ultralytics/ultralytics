---
comments: true
description: Comprehensive Guide to Understanding and Creating Line Graphs, Bar Plots, and Pie Charts
keywords: Analytics, Data Visualization, Line Graphs, Bar Plots, Pie Charts, Quickstart Guide, Data Analysis, Python, Visualization Tools
---

# Analytics using Ultralytics YOLOv8 📊

## Introduction

This guide provides a comprehensive overview of three fundamental types of data visualizations: line graphs, bar plots, and pie charts. Each section includes step-by-step instructions and code snippets on how to create these visualizations using Python.

###  Graphs

- Line graphs are ideal for tracking changes over short and long periods and for comparing changes for multiple groups over the same period. 
- Bar plots, on the other hand, are suitable for comparing quantities across different categories and showing relationships between a category and its numerical value. 
- Lastly, pie charts are effective for illustrating proportions among categories and showing parts of a whole.

!!! Analytics "Analytics Examples"
    
    === "Line Graph"

    ```python
    import cv2
    from ultralytics import YOLO, solutions
    
    model = YOLO("yolov8s.pt")
    
    cap = cv2.VideoCapture("Path/to/video/file.mp4")
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter('line_plot.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
    
    line = solutions.Analytics(type="line", 
                                writer=out, 
                                im0_shape=(w, h), 
                                view_img=True)
    total_counts = 0
    frame_count = 0
    
    while cap.isOpened():
        success, frame = cap.read()
    
        if success:
            frame_count += 1
            results = model.track(frame, persist=True, verbose=True)
    
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu()
                for box in boxes:
                    total_counts += 1
    
            line.update_line_graph(frame_count, total_counts)
    
            total_counts = 0
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    ```

    === "Pie Chart"

        ```python
        import cv2
        from ultralytics import YOLO, solutions
        
        model = YOLO("yolov8s.pt")
        
        cap = cv2.VideoCapture("Path/to/video/file.mp4")
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        
        out = cv2.VideoWriter('pie_chart.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
        
        analytics = solutions.Analytics(type="pie", writer=out, im0_shape=(w, h), view_img=True)
        
        clswise_count = {}
        
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                results = model.track(frame, persist=True, verbose=True)
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu()
                    for box in boxes:
                        if model.names[int(cls)] in clswise_count:
                            clswise_count[model.names[int(cls)]] += 1
                        else:
                            clswise_count[model.names[int(cls)]] = 1
        
                    analytics.update_pie_chart(clswise_count)
                    clswise_count = {}
        
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        ```
    
    === "Bar Plot"

        ```python
        import cv2
        from ultralytics import YOLO, solutions
        
        model = YOLO("yolov8s.pt")
        
        cap = cv2.VideoCapture("Path/to/video/file.mp4")
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        
        out = cv2.VideoWriter('bar_plot.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
        
        analytics = solutions.Analytics(type="bar", writer=out, im0_shape=(w, h), view_img=True)
        
        clswise_count = {}
        
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                results = model.track(frame, persist=True, verbose=True)
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu()
                    for box in boxes:
                        if model.names[int(cls)] in clswise_count:
                            clswise_count[model.names[int(cls)]] += 1
                        else:
                            clswise_count[model.names[int(cls)]] = 1
        
                    analytics.update_bar_graph(clswise_count)
                    clswise_count = {}
        
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        ```

## Conclusion

Understanding when and how to use different types of visualizations is crucial for effective data analysis. Line graphs, bar plots, and pie charts are fundamental tools that can help you convey your data's story more clearly and effectively.