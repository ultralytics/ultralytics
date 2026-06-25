---
title: CV Data Collection and Annotation Guide
comments: true
description: Learn data collection and annotation for computer vision: set up classes, source unbiased data, choose annotation types and formats, and run quality control.
keywords: What is Data Annotation, Data Annotation Tools, Annotating Data, Avoiding Bias in Data Collection, Ethical Data Collection, Annotation Strategies
---

# Data Collection and Annotation Strategies for Computer Vision

Data collection and annotation are the two foundational steps of every [computer vision project](./steps-of-a-cv-project.md): you gather representative images or video, then label them so a model can learn from them. The quality of this data directly determines model performance, which is why class definition, unbiased sourcing, and consistent annotation matter before any training begins.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/iBk6S-PHwS0"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Build Effective Data Collection and Annotation Strategies for Computer Vision 🚀
</p>

This guide covers [setting up classes and collecting data](#setting-up-classes-and-collecting-data), [what data annotation is](#what-is-data-annotation) along with the annotation types and formats to choose from, and [efficient labeling strategies](#efficient-data-labeling-strategies) — every decision aligned with [your project's goals](./defining-project-goals.md).

## Setting Up Classes and Collecting Data

Collecting images and video for a computer vision project comes down to three decisions: how many classes to define, where to source the data, and how to keep the dataset free of bias.

### Choosing the Right Classes for Your Project

One of the first questions when starting a computer vision project is how many classes to include. You need to determine the class membership, which involves the different categories or labels that you want your model to recognize and differentiate. The number of classes should be determined by the specific goals of your project.

For example, if you want to monitor traffic, your classes might include "car," "truck," "bus," "motorcycle," and "bicycle." On the other hand, for tracking items in a store, your classes could be "fruits," "vegetables," "beverages," and "snacks." Defining classes based on your project goals helps keep your dataset relevant and focused.

When you define your classes, another important distinction to make is whether to choose coarse or fine class counts. 'Count' refers to the number of distinct classes you are interested in. This decision influences the granularity of your data and the complexity of your model. Here are the considerations for each approach:

- **Coarse Class-Count**: These are broader, more inclusive categories, such as "vehicle" and "non-vehicle." They simplify annotation and require fewer computational resources but provide less detailed information, potentially limiting the model's effectiveness in complex scenarios.
- **Fine Class-Count**: More categories with finer distinctions, such as "sedan," "SUV," "pickup truck," and "motorcycle." They capture more detailed information, improving model accuracy and performance. However, they are more time-consuming and labor-intensive to annotate and require more computational resources.

Starting with more specific classes can be very helpful, especially in complex projects where details are important. More specific classes let you collect more detailed data, gain deeper insights, and establish clearer distinctions between categories. Not only does it improve the accuracy of the model, but it also makes it easier to adjust the model later if needed, saving both time and resources.

### Sources of Data

You can use public datasets or gather your own custom data. Public datasets like those on [Kaggle](https://www.kaggle.com/datasets) and [Google Dataset Search Engine](https://datasetsearch.research.google.com/) offer well-annotated, standardized data, making them great starting points for training and validating models.

Custom data collection, on the other hand, allows you to customize your dataset to your specific needs. You might capture images and videos with cameras or drones, scrape the web for images, or use existing internal data from your organization. Custom data gives you more control over its quality and relevance. Combining both public and custom data sources helps create a diverse and comprehensive dataset.

### Avoiding Bias in Data Collection

Bias occurs when certain groups or scenarios are underrepresented or overrepresented in your dataset. It leads to a model that performs well on some data but poorly on others. It's crucial to avoid [bias in AI](https://www.ultralytics.com/glossary/bias-in-ai) so that your computer vision model can perform well in a variety of scenarios.

Here is how you can avoid bias while collecting data:

- **Diverse Sources**: Collect data from many sources to capture different perspectives and scenarios.
- **Balanced Representation**: Include balanced representation from all relevant groups. For example, consider different ages, genders, and ethnicities.
- **Continuous Monitoring**: Regularly review and update your dataset to identify and address any emerging biases.
- **Bias Mitigation Techniques**: Use methods like oversampling underrepresented classes, [data augmentation](https://www.ultralytics.com/glossary/data-augmentation), and fairness-aware algorithms.

Following these practices helps create a more robust and fair model that can generalize well in real-world applications.

## What is Data Annotation?

Data annotation is the process of labeling data to make it usable for training [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) models. In computer vision, this means labeling images or videos with the information that a model needs to learn from. Without properly annotated data, models cannot accurately learn the relationships between inputs and outputs.

### Types of Data Annotation

Depending on the specific requirements of a [computer vision task](../tasks/index.md), there are different types of data annotation. Here are some examples:

- **Bounding Boxes**: Rectangular boxes drawn around objects in an image, used primarily for object detection tasks. These boxes are defined by their top-left and bottom-right coordinates.
- **Polygons**: Detailed outlines for objects, allowing for more precise annotation than bounding boxes. Polygons are used in tasks like [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation), where the shape of the object is important.
- **Masks**: Binary masks where each pixel is either part of an object or the background. Masks are used in [semantic segmentation](https://www.ultralytics.com/glossary/semantic-segmentation) tasks to provide pixel-level detail.
- **Keypoints**: Specific points marked within an image to identify locations of interest. Keypoints are used in tasks like [pose estimation](../tasks/pose.md) and facial landmark detection.

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/types-of-data-annotation.avif" alt="Data annotation types including bounding boxes, polygons, and masks">
</p>

### Common Annotation Formats

After selecting a type of annotation, it's important to choose the appropriate format for storing and sharing annotations. The most common formats are:

| Format                                  | File structure            | Commonly used for                                                                                                                                                                                              |
| --------------------------------------- | ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [COCO](../datasets/detect/coco.md)      | Single JSON file          | [Object detection](https://www.ultralytics.com/glossary/object-detection), keypoint detection, stuff and [panoptic segmentation](https://www.ultralytics.com/glossary/panoptic-segmentation), image captioning |
| [Pascal VOC](../datasets/detect/voc.md) | One XML file per image    | Object detection                                                                                                                                                                                               |
| YOLO                                    | One `.txt` file per image | Object detection, segmentation, and pose                                                                                                                                                                       |

The YOLO format stores one row per object as `class x_center y_center width height`, with the box coordinates normalized to a 0–1 range and class indices starting from 0.

### Setting Annotation Guidelines

With a type of annotation and format chosen, the next step is to establish clear and objective labeling rules. These rules act as a roadmap for consistency and [accuracy](https://www.ultralytics.com/glossary/accuracy) throughout the annotation process. Key aspects of these rules include:

- **Clarity and Detail**: Make sure your instructions are clear. Use examples and illustrations to show what's expected.
- **Consistency**: Keep your annotations uniform. Set standard criteria for annotating different types of data, so all annotations follow the same rules.
- **Reducing Bias**: Stay neutral. Train yourself to be objective and minimize personal biases to ensure fair annotations.
- **Efficiency**: Work smarter, not harder. Use tools and workflows that automate repetitive tasks, making the annotation process faster and more efficient.

Regularly reviewing and updating your labeling rules will help keep your annotations accurate, consistent, and aligned with your project goals.

### Annotation Tools

A good annotation tool lets you label every type your task needs, enforces consistent guidelines, and exports labels in a training-ready format. [Ultralytics Platform](https://platform.ultralytics.com) provides a built-in [annotation editor](../platform/data/annotation.md) covering detection, instance segmentation, pose, OBB, and classification, with [SAM-powered smart annotation](https://www.ultralytics.com/annotate) that turns a single click into a mask for detection, segmentation, and OBB tasks. Because every annotation is saved in [YOLO format](../datasets/detect/index.md#ultralytics-yolo-format), your labeled dataset moves straight into [training](../modes/train.md) with no conversion step.

### Annotation Quality: Accuracy, Precision, and Outliers

Before annotating at scale, it helps to understand accuracy, [precision](https://www.ultralytics.com/glossary/precision), outliers, and quality control, so you don't label your data in a counterproductive way.

#### Understanding Accuracy and Precision

It's important to understand the difference between accuracy and precision and how it relates to annotation. Accuracy refers to how close the annotated data is to the true values. It helps us measure how closely the labels reflect real-world scenarios. Precision indicates the consistency of annotations. It checks if you are giving the same label to the same object or feature throughout the dataset. High accuracy and precision lead to better-trained models by reducing noise and improving the model's ability to generalize from the [training data](https://www.ultralytics.com/glossary/training-data).

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/example-of-precision.avif" alt="Accuracy vs precision comparison for data annotation">
</p>

#### Identifying Outliers

Outliers are data points that deviate quite a bit from other observations in the dataset. With respect to annotations, an outlier could be an incorrectly labeled image or an annotation that doesn't fit with the rest of the dataset. Outliers are concerning because they can distort the model's learning process, leading to inaccurate predictions and poor generalization.

You can use various methods to detect and correct outliers:

- **Statistical Techniques**: To detect outliers in numerical features like pixel values, [bounding box](https://www.ultralytics.com/glossary/bounding-box) coordinates, or object sizes, you can use methods such as box plots, histograms, or z-scores.
- **Visual Techniques**: To spot anomalies in categorical features like object classes, colors, or shapes, use visual methods like plotting images, labels, or heat maps.
- **Algorithmic Methods**: Use tools like clustering (e.g., K-means clustering, [DBSCAN](https://www.ultralytics.com/glossary/dbscan-density-based-spatial-clustering-of-applications-with-noise)) and [anomaly detection](https://www.ultralytics.com/glossary/anomaly-detection) algorithms to identify outliers based on data distribution patterns.

#### Quality Control of Annotated Data

Just like other technical projects, quality control is a must for annotated data. It is a good practice to regularly check annotations to make sure they are accurate and consistent. This can be done in a few different ways:

- Reviewing samples of annotated data
- Using automated tools to spot common errors
- Having another person double-check the annotations

If you are working with multiple people, consistency between different annotators is important. Good inter-annotator agreement means that the guidelines are clear and everyone is following them the same way. It keeps everyone on the same page and the annotations consistent.

While reviewing, if you find errors, correct them and update the guidelines to avoid future mistakes. Provide feedback to annotators and offer regular training to help reduce errors. Having a strong process for handling errors keeps your dataset accurate and reliable.

## Efficient Data Labeling Strategies

To make the process of data labeling smoother and more effective, consider implementing these strategies:

- **Clear Annotation Guidelines**: Provide detailed instructions with examples to ensure all annotators interpret tasks consistently. For instance, when labeling birds, specify whether to include the entire bird or just specific parts.
- **Regular Quality Checks**: Set benchmarks and use specific metrics to review work, maintaining high standards through continuous feedback.
- **Use Pre-annotation Tools**: Many modern annotation platforms offer AI-assisted pre-annotation features that can significantly speed up the process by automatically generating initial annotations that humans can then refine.
- **Implement Active Learning**: This approach prioritizes labeling the most informative samples first, which can reduce the total number of annotations needed while maintaining model performance.
- **Batch Processing**: Group similar images together for annotation to maintain consistency and improve efficiency.

These strategies can help maintain high-quality annotations while reducing the time and resources required for the labeling process.

## Share Your Thoughts with the Community

Bouncing your ideas and queries off other [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) enthusiasts can help accelerate your projects. Here are some great ways to learn, troubleshoot, and network:

### Where to Find Help and Support

- **GitHub Issues:** Visit the YOLO26 GitHub repository and use the [Issues tab](https://github.com/ultralytics/ultralytics/issues) to raise questions, report bugs, and suggest features. The community and maintainers are there to help with any issues you face.
- **Ultralytics Discord Server:** Join the [Ultralytics Discord server](https://discord.com/invite/ultralytics) to connect with other users and developers, get support, share knowledge, and brainstorm ideas.

### Official Documentation

- **Ultralytics YOLO26 Documentation:** Refer to the [official YOLO26 documentation](./index.md) for thorough guides and valuable insights on numerous computer vision tasks and projects.

## Conclusion

Collecting diverse, unbiased data and annotating it consistently with the right tools is the foundation of a reliable computer vision model. With your dataset collected and labeled, continue to the [steps of a computer vision project](./steps-of-a-cv-project.md) guide to move into training and evaluation.

## FAQ

### What is the best way to avoid bias in data collection for computer vision projects?

To minimize bias, collect data from diverse sources, ensure balanced representation across all relevant groups (such as different ages, genders, and ethnicities), regularly review and update your dataset to catch emerging biases, and apply mitigation techniques like oversampling underrepresented classes, data augmentation, and fairness-aware algorithms. Avoiding bias this way keeps your computer vision model performing well across varied real-world scenarios and improves its generalization capability.

### How can I ensure high consistency and accuracy in data annotation?

Establish clear, objective labeling guidelines with detailed instructions, examples, and illustrations, then apply them uniformly across all data types so every annotation follows the same rules. Train annotators to stay neutral to reduce personal bias, review and update the guidelines regularly, and use automated consistency checks plus inter-annotator feedback to keep accuracy high and aligned with your project goals.

### How many images do I need for training Ultralytics YOLO models?

A few hundred annotated objects per class is enough to start experimenting with [transfer learning](https://www.ultralytics.com/glossary/transfer-learning), but for reliable real-world performance Ultralytics recommends [at least 1,500 images and 10,000 labeled instances per class](../yolov5/tutorials/tips-for-best-training-results.md). Pair a sufficiently large dataset with a reasonable training schedule — [around 300 epochs](model-training-tips.md#the-number-of-epochs-to-train-for) is a common starting point, reduced if the model overfits early — and keep your annotations rigorous and aligned with your project's specific goals. Explore detailed training strategies in the [YOLO26 training guide](../modes/train.md).

### Does Ultralytics provide a data annotation tool?

Yes. [Ultralytics Platform](https://platform.ultralytics.com) includes a built-in [annotation editor](../platform/data/annotation.md) that supports bounding boxes, polygons, keypoints, oriented boxes, and classification labels in a single workspace. [SAM-powered smart annotation](https://www.ultralytics.com/annotate) speeds up labeling for detection, segmentation, and OBB tasks by generating masks from a single click, and every annotation is stored in [YOLO format](../datasets/detect/index.md#ultralytics-yolo-format), ready for [training](../modes/train.md).

### What types of data annotation are commonly used in computer vision?

The most common data annotation types in computer vision are bounding boxes, polygons, masks, and keypoints, each suited to a different task:

- **Bounding Boxes**: Used primarily for object detection, these are rectangular boxes around objects in an image.
- **Polygons**: Provide more precise object outlines suitable for instance segmentation tasks.
- **Masks**: Offer pixel-level detail, used in semantic segmentation to differentiate objects from the background.
- **Keypoints**: Identify specific points of interest within an image, useful for tasks like pose estimation and facial landmark detection.

Selecting the appropriate annotation type depends on your project's requirements. Learn more about how to implement these annotations and their formats in our [data annotation guide](#what-is-data-annotation).
