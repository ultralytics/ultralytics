Inference or prediction of a task returns a list of `Results` objects. Alternatively, in the streaming mode, it returns a generator of `Results` objects which is memory efficient. Streaming mode can be enabled by passing `stream=True` in predictor's call method.

!!! example "Predict"
    === "Getting a List"
        ```python
        inputs = [img, img] # list of np arrays
        results = model(inputs) # List of Results objects
        for result in results:
            boxes = result.boxes # Boxes object for bbox outputs
            masks = result.masks # Masks object for segmenation masks outputs
            probs = result.probs # Class probabilities for classification outputs
            ...
        ```
    === "Getting a Generator"
        ```python
        inputs = [img, img] # list of np arrays
        results = model(inputs, stream="True") # Generator of Results objects
        for result in results:
            boxes = result.boxes # Boxes object for bbox outputs
            masks = result.masks # Masks object for segmenation masks outputs
            probs = result.probs # Class probabilities for classification outputs
            ...
        ```

## Working with Results
Result object consists of these component objects:

- `result.boxes` : It is an object of class `Boxes`. It has properties and methods for manipulating bboxes
- `result.masks` : It is an object of class `Masks`. It can be used to index masks or to get segment coordinates.
- `result.prob`  : It is a `Tensor` object. It contains the class probablities/logits.

Each result is composed of torch.Tensor by default, in which you can easily use following functionality:
```python
result = result.cuda()
result = result.cpu()
result = result.to("cpu")
result = result.numpy()
```
### Boxes
`Boxes` object can be used index, manipulate and convert bboxes to different formats. The box format conversion operations are cached, which means they're only calculated once per object and those values are reused for future calls.

- Indexing a `Boxes` objects returns a `Boxes` object
```python
boxes = result.boxes
box = boxes[0] # returns one box
box.xyxy 
```
- Properties and conversions
```
result.boxes.xyxy   # box with xyxy format, (N, 4)
result.boxes.xywh   # box with xywh format, (N, 4)
result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
result.boxes.conf   # confidence score, (N, 1)
result.boxes.cls    # cls, (N, 1)
```
### Masks
`Masks` object can be used index, manipulate and convert masks to segments. The segment conversion operation is cached.

```python
result.masks.masks     # masks, (N, H, W)
result.masks.segments  # bounding coordinates of masks, List[segment] * N
```

### probs
`probs` attribute of `Results` class is a `Tensor` containing class probabilities of a classification operation.
```python
result.probs     # cls prob, (num_class, )
```

Class reference documentation for `Result` module and its componenets can be found [here](reference/result.md)