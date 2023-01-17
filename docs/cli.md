If you want to train, validate or run inference on models and don't need to make any modifications to the code, using
YOLO command line interface is the easiest way to get started.

!!! tip "Syntax"

    ```bash
    yolo task=detect    mode=train    model=yolov8n.yaml    epochs=1 ...
                ...           ...           ...
              segment        predict        yolov8n-seg.pt
              classify        val           yolov8n-cls.pt
    ```

The experiment arguments can be overridden directly by pass `arg=val` covered in the next section. You can run any
supported task by setting `task` and `mode` in cli.
=== "Training"

    |                  | `task`     | snippet                                                    |
    |------------------|------------|------------------------------------------------------------|
    | Detection        | `detect`   | <pre><code>yolo task=detect mode=train       </code></pre> |
    | Instance Segment | `segment`  | <pre><code>yolo task=segment mode=train      </code></pre> |
    | Classification   | `classify` | <pre><code>yolo task=classify mode=train    </code></pre>  |

=== "Prediction"

    |                  | `task`     | snippet                                                      |
    |------------------|------------|--------------------------------------------------------------|
    | Detection        | `detect`   | <pre><code>yolo task=detect mode=predict       </code></pre> |
    | Instance Segment | `segment`  | <pre><code>yolo task=segment mode=predict     </code></pre>  |
    | Classification   | `classify` | <pre><code>yolo task=classify mode=predict    </code></pre>  |

=== "Validation"

    |                  | `task`     | snippet                                                   |
    |------------------|------------|-----------------------------------------------------------|
    | Detection        | `detect`   | <pre><code>yolo task=detect mode=val        </code></pre> |
    | Instance Segment | `segment`  | <pre><code>yolo task=segment mode=val       </code></pre> |
    | Classification   | `classify` | <pre><code>yolo task=classify mode=val      </code></pre> |

!!! note ""

    <b>Note:</b> The arguments don't require `'--'` prefix. These are reserved for special commands covered later

---

## Overriding default config arguments

All global default arguments can be overriden by simply passing them as arguments in the cli.

!!! tip ""

    === "Syntax"
        ```bash
        yolo task= ... mode= ... {++ arg=val ++}
        ```

    === "Example"
        Perform detection training for `10 epochs` with `learning_rate` of `0.01`
        ```bash
        yolo task=detect mode=train {++ epochs=10 lr0=0.01 ++}
        ```

---

## Overriding default config file

You can override config file entirely by passing a new file. You can create a copy of default config file in your
current working dir as follows:

```bash
yolo task=init
```

You can then use `cfg=name.yaml` command to pass the new config file

```bash
yolo cfg=default.yaml
```

??? example

    === "Command"
        ```bash
        yolo task=init
        yolo cfg=default.yaml
        ```
    === "Results"
        TODO: add terminal output


